#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:01:07 2019
Compute the performance metrics for graphencoder model 
performance metrics includes iou,  pixelAccuracy
@author: dipu
"""

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm.auto import tqdm

import models
import opts_dml
from dataloaders.dataloader_test_2 import RICO_ComponentDataset


def main():
    opt = opts_dml.parse_opt()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    opt.use_directed_graph = True
    opt.decoder_model = 'strided'
    opt.dim =1024
    
    #model_file = 'trained_models/model_dec_strided_dim1024_TRI_ep25.pth'
    model_file = 'trained_models/model_dec_strided_dim1024_ep35.pth'
      
    data_transform = transforms.Compose([  # Not used for 25Channel_images
            transforms.Resize([255,127]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]) 
    

    model: torch.nn.Module = models.create(opt.decoder_model, opt)
    #resume = load_checkpoint(model_file)
    resume = torch.load(model_file, map_location=torch.device('cpu'))
    model.load_state_dict(resume['state_dict'])
    #model = model.cuda()
    model.eval()
    
    loader = RICO_ComponentDataset(opt, data_transform)

    feat, fnames = [], []
    for split in ("query", "gallery", "train"):
        s_feat, s_fnames = extract_features(model, loader, split=split)
        feat += s_feat
        fnames += fnames

    feat = np.concatenate(feat)

    np.save("data/embeddings/vectors.npy", feat)
    with open("data/embeddings/fnames.npy", "w") as f:
        json.dump(fnames, f)

    
def extract_features(model, loader: RICO_ComponentDataset, split: str = 'gallery'):
    epoch_done = False 
    feat = []
    fnames = [] 
    progress = tqdm(desc=split, total=len(loader.split_ix[split]))

    torch.set_grad_enabled(False)
    while epoch_done == False:
        data = loader.get_batch(split)
        sg_data = {key: torch.from_numpy(data['sg_data'][key]) for key in data['sg_data']}
        x_enc, _ = model(sg_data)
        x_enc = F.normalize(x_enc)
        outputs = x_enc.detach().cpu().numpy()
        feat.append(outputs)
        fnames += [x['id'] for x in data['infos']]

        progress.update(len(data["infos"]))
    
        if data['bounds']['wrapped']:
            epoch_done = True

    progress.close()    
    print('Extracted features from {} images from {} split'.format(len(fnames), split))
    return feat, fnames


#%%
if __name__ == '__main__':
    main()

