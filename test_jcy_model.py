#!/usr/bin/env python
# coding: utf-8

# In[1]:

# CUDA_VISIBLE_DEVICES=3 python -u test_jcy_model.py test_model --type NeuralDTW_CNN_Mask_dilation_SPP6_3_HarmonicConv_Replace --name 0304_02:48:29.pth --mode covers80 > test_$(date "+%Y-%m-%d_%H:%M:%S").txt


from hpcp_loader_for_softdtw import *

from torch.utils.data import DataLoader
import models.BaseSPPNet as models
from config import DefaultConfig, opt
from tqdm import tqdm
import torch
from utility import *
import matplotlib.pyplot as plt
import json
import torch.nn.functional as F
from torch.nn.parallel.data_parallel import DataParallel
import os
import pandas as pd
import seaborn as sns
import resource
import librosa
import numpy as np

model_type = ''

@torch.no_grad()
def val_quick(softdtw, dataloader):
    softdtw.eval()
    softdtw.model.eval()
    labels = []
    temp = []
    count = -1
    for ii, (data, label) in tqdm(enumerate(dataloader)):
        labels.append(label)
    labels = torch.cat(labels, dim=0)
    N = labels.shape[0]
    dis2d = np.zeros((N, N))
    for ii, (data, label) in tqdm(enumerate(dataloader)):
        data = data.cuda()
        count += 1
        if count == 0:
            temp.append((data, count))
        else:
            for i in range(len(temp)):
                dis = softdtw.multi_compute_s(data, temp[i][0]).data.cpu().numpy()
                dis2d[temp[i][1]][count], dis2d[count][temp[i][1]] = -dis, -dis
            temp.append((data, count))

    MAP, top10, rank1 = calc_MAP(dis2d[0:labels.shape[0], 0:labels.shape[0]], labels)
    print(f'MAP:{MAP}\ttop10:{top10}\trank1:{rank1}')
    softdtw.train()
    softdtw.model.train()
    return MAP


@torch.no_grad()
def val_slow_batch(softdtw, dataloader, batch=200, is_dis=False):
    global model_type
    
    softdtw.eval()
    softdtw.model.eval()
    seqs, labels = [], []
    for ii, (data, label) in tqdm(enumerate(dataloader)):
        input = data.cuda()
        # _, seq, _ = softdtw.model(input)
        seqs.append(input)
        labels.append(label)
    seqs = torch.cat(seqs, dim=0)
    labels = torch.cat(labels, dim=0)

    N = labels.shape[0]
    query_l = [i // N for i in range(N * N)]
    ref_l = [i for i in range(N)] * N
    dis2d = np.zeros((N, N))

    N = N * N
    for st in tqdm(range(0, N, batch)):
        fi = (st + batch) if st + batch <= N else N
        query = seqs[query_l[st: fi], :, :]
        ref = seqs[ref_l[st: fi], :, :]
        if 'HarmonicConv' in model_type:
            if torch.cuda.device_count() > 1:
                s = softdtw.module.multi_compute_s(query, ref)[0].data.cpu().numpy()
            else:
                s = softdtw.multi_compute_s(query, ref)[0].data.cpu().numpy()
        else:
            if torch.cuda.device_count() > 1:
                s = softdtw.module.multi_compute_s(query, ref).data.cpu().numpy()
            else:
                s = softdtw.multi_compute_s(query, ref).data.cpu().numpy()
        for k in range(st, fi):
            i, j = query_l[k], ref_l[k]
            if is_dis:
                dis2d[i, j] = s[k - st]
            else:
                dis2d[i, j] = -s[k - st]

    MAP, top10, rank1 = calc_MAP(dis2d, labels)
    print(f'MAP: {MAP}\ttop10: {top10}\trank1: {rank1}')

    softdtw.train()
    softdtw.model.train()
    return MAP, top10, rank1

def test_model(**kwargs):
    # In[4]:
    global model_type
    
    model_type = kwargs['type']
    model_name = kwargs['name']

    print(model_type)
    print(model_name)



    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    torch.backends.cudnn.benchmark = True  # cudnn有很多种并行计算卷积的算法，
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


    model = getattr(models, model_type)(None)

    model.to('cuda:0')
    torch.multiprocessing.set_sharing_strategy('file_system')


    MAP_all, top10_all, rank1_all = 0, 0, 0
    
    if 'gdoras_test_cqt_' in kwargs['mode']:
        for number in range(1, 6):
            val_data = CQT(mode=f'gdoras_test_cqt_{number}', out_length=400)
            val_dataloader = DataLoader(val_data, 1, shuffle=False, num_workers=1)

            model.load(f'./check_points/<class \'models.BaseSPPNet.{model_type}\'>mask/{model_name}')

            print(f'-----------------test_{number}-----------------')
            MAP, top10, rank1 = val_slow_batch(model, val_dataloader)
            MAP_all += MAP
            top10_all += top10
            rank1_all += rank1
        print(f'MAP_avg: {MAP_all/5}\ttop10_avg: {top10_all/5}\trank1_avg: {rank1_all/5}')
        
    else:
        val_data = CQT(mode=kwargs['mode'], out_length=400)
        val_dataloader = DataLoader(val_data, 1, shuffle=False, num_workers=1)

        model.load(f'./check_points/<class \'models.BaseSPPNet.{model_type}\'>mask/{model_name}')

        mode = kwargs['mode']
        print(f'-----------------test_in_{mode}-----------------')
        MAP, top10, rank1 = val_slow_batch(model, val_dataloader)
        MAP_all += MAP
        top10_all += top10
        rank1_all += rank1
        print(f'MAP_avg: {MAP_all}\ttop10_avg: {top10_all}\trank1_avg: {rank1_all}')


    
if __name__ == '__main__':
    import fire

    fire.Fire()
