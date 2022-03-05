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
import sys

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
torch.backends.cudnn.benchmark = True  # cudnn有很多种并行计算卷积的算法，
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_ids = [0,1,2,3]

kwargs_global = None


def neuralwarp_train(**kwargs):
    # 多尺度图片训练 396+
    # sys.stdout.write(kwargs)
    # sys.stdout.write("Mask == 1")

    print(kwargs)
    global kwargs_global
    kwargs_global = kwargs
    
    test_cnt = 0

    with open(kwargs['params']) as f:
        params = json.load(f)
    if kwargs['manner'] == 'train':
        params['is_train'] = True
    else:
        params['is_train'] = False
    params['batch_size'] = kwargs['batch_size']
    if torch.cuda.device_count() > 1:
        sys.stdout.write("-------------------Parallel_GPU_Train--------------------------\n")
        parallel = True
    else:
        sys.stdout.write("------------------Single_GPU_Train----------------------\n")
        parallel = False
    opt.feature = 'cqt'
    opt.notes = 'SoftDTW'
    opt.model = 'SoftDTW'
    opt.batch_size = 'batch_size'
    opt._parse(kwargs) # opt.model = 你这次使用的模型名字
    model = getattr(models, opt.model)(params) # 这里的params目前没用，可以忽略

    p = 'check_points/' + model.model_name + opt.notes
    if kwargs['model'] == 'NeuralDTW_CNN_Mask_dilation_SPP6':
        f = os.path.join(p, '0819_01:03:39.pth')
    opt.load_model_path = f
    if kwargs['model'] != 'NeuralDTW' and kwargs['manner'] != 'train':
        if opt.load_latest is True: # opt.load_latest = False
            model.load_latest(opt.notes)
        elif opt.load_model_path: # 一般是None,不会执行
            print("load_model:", opt.load_model_path)
            model.load(opt.load_model_path)

    if parallel == True:
        model = DataParallel(model)
    model.to(opt.device)
    
    
    # HarmonicConv: 卷积核部分权重归零
    if 'HarmonicConv' in kwargs['model']:
        print('[HarmonicConv]: kernel weight is multiplied by a mask.')
        mask_inner = torch.zeros((57,3))
        mask_inner[[0,1,4,8,9,16,28,40,47,48,52,55,56]] = 1

        mask = torch.Tensor(mask_inner).unsqueeze(0).unsqueeze(0).cuda()
        model.model.HarmonicConv.weight.data *= mask
    
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    # step2: data
    out_length = 400
    
    train_data0 = triplet_CQT(out_length=400, is_label=kwargs['is_label'], is_random=kwargs['is_random'], datatype=kwargs['train_datatype'])
    train_data1 = triplet_CQT(out_length=400, is_label=kwargs['is_label'], is_random=kwargs['is_random'], datatype=kwargs['train_datatype'])
    train_data2 = triplet_CQT(out_length=400, is_label=kwargs['is_label'], is_random=kwargs['is_random'], datatype=kwargs['train_datatype'])
    val_data = CQT(mode=kwargs['val_datatype'], out_length=kwargs['test_length'])
    
    print('train_datatype:', kwargs['train_datatype'], ';', 'val_datatype', kwargs['val_datatype'])
    
    train_dataloader0 = DataLoader(train_data0, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    train_dataloader1 = DataLoader(train_data1, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    train_dataloader2 = DataLoader(train_data2, opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data, 1, shuffle=False, num_workers=1)
    
    if kwargs['manner'] == 'test':
        pass
    else:
        # step3: criterion and optimizer
        be = torch.nn.BCELoss()

        lr = opt.lr
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=opt.weight_decay)

        # if parallel is True:
        #     optimizer = torch.optim.Adam(model.module.parameters(), lr=lr, weight_decay=opt.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10,
                                                               verbose=True, min_lr=5e-6)
        # step4: train
        best_MAP = 0
        for epoch in range(opt.max_epoch):
            running_loss = 0
            num = 0
            for ii, ((a0, p0, n0, la0, lp0, ln0), (a1, p1, n1, la1, lp1, ln1), (a2, p2, n2, la2, lp2, ln2)) in tqdm(
                    enumerate(zip(train_dataloader0, train_dataloader1, train_dataloader2))):
                # for ii, (a2, p2, n2) in tqdm(enumerate(train_dataloader2)):
                for flag in range(3):
                    if flag == 0:
                        a, p, n, la, lp, ln = a0, p0, n0, la0, lp0, ln0
                    elif flag == 1:
                        a, p, n, la, lp, ln = a1, p1, n1, la1, lp1, ln1
                    else:
                        a, p, n, la, lp, ln = a2, p2, n2, la2, lp2, ln2
                    B, _, _, _ = a.shape
                    if kwargs["zo"] == True:
                        target = torch.cat((torch.zeros(B), torch.ones(B))).cuda()
                    else:
                        target = torch.cat((torch.ones(B), torch.zeros(B))).cuda()
                    # train model
                    a = a.requires_grad_().to(opt.device)
                    p = p.requires_grad_().to(opt.device)
                    n = n.requires_grad_().to(opt.device)

                    optimizer.zero_grad()
                    
                    # HarmonicConv: 谐波卷积 的 预测、求损失、优化 过程
                    if 'HarmonicConv' in kwargs['model']:
                        pred, sparse_a, sparse_p, sparse_n = model(a, p, n)
                        pred = pred.squeeze(1)
                        L1loss = (abs(sparse_a)).sum() + (abs(sparse_p)).sum() + (abs(sparse_n)).sum()
                        loss = be(pred, target) + float(kwargs['SparseL1loss'])*L1loss
                        loss.backward()
                        model.model.HarmonicConv.weight.grad *= mask
                        optimizer.step()
                    
                    else: # 非谐波卷积 的 预测、求损失、优化 过程
                        pred = model(a, p, n)
                        pred = pred.squeeze(1)
                        loss = be(pred, target)
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item()
                    num += a.shape[0]

                if ii % 300 == 0 and ii%3000 != 0:
                    sys.stdout.write(f'|| train_loss: {running_loss/num}\n')
                    sys.stdout.flush()
                    
                if ii % 3000 == 0 :
                    running_loss /= num
                    sys.stdout.write(f'train_loss: {running_loss}\n')

                    MAP = 0
                    sys.stdout.write(f"[{test_cnt}] gdoras test set:\n")
                    test_cnt += 1
                    MAP += val_slow_batch(model, val_dataloader, batch=100, is_dis=kwargs['zo'])
                    
                    if MAP > best_MAP:
                        best_MAP = MAP
                        sys.stdout.write('*****************BEST*****************\n')
                    if kwargs['save_model'] == True:
                        if parallel:
                            model.module.save(opt.notes, optimizer)
                        else:
                            model.save(opt.notes, optimizer)
#                         name = time.strftime('%m%d_%H:%M:%S.pth')
#                         torch.save(optimizer, f'./check_points/optim/{name}')
                    scheduler.step(running_loss)
                    running_loss = 0
                    num = 0
                    print()



# @torch.no_grad()
# def val_quick(softdtw, dataloader):
#     softdtw.eval()
#     softdtw.model.eval()
#     labels = []
#     temp = []
#     count = -1
#     for ii, (data, label) in tqdm(enumerate(dataloader)):
#         labels.append(label)
#     labels = torch.cat(labels, dim=0)
#     N = labels.shape[0]
#     dis2d = np.zeros((N, N))
#     for ii, (data, label) in tqdm(enumerate(dataloader)):
#         data = data.cuda()
#         count += 1
#         if count == 0:
#             temp.append((data, count))
#         else:
#             for i in range(len(temp)):
#                 dis = softdtw.multi_compute_s(data, temp[i][0]).data.cpu().numpy()
#                 dis2d[temp[i][1]][count], dis2d[count][temp[i][1]] = -dis, -dis
#             temp.append((data, count))

#     MAP, top10, rank1 = calc_MAP(dis2d[0:labels.shape[0], 0:labels.shape[0]], labels)
#     sys.stdout.write(f'MAP: {MAP}\ttop10: {top10}\trank1: {rank1}\n')
#     softdtw.train()
#     softdtw.model.train()
#     return MAP




@torch.no_grad()
def val_slow_batch(softdtw, dataloader, batch=200, is_dis='False'):
    
    global kwargs_global
    
    softdtw.eval()
    if torch.cuda.device_count() > 1:
        softdtw.module.model.eval()
    else:
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
    if N == 350:
        query_l = [i // 100 for i in range(100 * 100, 350 * 100)]
        ref_l = [i for i in range(100)] * 250
    else:
        query_l = [i // N for i in range(N * N)]
        ref_l = [i for i in range(N)] * N
    dis2d = np.zeros((N, N))

    N = N * N if N != 350 else 100 * 250
    for st in tqdm(range(0, N, batch)):
        fi = (st + batch) if st + batch <= N else N
        query = seqs[query_l[st: fi], :, :]
        ref = seqs[ref_l[st: fi], :, :]
        if 'HarmonicConv' in kwargs_global['model']:
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
            # sys.stdout.write(i, j)
            if is_dis:
                dis2d[i, j] = s[k - st]
            else:
                dis2d[i, j] = -s[k - st]

    if len(labels) == 350:
        MAP, top10, rank1 = calc_MAP(dis2d, labels, [100, 350])
    else:
        MAP, top10, rank1 = calc_MAP(dis2d, labels)
    sys.stdout.write(f'MAP: {MAP}\ttop10: {top10}\trank1: {rank1}\n')
    sys.stdout.flush()

    softdtw.train()
    if torch.cuda.device_count() > 1:
        softdtw.module.model.train()
    else:
        softdtw.model.train()
    return MAP



if __name__ == '__main__':
    import fire

    fire.Fire()