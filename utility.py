# -*- coding: utf-8 -*-

import numpy as np
# from scipy.spatial.distance import cdist
import time
from multiprocessing import Pool
import os, random
import torch
import torch.utils
import os
import pandas as pd
import seaborn as sns
import resource
import librosa
from hpcp_loader_for_softdtw import *
import numpy as np
from sklearn import preprocessing
from torch.autograd import Variable
import visdom
from tqdm import tqdm
# from scipy.spatial.distance import cosine
# other
def norm(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)

def calc_MAP(array2d, version, que_range=None, K=1e10):
    if que_range is not None:
        que_s, que_t = que_range[0], que_range[1]
        if que_s == 0:
            ref_s, ref_t = que_t, len(array2d)
        else:
            ref_s, ref_t = 0, que_s
    else:
        que_s, que_t, ref_s, ref_t = 0, len(array2d), 0, len(array2d)

    new_array2d = []
    for u, row in enumerate(array2d[que_s: que_t]):
        row = [(v + ref_s, col) for (v, col) in enumerate(row[ref_s: ref_t]) if u + que_s != v + ref_s]
        new_array2d.append(row)
    MAP, top10, rank1 = 0, 0, 0
    
    for u, row in enumerate(new_array2d):
        row.sort(key=lambda x: x[1])
        per_top10, per_rank1, per_MAP = 0, 0, 0
        version_cnt = 0.
        u = u + que_s
        for k, (v, val) in enumerate(row):
            
            if version[u] == version[v]:
                
                if k < K:
                    version_cnt += 1
                    per_MAP += version_cnt / (k + 1)
                if per_rank1 == 0:
                    per_rank1 = k + 1
                if k < 10:
                    per_top10 += 1
        per_MAP /= 1 if version_cnt < 0.0001 else version_cnt
        # if per_MAP < 0.1:
        #     print row
        MAP += per_MAP
        top10 += per_top10
        rank1 += per_rank1
    return MAP / float(que_t - que_s), top10 / float(que_t - que_s) / 10, rank1 / float(que_t - que_s)

def calc_MAP(array2d, version, que_range=None, K=1e10):
    if que_range is not None:
        que_s, que_t = que_range[0], que_range[1]
        if que_s == 0:
            ref_s, ref_t = que_t, len(array2d)
        else:
            ref_s, ref_t = 0, que_s
    else:
        que_s, que_t, ref_s, ref_t = 0, len(array2d), 0, len(array2d)

    new_array2d = []
    for u, row in enumerate(array2d[que_s: que_t]):
        row = [(v + ref_s, col) for (v, col) in enumerate(row[ref_s: ref_t]) if u + que_s != v + ref_s]
        new_array2d.append(row)
    MAP, top10, rank1 = 0, 0, 0
    
    for u, row in enumerate(new_array2d):
        row.sort(key=lambda x: x[1])
        per_top10, per_rank1, per_MAP = 0, 0, 0
        version_cnt = 0.
        u = u + que_s
        for k, (v, val) in enumerate(row):
            if version[u] == version[v]:
                if k < K:
                    version_cnt += 1
                    per_MAP += version_cnt / (k + 1)
                if per_rank1 == 0:
                    per_rank1 = k + 1
                if k < 10:
                    per_top10 += 1
        per_MAP /= 1 if version_cnt < 0.0001 else version_cnt
        # print(per_MAP, u, row[0][0], '%d_%d' % (u / 2, u % 2), '%d_%d' % (row[0][0] / 2, row[0][0] % 2), row[0][1])
        # if per_MAP < 0.1:
        #     print row
        MAP += per_MAP
        top10 += per_top10
        rank1 += per_rank1
    return MAP / float(que_t - que_s), top10 / float(que_t - que_s) / 10, rank1 / float(que_t - que_s)

def get_dis2d4(seqs, verbose=False):
    start_time = time.time()
    dis2d = np.zeros((len(seqs), len(seqs)))
    for i, seq1 in enumerate(seqs):
        idx = np.where(seq1 != 0)
        x = seq1[idx].squeeze()
        for j, seq2 in enumerate(seqs):
            y = seq2[idx].squeeze()
            dis2d[i][j] = 1 - np.dot(x, y)
    end_time = time.time()
    if verbose:
        print( 'time: %fs' % (end_time - start_time))
    return dis2d
def get_disEu(seqs, verbose=False):
    start_time = time.time()
    dis2d = np.zeros((len(seqs), len(seqs)))
    for i, seq1 in enumerate(seqs):
        idx = np.where(seq1 != 0)
        x = seq1[idx].squeeze()
        for j, seq2 in enumerate(seqs):
            y = seq2[idx].squeeze()
            dis2d[i][j] = np.sqrt(np.sum(np.square(x - y)))
    end_time = time.time()
    if verbose:
        print( 'time: %fs' % (end_time - start_time))
    return dis2d
def vis(model, dataloader):
    seqs = []
    labels = []
    for ii, (data,label) in enumerate(dataloader):
        if ii==0 or ii==1: print(data.shape)
        seqs.append(data)
        labels.append(label)
    vis = visdom.Visdom()
    for i in range(0,79,2):
        seq_o,seq_p  = model.model(seqs[i].cuda()),model.model(seqs[i+1].cuda())
        out = model.metric(seq_o,seq_p,debug=True)
        s_ap, d_ap, align_ap, d_ap_s = out[0],out[1],out[2],out[3]
        print(s_ap.squeeze(0).data.cpu().numpy())

        vis.heatmap(align_ap.squeeze(0).data.cpu().numpy(),opts={'title':"alignment"})
        vis.heatmap(d_ap.squeeze(0).data.cpu().numpy(), opts={'title': "matrix"})
        vis.heatmap(d_ap_s.squeeze(0).data.cpu().numpy(), opts={'title': "matrix_S"})


@torch.no_grad()
def val_slow_batch1(softdtw, dataloader, batch=200, is_dis='False'):
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
    for st in range(0, N, batch):
        fi = (st + batch) if st + batch <= N else N
        query = seqs[query_l[st: fi], :, :]
        ref = seqs[ref_l[st: fi], :, :]
        if torch.cuda.device_count() > 1:
            s = softdtw.module.multi_compute_seq(query, ref).data.cpu().numpy()
        else:
            s = softdtw.multi_compute_seq(query, ref).data.cpu().numpy()

        for k in range(st, fi):
            i, j = query_l[k], ref_l[k]
            # print(i, j)
            if is_dis:
                dis2d[i, j] = s[k - st]
            else:
                dis2d[i, j] = -s[k - st]

    if len(labels) == 350:
        MAP, top10, rank1 = calc_MAP(dis2d, labels, [100, 350])
    else:
        MAP, top10, rank1 = calc_MAP(dis2d, labels)
    print(MAP, top10, rank1)

    softdtw.train()
    if torch.cuda.device_count() > 1:
        softdtw.module.model.train()
    else:
        softdtw.model.train()
    return MAP


@torch.no_grad()
def val_slow_batch2(softdtw, dataloader, batch=200, is_dis='False'):
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
    for st in range(0, N, batch):
        fi = (st + batch) if st + batch <= N else N
        query = seqs[query_l[st: fi], :, :]
        ref = seqs[ref_l[st: fi], :, :]
        if torch.cuda.device_count() > 1:
            s = softdtw.module.multi_compute_vec(query, ref).data.cpu().numpy()
        else:
            s = softdtw.multi_compute_vec(query, ref).data.cpu().numpy()
        for k in range(st, fi):
            i, j = query_l[k], ref_l[k]
            # print(i, j)
            if is_dis:
                dis2d[i, j] = s[k - st]
            else:
                dis2d[i, j] = -s[k - st]

    if len(labels) == 350:
        MAP, top10, rank1 = calc_MAP(dis2d, labels, [100, 350])
    else:
        MAP, top10, rank1 = calc_MAP(dis2d, labels)
    print(MAP, top10, rank1)

    softdtw.train()
    if torch.cuda.device_count() > 1:
        softdtw.module.model.train()
    else:
        softdtw.model.train()
    return MAP
def test_visualize(softdtw):
    softdtw.eval()
    softdtw.model.eval()
    # filename_o = "robert_palmer+Riptide+03-Addicted_To_Love.wav"
    # filename_c = "tina_turner+Tina_Live_In_Europe_CD_1_+09-Addicted_To_Love.wav"
    # filename_n = "test_song/lemon_tree/p.mp3"
    filename_o = "covers80/coversongs/covers32k/Cecilia/paul_simon+Concert_in_the_Park_Disc_2+11-Cecilia.mp3"
    filename_c = "covers80/coversongs/covers32k/Cecilia/simon_and_garfunkel+Collected_Works_-_Disc_3+03-Cecilia.mp3"
    filename_n = "covers80/coversongs/covers32k/Addicted_To_Love/tina_turner+Tina_Live_In_Europe_CD_1_+09-Addicted_To_Love.mp3"
    print(filename_o)
    namelist = [filename_o, filename_c, filename_n]
    y, sr = librosa.load(filename_o, sr=None)
    mean_size = 20
    seqlist = []
    seqlist2 = []
    seqlist3 = []
    seqlist4 = []

    s_cqt = []

    # cqt = np.abs(librosa.feature.chroma_cqt(y=y, sr=sr))
    for i in range(3):
        y, sr = librosa.load(namelist[i], sr=None)
        # if i == 0:
        #     # y = librosa.effects.time_stretch(y,1.2)
        #     y = librosa.effects.pitch_shift(y,sr,n_steps=6.0)
        cqt = np.abs(librosa.cqt(y=y, sr=sr))
        height, length = cqt.shape
        new_cqt = np.zeros((height, int(length / mean_size)), dtype=np.float64)
        for i in range(int(length / mean_size)):
            new_cqt[:, i] = cqt[:, i * mean_size:(i + 1) * mean_size].mean(axis=1)
        data = new_cqt[:, :400].T
        data = data.T
        data = cut_data_front(data, 400)
        data = torch.from_numpy(data).float()
        s = data.permute(1, 0).cuda().unsqueeze(0).unsqueeze(0)
        # s = torch.from_numpy(s.astype(np.float32) / (np.max(np.abs(s))+ 1e-6)).permute(1,0).cuda().unsqueeze(0).unsqueeze(0)
        s_cqt.append(s)
        if i == 0:
            seqa, seqa2, seqa3, seqa4 = softdtw.model(s)
            print(seqa.shape)
            print(seqa2.shape)
            print(seqa3.shape)
            print(seqa4.shape)
            seqlist.append(seqa)
            seqlist2.append(seqa2)
            seqlist3.append(seqa3)
            seqlist4.append(seqa4)
        elif i == 1:
            seqa, seqa2, seqa3, seqa4 = softdtw.model(s)
            print(seqa.shape)
            print(seqa2.shape)
            print(seqa3.shape)
            print(seqa4.shape)
            seqlist.append(seqa)
            seqlist2.append(seqa2)
            seqlist3.append(seqa3)
            seqlist4.append(seqa4)
        else:
            seqa, seqa2, seqa3, seqa4 = softdtw.model(s)
            print(seqa.shape)
            print(seqa2.shape)
            print(seqa3.shape)
            print(seqa4.shape)
            seqlist.append(seqa)
            seqlist2.append(seqa2)
            seqlist3.append(seqa3)
            seqlist4.append(seqa4)

    metrix_p = softdtw.metric(seqlist[0], seqlist[1], True).squeeze(0).data.cpu().numpy()
    metrix_n = softdtw.metric(seqlist[0], seqlist[2], True).squeeze(0).data.cpu().numpy()
    metrix_p2 = softdtw.metric(seqlist2[0], seqlist2[1], True).squeeze(0).data.cpu().numpy()
    metrix_n2 = softdtw.metric(seqlist2[0], seqlist2[2], True).squeeze(0).data.cpu().numpy()
    metrix_p3 = softdtw.metric(seqlist3[0], seqlist3[1], True).squeeze(0).data.cpu().numpy()
    metrix_n3 = softdtw.metric(seqlist3[0], seqlist3[2], True).squeeze(0).data.cpu().numpy()
    metrix_p4 = softdtw.metric(seqlist4[0], seqlist4[1], True).squeeze(0).data.cpu().numpy()
    metrix_n4 = softdtw.metric(seqlist4[0], seqlist4[2], True).squeeze(0).data.cpu().numpy()
    vis = visdom.Visdom()
    # vis.heatmap(seqlist[0].squeeze(0).data.cpu().numpy()[:,:25])
    # vis.heatmap(seqlist[1].squeeze(0).data.cpu().numpy()[:,:25])
    # vis.heatmap(seqlist[2].squeeze(0).data.cpu().numpy()[:,:25])
    print(softdtw.multi_compute_s(s_cqt[0], s_cqt[1]))
    vis.heatmap(metrix_p, opts={"title": "deep featrue 1"})
    vis.heatmap(metrix_p2, opts={"title": "deep featrue 2"})
    vis.heatmap(metrix_p3, opts={"title": "deep featrue 3"})
    vis.heatmap(metrix_p4, opts={"title": "deep featrue 4"})
    # vis.heatmap(metrix_n ,opts={"title":"deep featrue 1"})
    # vis.heatmap(metrix_n2,opts={"title":"deep featrue 2"})
    # vis.heatmap(metrix_n3,opts={"title":"deep featrue 3"})
    # vis.heatmap(metrix_n4,opts={"title":"deep featrue 4"})


def test_covers80(model, dataloader):
    model.eval()
    model.model.eval()
    song = []
    for ii, (data, label) in tqdm(enumerate(dataloader)):
        input = data.cuda()
        song.append(input)
    max_list = []
    cover_list = []
    vis = visdom.Visdom()
    for i in range(0, len(song), 1):
        song_id = i
        if i % 2 == 0:
            cover_id = song_id + 1
        else:
            cover_id = song_id - 1
        s_list, c_list = model.model(song[song_id]), model.model(song[cover_id])
        m = [model.metric(s_list[i], c_list[i]).squeeze(0).data.cpu().numpy()
             for i in range(len(s_list))]
        sam = model.multi_compute_s(song[song_id], song[cover_id]).data.cpu().numpy()
        cover_list.append({"metrix_list": m, "similarity": sam, "cover_id": cover_id, "id": song_id})
        max_id = cover_id
        max_similarity = sam
        max_m = m
        for j in range(0, len(song)):
            if j != song_id and j != cover_id:
                r_list = model.model(song[j])
                m_r = [model.metric(s_list[z], r_list[z]).squeeze(0).data.cpu().numpy()
                       for z in range(len(s_list))]
                sam_r = model.multi_compute_s(song[song_id], song[j]).data.cpu().numpy()

                if max_similarity < sam_r:
                    max_id = j
                    max_similarity = sam_r
                    max_m = m_r
        max_list.append({"id": song_id, "cover_id": cover_id, "max_id": max_id, "max_similarity": max_similarity,
                         "max_matrix_list": max_m})
    # print(max_list)
    # print(cover_list)
    count = 0

    with open("/S1/DAA/jcy/ner/covers80/coversongs/list1.txt", "r") as l1:
        line_list1 = l1.readlines()

    with open("/S1/DAA/jcy/ner/covers80/coversongs/list2.txt", "r") as l2:
        line_list2 = l2.readlines()
    fl = []
    for i in range(len(line_list1)):
        fl.append(line_list1[i][:-1])
        fl.append(line_list2[i][:-1])

    covers_info_file = open("covers_info3.txt", "w")
    wrong_info_file = open("wrong_info3.txt", "w")
    for i in range(len(cover_list)):
        print(cover_list[i]["id"])
        print(cover_list[i]["similarity"])
        covers_info_file.write(str(cover_list[i]["id"]) + "|" + str(cover_list[i]["similarity"]) + "\n")
        if max_list[i]["cover_id"] != max_list[i]["max_id"]:
            count += 1
            print("max:", max_list[i]["max_id"], " similarity:", max_list[i]["max_similarity"])
            for k in range(4):
                vis.heatmap(cover_list[i]["metrix_list"][k], opts={"title": str(cover_list[i]["id"]) + "&" + str(
                    cover_list[i]["cover_id"]) + "cover deep featrue" + str(k)})
                vis.heatmap(max_list[i]["max_matrix_list"][k], opts={
                    "title": str(cover_list[i]["id"]) + "&" + str(max_list[i]["max_id"]) + "max deep featrue" + str(k)})
            wrong_info_file.write(
                str(cover_list[i]["id"]) + "|" + fl[cover_list[i]["id"]] + "|" + str(max_list[i]["max_id"]) + "|"
                + fl[max_list[i]["max_id"]] + "|" +
                str(cover_list[i]["similarity"]) + "|" + str(max_list[i]["max_similarity"]) + "\n")
        print("-------------------- wrong_count :", count, " -------------------")

    covers_info_file.close()
    wrong_info_file.close()


def visualize(softdtw, dataloader):
    # softdtw.eval()
    # softdtw.model.eval()
    softdtw.eval()
    softdtw.model.eval()

    seqs = []
    Song_input = []
    i, j, k = 26, 27, 80
    Song_data = []
    Song_Cover_data = []
    Song_Cover_same = []
    Song_Cover_metrix = []
    Song_MaxSame_id = []
    Song_MaxSame_metrix = []
    Song_MaxSame_sam = []
    Song_nocover_metrix = []
    Song_nocover_id = []
    Song_nocover_sam = []
    Song_inhance = []
    inhance_kernel = np.array([[0, 0, 0, 0, -1],
                               [0, 0, 0, -1, 0],
                               [0, 0, 4, 0, 0],
                               [0, -1, 0, 0, 0],
                               [-1, 0, 0, 0, 0]])
    inhance_kernel = torch.autograd.Variable(torch.from_numpy(inhance_kernel).unsqueeze(0).unsqueeze(0).float())
    print(inhance_kernel.shape)
    for ii, (data, label) in tqdm(enumerate(dataloader)):
        # if ii == i or ii == j or ii == k:
        #     input = data.cuda()
        #     Song_input.append(input)
        #     _,seq,_,_= softdtw.model(input)
        #     seqs.append(seq)
        # if ii<80:
        input = data.cuda()
        Song_input.append(input)

    # for i in range(0, len(Song_input) - 1, 2):
    for i in range(0, 2, 2):
        _, _, seqa, _ = softdtw.model(Song_input[i])
        Song_data.append(seqa.data.cpu().numpy())
        _, _, seqb, _ = softdtw.model(Song_input[i + 1])
        Song_Cover_data.append(seqb.data.cpu().numpy())
        metrix = softdtw.metric(seqa, seqb, True)
        # si= F.conv2d(metrix.unsqueeze(0),inhance_kernel.cuda(),padding=2)####锐化
        # Song_inhance.append(si.squeeze(0).squeeze(0).data.cpu().numpy())####锐化
        Song_Cover_metrix.append(metrix.squeeze(0).data.cpu().numpy())
        sam = softdtw.multi_compute_s(Song_input[i], Song_input[i + 1])
        Song_Cover_same.append(sam.data.cpu().numpy())
        max_num = sam
        max_id = i + 1
        max_metrix = metrix
        Song_i_nocover_metrix = []
        Song_i_nocover_sam = []
        Song_i_nocover_id = []
        Song_i_nocover_data = []
        for j in range(0, i):
            _, _, seqb, _ = softdtw.model(Song_input[j])
            Song_i_nocover_data.append(seqb.data.cpu().numpy())
            nocover_metrix = softdtw.metric(seqa, seqb, True)
            Song_i_nocover_metrix.append(nocover_metrix.squeeze(0).data.cpu().numpy())
            Song_i_nocover_id.append(j)
            nocover_sam = softdtw.multi_compute_s(Song_input[i], Song_input[j])
            Song_i_nocover_sam.append(nocover_sam.data.cpu().numpy())
            if max_num < nocover_sam:
                max_num = nocover_sam
                max_id = j
                max_metrix = nocover_metrix
        for j in range(i + 2, len(Song_input)):
            _, _, seqb, _ = softdtw.model(Song_input[j])
            Song_i_nocover_data.append(seqb.data.cpu().numpy())
            nocover_metrix = softdtw.metric(seqa, seqb, True)
            Song_i_nocover_metrix.append(nocover_metrix.squeeze(0).data.cpu().numpy())
            Song_i_nocover_id.append(j)
            nocover_sam = softdtw.multi_compute_s(Song_input[i], Song_input[j])
            Song_i_nocover_sam.append(nocover_sam.data.cpu().numpy())
            if max_num < nocover_sam:
                max_num = nocover_sam
                max_id = j
                max_metrix = nocover_metrix
        Song_MaxSame_id.append(max_id)
        Song_MaxSame_metrix.append(max_metrix.squeeze(0).data.cpu().numpy())
        Song_MaxSame_sam.append(max_num.data.cpu().numpy())
        Song_nocover_metrix.append(Song_i_nocover_metrix)
        Song_nocover_id.append(Song_i_nocover_id)
        Song_nocover_sam.append(Song_i_nocover_sam)
    # sap, d_ap, b_d_ap,align_ap,s_ap = softdtw.metric(seqs[0], seqs[1], True)
    # san, d_an, b_d_an,align_an,s_an = softdtw.metric(seqs[0], seqs[2], True)
    # print(softdtw(Song_input[0],Song_input[1],Song_input[2]))
    # b_d_ap = softdtw.metric(seqs[0], seqs[1], True)
    # b_d_an = softdtw.metric(seqs[0], seqs[2], True)
    # b_d_ap ,b_d_an = b_d_ap.squeeze(0),b_d_an.squeeze(0)
    image_count = 0
    vis = visdom.Visdom( )
    
    for i in range(len(Song_data)):
        print("Song_", i, " :")
        print("  Song_Cover sam:", Song_Cover_same[i])
        print(Song_Cover_metrix[i].shape)
        vis.heatmap(Song_Cover_metrix[i], opts={
            'title': "image" + str(image_count) + ' Cover_metrix' + str(i) + "&" + str(i + 1) + ':' + str(
                Song_Cover_same[i]), 'colormap': 'Winter'})
        image_count += 1
        # vis.heatmap(      Song_inhance[i],opts={'title':"image"+str(image_count)+' inhance_Cover_metrix'+str(i)+"&"+str(i+1)})
        if Song_MaxSame_id[i] != (2 * i + 1):
            print("  Song_", i, "_maxSam: id ", Song_MaxSame_id[i], " Sam ", Song_MaxSame_sam[i])
            print("-------------------Error pred: image_id ", image_count)
            print(Song_MaxSame_metrix[i].shape)
            print(Song_MaxSame_metrix[i])
            vis.heatmap(Song_MaxSame_metrix[i], opts={
                'title': "image" + str(image_count) + ' Max_metrix' + str(i) + "&" + str(
                    Song_MaxSame_id[i]) + ':' + str(Song_MaxSame_sam[i]), 'colormap': 'Cool'})

    # d_ap, b_d_ap,align_ap =  d_ap.squeeze(0), b_d_ap.squeeze(0),align_ap.squeeze(0)
    # d_an, b_d_an,align_an =  d_an.squeeze(0), b_d_an.squeeze(0),align_an.squeeze(0)
    # print(align_an.shape)
    # d_ap, b_d_ap, align_ap = d_ap.data.cpu().numpy(), b_d_ap.data.cpu().numpy(), align_ap.data.cpu().numpy()
    # d_an, b_d_an, align_an = d_an.data.cpu().numpy(), b_d_an.data.cpu().numpy(), align_an.data.cpu().numpy()
    # s_ap = s_ap.data.cpu().numpy()[0]
    # s_an = s_an.data.cpu().numpy()[0]

    # vis.heatmap(align_ap,opts={'title':'align_ap'})
    # vis.heatmap(s_ap,opts={'title':'s_ap'})
    # vis.heatmap(b_d_ap,opts={'title':'b_d_ap'})

    # vis.heatmap(align_an,opts={'title':'align_an'})
    # vis.heatmap(s_an,opts={'title':'s_an'})
    # vis.heatmap(b_d_an,opts={'title':'b_d_an'})

    # print(sap, san)

    softdtw.train()
    softdtw.model.train()
