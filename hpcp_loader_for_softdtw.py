import os,sys
from torchvision import transforms
import torch, torch.utils
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random
import bisect




def cut_data_front(data, out_length):
    if out_length is not None:
        if data.shape[0] > out_length:
            max_offset = data.shape[0] - out_length
            #offset = np.random.randint(max_offset)
            offset = 0
            data = data[offset:(out_length+offset),:]
        else:
            offset = out_length - data.shape[0]
            data = np.pad(data, ((0,offset),(0,0)), "constant")
    if data.shape[0] < 200:
        offset = 200 - data.shape[0]
        data = np.pad(data, ((0,offset),(0,0)), "constant")
    return data



def cut_data(data, out_length, is_random=True):
    if out_length is not None:
        if data.shape[0] > out_length:
            if is_random:
                max_offset = data.shape[0] - out_length
                offset = np.random.randint(max_offset)
                data = data[offset:(out_length+offset),:]
            else:
                data = data[:out_length, :]
        else:
            offset = out_length - data.shape[0]
            data = np.pad(data, ((0,offset),(0,0)), "constant")
    if data.shape[0] < 150:
        offset = 150 - data.shape[0]
        data = np.pad(data, ((0,offset),(0,0)), "constant")
    return data


class triplet_CQT(Dataset):
    def __init__(self, out_length, datatype, filepath='/S3/DAA/wxy/cover_song_identification/similarity/list/gdoras_train_triplet_long_r15.list', is_random=True, is_label=False):
        if datatype == 'cqt':
            self.indir = '/S3/DAA/gdoras_dataset/avg5/cqts_padded_1937x72/'
        elif datatype == 'multif0':
            self.indir = '/S3/DAA/gdoras_dataset/avg5/multif0_cqts_padded_1937x72/'
        elif datatype == 'shs100k':
            self.indir = '/S3/DAA/jcy/SCNN/data/youtube_cqt_npy/'
            filepath='/S3/DAA/wxy/cover_song_identification/similarity/list/triplet_SHS100K-TRAIN.list'
        self.datatype = datatype
        self.out_length = out_length
        with open(filepath, 'r') as fp:
            self.file_list = [line.rstrip() for line in fp]
        self.is_random = is_random
        self.is_label = is_label

    def __getitem__(self, index):
        filename = self.file_list[index].strip()
        name_list = filename.split(',')
        data_list = []
        for name in name_list: # name: 'sid_vid'
            song_id, version_id = name.split('_')
            song_id = int(song_id)
            if self.datatype == 'multif0':
                data = np.load(self.indir+version_id+'.multif0_cqt.npy')
            elif self.datatype == 'cqt':
                data = np.load(self.indir+version_id+'.cqt.npy')
            elif self.datatype == 'shs100k':
                data = np.load(self.indir+name+'.npy')
            data = data.T
            data = cut_data(data, self.out_length, self.is_random)
            data = torch.from_numpy(data).float()
            data = data.permute(1, 0).unsqueeze(0)
            data_list.append(data)
            data_list.append(song_id)
        if self.is_label:
            return data_list[0], data_list[2], data_list[4], data_list[1], data_list[3], data_list[5]
        else:
            return data_list[0], data_list[2], data_list[4]

    def __len__(self):
        return len(self.file_list)



# 一般测试用
class CQT(Dataset):
    def __init__(self, mode='train', out_length=None):
        self.mode = mode
        
        if mode == 'multif0':
            self.indir = '/S3/DAA/gdoras_dataset/avg5/multif0_cqts_padded_1937x72/'
            filepath = '/S3/DAA/wxy/cover_song_identification/similarity/list/gdoras_test_for_train.list'
        elif mode[:-1] == 'gdoras_test_multif0_':
            self.indir = '/S3/DAA/gdoras_dataset/avg5/multif0_cqts_padded_1937x72/'
            filepath = f'/S3/DAA/wxy/cover_song_identification/similarity/list/gdoras_test_{mode[-1]}.list'
        elif mode[:-1] == 'gdoras_test_cqt_':
            self.indir = '/S3/DAA/gdoras_dataset/avg5/cqts_padded_1937x72/'
            filepath = f'/S3/DAA/wxy/cover_song_identification/similarity/list/gdoras_test_{mode[-1]}.list'
        elif mode == 'cqt':
            self.indir = '/S3/DAA/gdoras_dataset/avg5/cqts_padded_1937x72/'
            filepath = '/S3/DAA/wxy/cover_song_identification/similarity/list/gdoras_test_for_train.list'
#         elif mode == 'val':
#             # filepath='hpcp/val_list.txt'
#             filepath = 'hpcp/SHS100K-VAL'
        elif mode == 'songs350':
            self.indir = '/S3/DAA/jcy/SCNN/data/you350_cqt_npy/'
            filepath = '/S3/DAA/jcy/SCNN/hpcp/you350_list.txt'
        elif mode == 'shs100k_val':
            self.indir = '/S3/DAA/jcy/SCNN/data/youtube_cqt_npy/'
            filepath = '/S3/DAA/wxy/cover_song_identification/similarity/list/SHS100K-VAL-short'
        elif mode[:-1] == 'shs100k_test_':
            self.indir = '/S3/DAA/jcy/SCNN/data/youtube_cqt_npy/'
            filepath = f'/S3/DAA/wxy/cover_song_identification/similarity/list/SHS100K-TEST-{mode[-1]}'
            # filepath='hpcp/test_list.txt'
        elif mode == 'covers80':
            self.indir = '/S3/DAA/jcy/SCNN/data/covers80_cqt_npy/'
            filepath = '/S3/DAA/jcy/SCNN/hpcp/80.list'
#         elif mode == 'songs2000':
#             self.indir = 'data/songs2000_cqt_npy/'
#             filepath = 'hpcp/songs2000_list.txt'
#         elif mode == 'new80':
#             self.indir = 'data/songs2000_cqt_npy/'
#             filepath = 'hpcp/new80_list.txt'
        elif mode == 'Mazurkas':
            self.indir = '/S3/DAA/jcy/SCNN/data/Mazurkas_cqt_npy/'
            filepath = '/S3/DAA/jcy/SCNN/hpcp/Mazurkas_list.txt'
        
        with open(filepath, 'r') as fp:
            self.file_list = [line.rstrip() for line in fp]
        self.out_length = out_length

    def __getitem__(self, index):

        filename = self.file_list[index].strip()
        set_id, version_id = filename.split('.')[0].split('_')
        if self.mode == 'multif0' or self.mode[:-1] == 'gdoras_test_multif0_':
            in_path = self.indir + version_id + '.multif0_cqt.npy'
        elif self.mode == 'cqt' or self.mode[:-1] == 'gdoras_test_cqt_':
            in_path = self.indir + version_id + '.cqt.npy'
        elif self.mode in ['songs350','shs100k_val','covers80','Mazurkas'] or 'shs100k_test_' in self.mode:
            in_path = self.indir + filename + '.npy'
        data = np.load(in_path)  # from 12xN to Nx12
        data = data.T
        # Cut to 394
        if self.mode == 'train':
            data = cut_data(data, self.out_length)  # L, 84
        else:
            data = cut_data_front(data, self.out_length)
        # 12 to 23
        data = torch.from_numpy(data).float()
        # data = torch.from_numpy(data[:,0:-13] ).float()
        data = data.permute(1, 0).unsqueeze(0)

        return data, int(set_id)

    def __len__(self):
        return len(self.file_list)

    
    
if __name__=='__main__':
    pass
