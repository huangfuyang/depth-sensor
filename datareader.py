import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from sensor import cal_angle_from_pos
import os,glob
from params import *
from tsdf import *
from time import time
import warnings
import tables

# joint  sequence: 21*3
# wrist, index_mcp, index_pip, index_dip, index_tip, middle_mcp, middle_pip, middle_dip, middle_tip, ring_mcp,
# ring_pip, ring_dip, ring_tip, little_mcp, little_pip, little_dip, little_tip, thumb_mcp, thumb_pip,
# thumb_dip, thumb_tip
class MSRADataSet(Dataset):
    def __init__(self,root_path, use_sensor=False, use_preprocessing = False):
        self.root_path = root_path
        self.subjects = filter(lambda x: os.path.isdir(os.path.join(root_path, x)), os.listdir(root_path))
        self.subjects.sort()
        self.gestures = GESTURES
        self.samples = []
        self.subjects_length = []  # samples per subject
        self.imgs = []
        self.use_sensor = use_sensor
        self.use_preprocessing = use_preprocessing
        if use_preprocessing:
            self.f = tables.open_file('tsdf.h5', mode='r')
        for i in range(MAX_SAMPLE_LEN):
            self.samples.append('{:06d}'.format(i)+DATA_EXT)
            self.imgs.append('{:06d}'.format(i)+IMG_EXT)
        self.label = LABEL
        self.data = []
        for sub in self.subjects:
            sub_len = 0
            for ges in self.gestures:
                with open(os.path.join(root_path,sub,ges,LABEL)) as f:
                    lines = int(f.readline())
                    label = np.loadtxt(f,delimiter=' ',dtype=np.float32)
                for i in range(lines):
                    sub_len+=1
                    data = {'path':os.path.join(root_path,sub,ges,self.samples[i]),'label':label[i,:]}
                    self.data.append(data)
            self.subjects_length.append(sub_len)
        self.length = len(self.data)
        assert np.array(self.subjects_length).sum() == self.length

    # return 32*32*32 tsdf, 63 label, 1 max_l
    def __getitem__(self, item):
        with open(self.data[item]['path']) as f:
            # img_width img_height left top right bottom
            header = np.fromfile(f,np.int32,6)
            data = np.fromfile(f,np.float32)

        sample = {'header': header,
                  'data': data}

        # tsdf output
        if self.use_preprocessing:
            row = self.f.root.data[item,:]
            return row[:VOXEL_RES**3].reshape((VOXEL_RES,VOXEL_RES,VOXEL_RES)), row[VOXEL_RES**3:-1], row[-1]
        else:
            r = cal_tsdf_cuda(sample)
            if r is None:
                return None
            else:
                tsdf, max_l, mid_p = r
                label = self.data[item]['label']
                if self.use_sensor:
                    angles = cal_angle_from_pos(label.copy())
                    return (tsdf,angles), label, max_l, mid_p
                else:
                    return tsdf, label, max_l, mid_p

    def get_point_cloud(self, index):
        sample = self.get_raw_data(self.data[index]['path'])
        # point cloud output
        pc = cal_pointcloud(sample)
        return pc, self.data[index]['label']

    def get_raw_data(self, path):
        with open(path) as f:
            # img_width img_height left top right bottom
            header = np.fromfile(f,np.int32,6)
            data = np.fromfile(f,np.float32)
        return {'header': header,
                'data': data}

    def __len__(self):
        return self.length

    def get_train_test_indices(self,test_id = 0):
        if test_id >= len(self.subjects_length) or test_id < 0:
            raise ValueError("value must within range (0,{})".format(len(self.subjects_length)-1))
        if len(self.subjects_length) < 2:
            raise ValueError("dataset cannot be split")
        n_lens = np.array(self.subjects_length,dtype=np.int32)
        train = []
        test = []
        l = 0
        for i in range(len(self.subjects_length)):
            if i == test_id:
                test.extend(range(l,l+self.subjects_length[i]))
            else:
                train.extend(range(l,l+self.subjects_length[i]))
            l+=self.subjects_length[i]
        return train,test


if __name__ == "__main__":
    pass
    # m = MSRADataSet(DATA_DIR, use_preprocessing=True)
    from visualization import plot_tsdf, plot_pointcloud
    # p = "/home/hfy/data/msra15/P0/4/000001_depth.bin"
    # s = m.get_raw_data(p)
    # pc = cal_pointcloud(s)
    # plot_pointcloud(pc)
    # print pc[pc[:,2]>-240]
    # for i in range(1520,1540):
    #     t,l,ma = m[i]
    #     plot_tsdf(t,l)

    # for i in range(10):
    #     t = time()
    #     pc, label = m.get_point_cloud(i)
    #     plot_pointcloud(pc, label)
    #     data, label,(mid,max_p) = m[i]
    #     plot_tsdf(data,label)
    #
    #     print time()-t
    # data,label = m[0]

    # pc_p = get_project_data(m[0][0])
    # plot_pointcloud(pc)
