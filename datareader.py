import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
import os,glob
from params import *
from tsdf import *
from time import time

class MSRADataSet(Dataset):
    def __init__(self,root_path):
        self.root_path = root_path
        self.subjects = filter(lambda x: os.path.isdir(os.path.join(root_path, x)), os.listdir(root_path))
        self.gestures = GESTURES
        self.samples = []
        self.imgs = []
        for i in range(MAX_SAMPLE_LEN):
            self.samples.append('{:06d}'.format(i)+DATA_EXT)
            self.imgs.append('{:06d}'.format(i)+IMG_EXT)
        self.label = LABEL
        self.data = []
        for sub in self.subjects:
            for ges in self.gestures:
                with open(os.path.join(root_path,sub,ges,LABEL)) as f:
                    lines = int(f.readline())
                    label = np.loadtxt(f,delimiter=' ',dtype=np.float32)
                for i in range(lines):
                    data = {'path':os.path.join(root_path,sub,ges,self.samples[i]),'label':label[i,:]}
                    self.data.append(data)
        self.length = len(self.data)

    def __getitem__(self, item):
        with open(self.data[item]['path']) as f:
            # img_width img_height left top right bottom
            header = np.fromfile(f,np.int32,6)
            data = np.fromfile(f,np.float32)

        sample = {'header': header,
                  'data': data}
        pc, tsdf, labels = cal_tsdf_cuda([sample,self.data[item]['label']])
        # if self.transform:
        #     sample = self.transform(sample)

        # return sample, self.data[item]['label']

        return tsdf,labels

    def __len__(self):
        return self.length


if __name__ == "__main__":
    t1 = time()
    m = MSRADataSet('/home/hfy/data/msra15/')
    t2 = time()
    for i in range(100):
        data,label = m[0]
    t3 = time()

    # pc_p = get_project_data(m[0][0])
    # plot_pointcloud(pc)
    # plot_tsdf(data.numpy(),label.numpy())
    # t4 = time()
    # print t2-t1,t3-t2,t4-t3
    # show()