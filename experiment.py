import numpy as np

from dataset.MSRA import *
from utils.img import DrawGaussian

def visual_img(img):
    a = img*255
    Image.fromarray(a).show()

if __name__ == "__main__":
    m = MSRADataSet(DATA_DIR)
    # m.use_raw = True
    data, heatmaps, label = m[0]
    visual_img(data[0])
    visual_img(data[1])
    visual_img(data[2])
    visual_img(heatmaps[0])
    # heatmaps = np.sum(heatmaps.reshape((JOINT_LEN,-1)),axis=0).reshape((HM_SIZE,HM_SIZE))
    # img = DrawGaussian(np.zeros((64,64),dtype=np.float32),np.array([10,1]),1)
    # visual_img(img)
    # for i in range(1):
    #     visual_img(heatmaps[i])
    # maps = gt2heatmaps(label)
    # print maps
    # for i in range(10):
    #     t = time()
    #     # pc, label = m.get_point_cloud(i)
    #     # plot_pointcloud(pc, label)
    #     data, label, max_p, mid_p = m[i]
    #     plot_tsdf(data, max_p, mid_p, label,0)
    #     plot_tsdf(data, max_p, mid_p, label,1)
    #     plot_tsdf(data, max_p, mid_p, label,2)
    #
    #     print time()-t
