from helper import *

def gt2heatmaps(gt):
    gt = gt.reshape(-1,3)
    maps = np.zeros((JOINT_LEN,HM_SIZE,HM_SIZE),dtype=np.bool)
    gt_pixel = camera2pixel(gt, *get_param('msra'))
    for i in range(JOINT_LEN):
        print (gt_pixel[i,:2])
        # maps[i, gt_pixel[i,:2]] = 1
    return maps

def visualize_heatmap(hms):
    pass

