from mayavi.mlab import *
import numpy as np
from params import *


def plot_tsdf(data, max_p, mid_p, label=None, axis=0):
    data = np.squeeze(data[axis,:,:,:]).reshape(-1)
    scale = np.zeros(data.shape)+0.5
    scale[data == 1] = 0
    scale[data == -1] = 0
    scale[data == 0] = 0
    data = (data+1)/2
    graph = points3d(np.tile(np.arange(-VOXEL_RES / 2, VOXEL_RES / 2), (VOXEL_RES * VOXEL_RES)),
             np.tile(np.repeat(np.arange(-VOXEL_RES / 2, VOXEL_RES / 2), VOXEL_RES), VOXEL_RES),
             np.repeat(np.arange(-VOXEL_RES / 2, VOXEL_RES / 2), (VOXEL_RES * VOXEL_RES)),
                     scale_factor=1)
    graph.glyph.scale_mode = 'scale_by_vector'

    if label is not None:
        label = (label.reshape(-1, 3) - mid_p).reshape(-1) / max_p + 0.5
        label[label < 0] = 0
        label[label >= 1] = 1
        label = (label * VOXEL_RES).astype(int)
        label = label.reshape(-1,3)
        label = label[:,0]+label[:,1]*VOXEL_RES+label[:,2]*VOXEL_RES*VOXEL_RES
        scale[label] = 1.5
    graph.mlab_source.dataset.point_data.vectors = np.repeat(scale, 3).reshape(-1,3)
    graph.mlab_source.dataset.point_data.scalars = data
    show()

def plot_voxel(data):
    x,y,z = np.where(data==1)
    scale = np.ones(x.shape[0])
    graph = points3d(x,y,z)
    show()


def plot_gt(gt):
    joint = gt.shape[0]
    scalar, scale, x, y, z = np.empty(0),np.empty(0),np.empty(0,dtype=np.int32),np.empty(0,dtype=np.int32),np.empty(0,dtype=np.int32)
    print gt.shape
    for i in range(joint):
        x1, y1, z1 = np.where(gt[i] >0)
        x = np.append(x,x1)
        y = np.append(y,y1)
        z = np.append(z,z1)
        scalar = np.append(scalar, np.zeros(x1.shape[0])+float(i)/JOINT_LEN)
        scale = np.append(scale, gt[i][x1,y1,z1])
    graph = points3d(x,y,z,
                     scale_factor=1)
    graph.glyph.scale_mode = 'scale_by_vector'

    graph.mlab_source.dataset.point_data.vectors = np.repeat(scale, 3).reshape(-1, 3)
    graph.mlab_source.dataset.point_data.scalars = scalar
    show()


def plot_voxel_label(data):
    x,y,z = np.where(data==1)
    x1,y1,z1 = np.where(data==2)
    scale = np.ones(x.shape[0]+x1.shape[0])
    scale[x.shape[0]:] = 2
    graph = points3d(np.append(x,x1),np.append(y,y1),np.append(z,z1),
                     scale_factor=1)
    graph.glyph.scale_mode = 'scale_by_vector'

    # if label is not None:
    #     label = (label.reshape(-1, 3) - mid_p).reshape(-1) / max_p + 0.5
    #     label[label < 0] = 0
    #     label[label >= 1] = 1
    #     label = (label * size).astype(int)
    #     label = label.reshape(-1, 3)
    #     label = label[:, 0] + label[:, 1] * size + label[:, 2] * size * size
    #     scale[label] = 1.5
    graph.mlab_source.dataset.point_data.vectors = np.repeat(scale, 3).reshape(-1, 3)
    graph.mlab_source.dataset.point_data.scalars = scale-1
    show()


def plot_voxel_label(data, label, min_p, voxel_length):
    label = label.reshape(-1,3).copy()
    label = label-np.repeat(np.expand_dims(min_p,axis=0),label.shape[0],axis=0)
    label = (label/voxel_length).astype(int)
    data = data.copy()
    data[label[:,0],label[:,1],label[:,2]] = 2
    x,y,z = np.where(data==1)
    x1,y1,z1 = np.where(data==2)
    scale = np.ones(x.shape[0]+x1.shape[0])
    scale[x.shape[0]:] = 2
    graph = points3d(np.append(x,x1),np.append(y,y1),np.append(z,z1),
                     scale_factor=1)
    graph.glyph.scale_mode = 'scale_by_vector'

    # if label is not None:
    #     label = (label.reshape(-1, 3) - mid_p).reshape(-1) / max_p + 0.5
    #     label[label < 0] = 0
    #     label[label >= 1] = 1
    #     label = (label * size).astype(int)
    #     label = label.reshape(-1, 3)
    #     label = label[:, 0] + label[:, 1] * size + label[:, 2] * size * size
    #     scale[label] = 1.5
    graph.mlab_source.dataset.point_data.vectors = np.repeat(scale, 3).reshape(-1, 3)
    graph.mlab_source.dataset.point_data.scalars = scale-1
    show()


def plot_voxel_result(data, label, mid_p, voxel_length):
    label = label.reshape(-1,3)-np.repeat(np.expand_dims(mid_p,axis=0),label.shape[0],axis=0)
    label = (label/voxel_length).astype(int)
    data[label[:,0],label[:,1],label[:,2]] = 2
    x,y,z = np.where(data==1)
    x1,y1,z1 = np.where(label==2)
    scale = np.ones(x.shape[0]+x1.shape[0])
    scale[x.shape[0]:] = 2
    graph = points3d(np.append(x,x1),np.append(y,y1),np.append(z,z1),
                     scale_factor=1)
    graph.glyph.scale_mode = 'scale_by_vector'

    # if label is not None:
    #     label = (label.reshape(-1, 3) - mid_p).reshape(-1) / max_p + 0.5
    #     label[label < 0] = 0
    #     label[label >= 1] = 1
    #     label = (label * size).astype(int)
    #     label = label.reshape(-1, 3)
    #     label = label[:, 0] + label[:, 1] * size + label[:, 2] * size * size
    #     scale[label] = 1.5
    graph.mlab_source.dataset.point_data.vectors = np.repeat(scale, 3).reshape(-1, 3)
    graph.mlab_source.dataset.point_data.scalars = scale-1
    show()


def plot_pointcloud(data, label = None):
    scale = np.ones(data.shape[0])/4
    if label is not None:
        label = label.reshape(-1,3)
        data = np.concatenate((data,label),axis=0)
        scale = np.concatenate((scale,np.ones(label.shape[0])),axis=0)
    graph = points3d(data[:,0],data[:,1],data[:,2],scale,scale_factor = 3)
    # graph.glyph.scale_mode = 'scale_by_vector'
    #
    # graph.mlab_source.dataset.point_data.vectors = np.repeat(scale, 3).reshape(-1, 3)
    # graph.mlab_source.dataset.point_data.scalars = scale
    show()