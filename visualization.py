from mayavi.mlab import *
import numpy as np
from params import *


def plot_tsdf(data, label = None):
    data = np.reshape(data+1,-1)/2
    graph = points3d(np.tile(np.arange(-VOXEL_RES / 2, VOXEL_RES / 2), (VOXEL_RES * VOXEL_RES)),
             np.tile(np.repeat(np.arange(-VOXEL_RES / 2, VOXEL_RES / 2), VOXEL_RES), VOXEL_RES),
             np.repeat(np.arange(-VOXEL_RES / 2, VOXEL_RES / 2), (VOXEL_RES * VOXEL_RES)),
                     scale_factor=1)
    graph.glyph.scale_mode = 'scale_by_vector'
    scale = np.where(data == 1, 0, 0.5)

    if label is not None:
        label = (label * VOXEL_RES).astype(int)
        label = label.reshape(-1,3)
        label = label[:,0]+label[:,1]*VOXEL_RES+label[:,2]*VOXEL_RES*VOXEL_RES
        scale[label] = 1.5
    graph.mlab_source.dataset.point_data.vectors = np.repeat(scale, 3).reshape(-1,3)
    graph.mlab_source.dataset.point_data.scalars = data
    show()


def plot_pointcloud(data, label = None):
    scale = np.ones(data.shape[0])/4
    if label is not None:
        label = label.reshape(-1,3)
        data = np.concatenate((data,label),axis=0)
        scale = np.concatenate((scale,np.ones(label.shape[0])),axis=0)
    graph = points3d(data[:,0],data[:,1],data[:,2],scale,scale_factor = 3)
    graph = points3d(data[:,0],data[:,1],data[:,2],scale,scale_factor = 3)
    # graph.glyph.scale_mode = 'scale_by_vector'
    #
    # graph.mlab_source.dataset.point_data.vectors = np.repeat(scale, 3).reshape(-1, 3)
    # graph.mlab_source.dataset.point_data.scalars = scale
    show()