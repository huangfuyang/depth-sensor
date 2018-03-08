import scipy.io
import numpy as np
m = scipy.io.loadmat('tsdf_xyz_1.mat')
m = m['tsdf_xyz']
print m.shape
print m[m==-1].shape