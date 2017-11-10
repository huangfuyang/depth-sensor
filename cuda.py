from numba import cuda
import numpy as np
import numba as nb
from time import time

threadsperblock = 128


@cuda.jit
def increment_by_one_gpu(an_array):
    """
    Increment all array elements by one.
    """
    pos = cuda.grid(1)
    if pos < an_array.size:
        an_array[pos] += 1

@cuda.jit
def sum_gpu(a,b):
    tid = cuda.threadIdx.x
    sA = cuda.shared.array(shape=threadsperblock, dtype=nb.types.float32)
    pos = cuda.grid(1)
    sA[tid] = a[pos]
    cuda.syncthreads()
    s = 1
    while s<cuda.blockDim.x:
        index = 2*s*tid
        if index+s < cuda.blockDim.x:
            sA[index] += sA[index+s]
        s*=2
        cuda.syncthreads()
    if tid == 0:
        b[cuda.blockIdx.x] = sA[0]

@cuda.jit
def add_3d(p,a,b):
    x = cuda.threadIdx.x
    y = cuda.blockIdx.x
    z = cuda.blockIdx.y
    b[z,y,x] = 1
    if a[x+y*32+z*32*32] >0:
        b[z, y, x] += a[x+y*32+z*32*32]/p[0]
        return
    else:
        b[z, y, x] += a[x+y*32+z*32*32]/p[0]
    b[z,y,x] = 1


def sum(a):
    s = 0
    for i in a:
        s+=i
    return s


def increment_by_one(an_array):
    for i in an_array:
        i += 1


a = np.ones(10000003)
b = np.ones(10000005)
d_a = cuda.to_device(a)
d_b = cuda.to_device(b)
an_array = a
t = time()
print sum(an_array)
# increment_by_one(an_array)
print time()-t
t = time()
blockspergrid = (an_array.size + (threadsperblock - 1)) / threadsperblock
print threadsperblock,blockspergrid
# increment_by_one_gpu[blockspergrid, threadsperblock](b)
# increment_by_one_gpu[blockspergrid, threadsperblock](b)
d_output_a = cuda.device_array(shape=blockspergrid)
d_output_b = cuda.device_array(shape=blockspergrid)
# output = np.empty(threadsperblock)
sum_gpu[blockspergrid, threadsperblock](d_a,d_output_a)
print time()-t
t= time()

sum_gpu[blockspergrid, threadsperblock](d_b,d_output_b)
output = np.empty(threadsperblock)
output = d_output_a.copy_to_host()
print time()-t
output = d_output_b.copy_to_host()
t = time()
i = np.random.rand(32*32*32)
d_depth = cuda.to_device(a)
d_tsdf = cuda.device_array([32,32,32])
center = np.array([2,2,3])
blockspergrid = [32,32]
add_3d[blockspergrid,32](center,d_depth,d_tsdf)
tsdf = np.empty([32,32,32])
tsdf = d_tsdf.copy_to_host()
print time()-t
# print sum(output)