import numpy as np
import numba as nb
from params import *
from timeit import default_timer as timer
from numba import cuda

threadsperblock = VOXEL_RES
mm_threadsperblock = 128

@cuda.jit
def tsdf_kernel(vox_ori,voxel_len,l,r,t,b,trunc_dis,i,o):
    b_w = r-l
    x = cuda.threadIdx.x
    y = cuda.blockIdx.x
    z = cuda.blockIdx.y
    v_x = vox_ori[0]+x * voxel_len
    v_y = vox_ori[1]+y * voxel_len
    v_z = vox_ori[2]+z * voxel_len
    vox_depth = -v_z
    q = FOCAL / vox_depth
    pix_x = int((v_x * q + CENTER_X))
    pix_y = int((-v_y * q + CENTER_Y))
    o[z, y, x] = 1
    if pix_x < l or pix_x >= r or pix_y < t or pix_y >= b:
        return
    idx = (pix_y - t) * b_w + pix_x - l
    pix_depth = i[idx]
    if pix_depth == 0:
        return
    diff = (pix_depth - vox_depth)/trunc_dis
    if diff >= 1 or diff <= -1:
        return
    o[z, y, x] = diff


@cuda.jit
def min_max_kernel(l,r,t,b,a,o):
    b_w = r-l
    tid = cuda.threadIdx.x
    min_p = cuda.shared.array(shape=(mm_threadsperblock,3), dtype=nb.types.float32)
    max_p = cuda.shared.array(shape=(mm_threadsperblock,3), dtype=nb.types.float32)
    pos = cuda.grid(1)
    x = pos % b_w + l
    y = pos / b_w + t
    cam_z = a[pos]
    if cam_z == 0:
        min_p[tid][0],min_p[tid][1],min_p[tid][2] = 99999,99999,99999
        max_p[tid][0],max_p[tid][1],max_p[tid][2] = -99999,-99999,-99999
    else:
        q = cam_z / FOCAL
        cam_x = q * (x - CENTER_X)
        cam_y = -q * (y - CENTER_Y)  # change to right hand axis
        cam_z = -cam_z
        min_p[tid][0], min_p[tid][1], min_p[tid][2] = cam_x, cam_y, cam_z
        max_p[tid][0], max_p[tid][1], max_p[tid][2] = cam_x, cam_y, cam_z
    cuda.syncthreads()
    s = 1
    while s < cuda.blockDim.x:
        index = 2 * s * tid
        if index + s < cuda.blockDim.x:
            min_p[index][0] = min(min_p[index + s][0], min_p[index][0])
            min_p[index][1] = min(min_p[index + s][1], min_p[index][1])
            min_p[index][2] = min(min_p[index + s][2], min_p[index][2])
            max_p[index][0] = max(max_p[index + s][0], max_p[index][0])
            max_p[index][1] = max(max_p[index + s][1], max_p[index][1])
            max_p[index][2] = max(max_p[index + s][2], max_p[index][2])
        s *= 2
        cuda.syncthreads()
    if tid == 0:
        o[cuda.blockIdx.x][0] = min_p[0][0]
        o[cuda.blockIdx.x][1] = min_p[0][1]
        o[cuda.blockIdx.x][2] = min_p[0][2]
        o[cuda.blockIdx.x][3] = max_p[0][0]
        o[cuda.blockIdx.x][4] = max_p[0][1]
        o[cuda.blockIdx.x][5] = max_p[0][2]


def cal_tsdf_cuda(s):
    t1 = timer()
    s, label = s
    w = s['header'][0]
    h = s['header'][1]
    l = s['header'][2]
    t = s['header'][3]
    r = s['header'][4]
    b = s['header'][5]
    b_w = r - l
    b_h = b - t
    blockdim = int((s['data'].size + mm_threadsperblock -1)/mm_threadsperblock)
    d_depth = cuda.to_device(s['data'])
    d_mm = cuda.device_array([blockdim, 6],dtype=np.float32)
    min_max_kernel[blockdim,mm_threadsperblock](l,r,t,b,d_depth,d_mm)
    mm_p = np.empty([blockdim,6],dtype=np.float32)
    mm_p = d_mm.copy_to_host()
    min_p = mm_p[0,:3]
    max_p = mm_p[0,3:6]
    for i in mm_p:
        min_p[0] = min(i[0], min_p[0])
        min_p[1] = min(i[1], min_p[1])
        min_p[2] = min(i[2], min_p[2])
        max_p[0] = max(i[3], max_p[0])
        max_p[1] = max(i[4], max_p[1])
        max_p[2] = max(i[5], max_p[2])
    mid_p = (min_p + max_p) / 2
    len_e = max_p - min_p
    max_l = np.max(len_e)
    voxel_len = max_l / VOXEL_RES
    trunc_dis = (voxel_len * TRUC_DIS_T)
    vox_ori = mid_p - max_l / 2 + voxel_len / 2
    t2 = timer()
    d_depth = cuda.to_device(s['data'])
    d_tsdf = cuda.device_array([VOXEL_RES ,VOXEL_RES , VOXEL_RES],dtype=np.float32)
    blockspergrid = [VOXEL_RES,VOXEL_RES]
    tsdf_kernel[blockspergrid,threadsperblock](vox_ori,voxel_len,l,r,t,b,trunc_dis,d_depth,d_tsdf)
    label = (label.reshape(-1, 3) - mid_p) / max_l + 0.5
    label[label<0] = 0
    label[label>=1] = 1
    tsdf = np.empty([VOXEL_RES ,VOXEL_RES , VOXEL_RES],dtype=np.float32)
    tsdf = d_tsdf.copy_to_host()
    t3 = timer()
    # print "time 1: %.4f, time 1+2: %.4f" % (t2 - t1, t3 - t1)

    return tsdf, label.reshape(-1), (mid_p,max_l)

def cal_pointcloud(s):
    l = s['header'][2]
    t = s['header'][3]
    r = s['header'][4]
    b = s['header'][5]
    b_w = r - l
    p_clouds = []
    for y in range(t, b):
        for x in range(l, r):
            idx = (y - t) * b_w + x - l
            cam_z = s['data'][idx]
            if cam_z == 0:
                continue
            q = cam_z / FOCAL
            cam_x = q * (x - CENTER_X)
            cam_y = -q * (y - CENTER_Y)  # change to right hand axis
            cam_z = -cam_z
            p_clouds.append([cam_x, cam_y, cam_z])
    npa = np.asarray(p_clouds, dtype=np.float32)
    return npa

# return point cloud in camera coordination and tsdf data
def cal_tsdf(s):
    s, label = s
    w = s['header'][0]
    h = s['header'][1]
    l = s['header'][2]
    t = s['header'][3]
    r = s['header'][4]
    b = s['header'][5]
    b_w = r-l
    b_h = b-t
    p_clouds = []
    minx=miny=minz = 99999
    maxx=maxy=maxz = -99999
    for y in range(t,b):
        for x in range(l,r):
            idx = (y-t) * b_w + x-l
            cam_z = s['data'][idx]
            if cam_z == 0:
                continue
            q = cam_z/FOCAL
            cam_x = q * (x-CENTER_X)
            cam_y = -q * (y-CENTER_Y) # change to right hand axis
            cam_z = -cam_z
            p_clouds.append([cam_x,cam_y,cam_z])
            minx = cam_x if cam_x<minx else minx
            miny = cam_y if cam_y<miny else miny
            minz = cam_z if cam_z<minz else minz

            maxx = cam_x if cam_x > maxx else maxx
            maxy = cam_y if cam_y > maxy else maxy
            maxz = cam_z if cam_z > maxz else maxz
    npa = np.asarray(p_clouds,dtype=np.float32)
    min_p = np.array([minx,miny,minz],dtype=np.float32)
    max_p = np.array([maxx,maxy,maxz],dtype=np.float32)
    mid_p = (min_p+max_p)/2
    len_e = max_p-min_p
    max_l = np.max(len_e)
    voxel_len = max_l/VOXEL_RES
    trunc_dis = voxel_len*TRUC_DIS_T
    vox_ori = mid_p-max_l/2+voxel_len/2
    tsdf = np.ones((VOXEL_RES,VOXEL_RES,VOXEL_RES))

    # print VOXEL_RES*VOXEL_RES*VOXEL_RES,"loop 2 iter"
    for z in range(VOXEL_RES):
        for y in range(VOXEL_RES):
            for x in range(VOXEL_RES):
                vox_center = vox_ori + np.array([x*voxel_len,y*voxel_len,z*voxel_len])
                vox_depth = -vox_center[2]
                q = FOCAL/vox_depth
                pix_x = int(round(vox_center[0]*q+CENTER_X))
                pix_y = int(round(-vox_center[1]*q+CENTER_Y))
                if pix_x <l or pix_x>=r or pix_y<t or pix_y>=b:
                    continue
                idx = (pix_y-t)*b_w+pix_x-l
                # print vox_center,x,y,z,pix_x,pix_y,idx
                pix_depth = s['data'][idx]
                if pix_depth == 0:
                    continue
                diff = (pix_depth - vox_depth)/trunc_dis
                if diff >= 1 or diff <= -1:
                    continue
                tsdf[z,y,x] = diff

    label = (label.reshape(-1,3) - mid_p)/max_l+0.5
    return tsdf,label.reshape(-1)


# point cloud in image coordination
def get_project_data(s):
    w = s['header'][0]
    h = s['header'][1]
    l = s['header'][2]
    t = s['header'][3]
    r = s['header'][4]
    b = s['header'][5]
    b_w = r - l
    b_h = b - t
    p_clouds = []
    for y in range(t, b):
        for x in range(l, r):
            idx = (y - t) * b_w + x - l
            z = s['data'][idx]
            if z == 0:
                continue
            p_clouds.append([x, y, z])

    npa = np.asarray(p_clouds, dtype=np.float32)
    return npa


if __name__ == "__main__":
    pass
    # sample = {'header': np.array([320,240,100,100,200,200],dtype=np.int32),
    #           'data': np.random.rand(100*100)*100+100}
    # label = np.ones(63)
    # # pc, tsdf, labels = cal_tsdf((sample, label))
    # tsdf, labels = cal_tsdf_cuda((sample, label))

