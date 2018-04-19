import torch
import numpy as np
# import cv2
# import ref
#
# sigma_inp = ref.hmGaussInp
# n = sigma_inp * 6 + 1
# g_inp = np.zeros((n, n))
# for i in range(n):
#     for j in range(n):
#         g_inp[i, j] = np.exp(-((i - n / 2) ** 2 + (j - n / 2) ** 2) / (2. * sigma_inp * sigma_inp))
#
# def GetTransform(center, scale, rot, res):
#   h = scale
#   t = np.eye(3)
#
#   t[0, 0] = res / h
#   t[1, 1] = res / h
#   t[0, 2] = res * (- center[0] / h + 0.5)
#   t[1, 2] = res * (- center[1] / h + 0.5)
#
#   if rot != 0:
#     rot = -rot
#     r = np.eye(3)
#     ang = rot * np.math.pi / 180
#     s = np.math.sin(ang)
#     c = np.math.cos(ang)
#     r[0, 0] = c
#     r[0, 1] = - s
#     r[1, 0] = s
#     r[1, 1] = c
#     t_ = np.eye(3)
#     t_[0, 2] = - res / 2
#     t_[1, 2] = - res / 2
#     t_inv = np.eye(3)
#     t_inv[0, 2] = res / 2
#     t_inv[1, 2] = res / 2
#     t = np.dot(np.dot(np.dot(t_inv,  r), t_), t)
#
#   return t
#
#
# def Transform(pt, center, scale, rot, res, invert = False):
#   pt_ = np.ones(3)
#   pt_[0], pt_[1] = pt[0], pt[1]
#
#   t = GetTransform(center, scale, rot, res)
#   if invert:
#     t = np.linalg.inv(t)
#   new_point = np.dot(t, pt_)[:2]
#   new_point = new_point.astype(np.int32)
#   return new_point
#
#
# def getTransform3D(center, scale, rot, res):
#   h = 1.0 * scale
#   t = np.eye(4)
#
#   t[0][0] = res / h
#   t[1][1] = res / h
#   t[2][2] = res / h
#
#   t[0][3] = res * (- center[0] / h + 0.5)
#   t[1][3] = res * (- center[1] / h + 0.5)
#
#   if rot != 0:
#     raise Exception('Not Implement')
#
#   return t
#
#
# def Transform3D(pt, center, scale, rot, res, invert = False):
#   pt_ = np.ones(4)
#   pt_[0], pt_[1], pt_[2] = pt[0], pt[1], pt[2]
#   #print 'c s r res', center, scale, rot, res
#   t = getTransform3D(center, scale, rot, res)
#   if invert:
#     t = np.linalg.inv(t)
#   #print 't', t
#   #print 'pt_', pt_
#   new_point = np.dot(t, pt_)[:3]
#   #print 'new_point', new_point
#   #if not invert:
#   #  new_point = new_point.astype(np.int32)
#   return new_point
#
#
# def Crop(img, center, scale, rot, res):
#   ht, wd = img.shape[0], img.shape[1]
#   tmpImg, newImg = img.copy(), np.zeros((res, res, 3), dtype = np.uint8)
#
#   scaleFactor = scale / res
#   if scaleFactor < 2:
#     scaleFactor = 1
#   else:
#     newSize = int(np.math.floor(max(ht, wd) / scaleFactor))
#     newSize_ht = int(np.math.floor(ht / scaleFactor))
#     newSize_wd = int(np.math.floor(wd / scaleFactor))
#     if newSize < 2:
#       return torch.from_numpy(newImg.transpose(2, 0, 1).astype(np.float32) / 256.)
#     else:
#       tmpImg = cv2.resize(tmpImg, (newSize_wd, newSize_ht)) #TODO
#       ht, wd = tmpImg.shape[0], tmpImg.shape[1]
#
#   c, s = 1.0 * center / scaleFactor, scale / scaleFactor
#   c[0], c[1] = c[1], c[0]
#   ul = Transform((0, 0), c, s, 0, res, invert = True)
#   br = Transform((res, res), c, s, 0, res, invert = True)
#
#   if scaleFactor >= 2:
#     br = br - (br - ul - res)
#
#   pad = int(np.math.ceil((((ul - br) ** 2).sum() ** 0.5) / 2 - (br[0] - ul[0]) / 2))
#   if rot != 0:
#     ul = ul - pad
#     br = br + pad
#
#   old_ = [max(0, ul[0]),   min(br[0], ht),         max(0, ul[1]),   min(br[1], wd)]
#   new_ = [max(0, - ul[0]), min(br[0], ht) - ul[0], max(0, - ul[1]), min(br[1], wd) - ul[1]]
#
#   newImg = np.zeros((br[0] - ul[0], br[1] - ul[1], 3), dtype = np.uint8)
#   #print 'new old newshape tmpshape center', new_[0], new_[1], old_[0], old_[1], newImg.shape, tmpImg.shape, center
#   try:
#     newImg[new_[0]:new_[1], new_[2]:new_[3], :] = tmpImg[old_[0]:old_[1], old_[2]:old_[3], :]
#   except:
#     #print 'ERROR: new old newshape tmpshape center', new_[0], new_[1], old_[0], old_[1], newImg.shape, tmpImg.shape, center
#     return np.zeros((3, res, res), np.uint8)
#   if rot != 0:
#     M = cv2.getRotationMatrix2D((newImg.shape[0] / 2, newImg.shape[1] / 2), rot, 1)
#     newImg = cv2.warpAffine(newImg, M, (newImg.shape[0], newImg.shape[1]))
#     newImg = newImg[pad+1:-pad+1, pad+1:-pad+1, :].copy()
#
#   if scaleFactor < 2:
#     newImg = cv2.resize(newImg, (res, res))
#
#   return newImg.transpose(2, 0, 1).astype(np.float32)

def Gaussian(sigma):
  if sigma == 7:
    return np.array([0.0529,  0.1197,  0.1954,  0.2301,  0.1954,  0.1197,  0.0529,
                     0.1197,  0.2709,  0.4421,  0.5205,  0.4421,  0.2709,  0.1197,
                     0.1954,  0.4421,  0.7214,  0.8494,  0.7214,  0.4421,  0.1954,
                     0.2301,  0.5205,  0.8494,  1.0000,  0.8494,  0.5205,  0.2301,
                     0.1954,  0.4421,  0.7214,  0.8494,  0.7214,  0.4421,  0.1954,
                     0.1197,  0.2709,  0.4421,  0.5205,  0.4421,  0.2709,  0.1197,
                     0.0529,  0.1197,  0.1954,  0.2301,  0.1954,  0.1197,  0.0529]).reshape(7, 7)
  elif sigma == 5:
    return np.array([0.0937, 0.2277, 0.3062, 0.2277, 0.0937,
                     0.2277, 0.5533, 0.7438, 0.5533, 0.2277,
                     0.3062, 0.7438, 1.0, 0.7438, 0.3062,
                     0.2277, 0.5533, 0.7438, 0.5533, 0.2277,
                     0.0937, 0.2277, 0.3062, 0.2277, 0.0937,]).reshape(5,5)
  else:
    raise Exception('Gaussian {} Not Implement'.format(sigma))


def DrawGaussian(img, pt, g_size):
  tmpSize = int((g_size-1)/2)
  ul = [int(np.math.floor(pt[0] - tmpSize)), int(np.math.floor(pt[1] - tmpSize))]
  br = [int(np.math.floor(pt[0] + tmpSize+1)), int(np.math.floor(pt[1] + tmpSize+1))]
  # print ul,br
  if ul[0] > img.shape[1] or ul[1] > img.shape[0] or br[0] < 1 or br[1] < 1:
    return img
  
  g = Gaussian(g_size)
  
  g_x = [max(0, -ul[0]), min(br[0], img.shape[1]) - max(0, ul[0]) + max(0, -ul[0])]
  g_y = [max(0, -ul[1]), min(br[1], img.shape[0]) - max(0, ul[1]) + max(0, -ul[1])]
  img_x = [max(0, ul[0]), min(br[0], img.shape[1])]
  img_y = [max(0, ul[1]), min(br[1], img.shape[0])]
  
  img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
  return img


def Gaussian3D(size):
    if size == 7:
        r = np.array([0.0342, 0.0639, 0.0930, 0.1053, 0.0930, 0.0639, 0.0342, 0.0639, 0.1194, 0.1737, 0.1969, 0.1737, 0.1194, 0.0639, 0.0930, 0.1737, 0.2528, 0.2865, 0.2528, 0.1737, 0.0930, 0.1053, 0.1969, 0.2865, 0.3246, 0.2865, 0.1969, 0.1053, 0.0930, 0.1737, 0.2528, 0.2865, 0.2528, 0.1737, 0.0930, 0.0639, 0.1194, 0.1737, 0.1969, 0.1737, 0.1194, 0.0639, 0.0342, 0.0639, 0.0930, 0.1053, 0.0930, 0.0639, 0.0342, 0.0639, 0.1194, 0.1737, 0.1969, 0.1737, 0.1194, 0.0639, 0.1194, 0.2231, 0.3246, 0.3678, 0.3246, 0.2231, 0.1194, 0.1737, 0.3246, 0.4723, 0.5352, 0.4723, 0.3246, 0.1737, 0.1969, 0.3678, 0.5352, 0.6065, 0.5352, 0.3678, 0.1969, 0.1737, 0.3246, 0.4723, 0.5352, 0.4723, 0.3246, 0.1737, 0.1194, 0.2231, 0.3246, 0.3678, 0.3246, 0.2231, 0.1194, 0.0639, 0.1194, 0.1737, 0.1969, 0.1737, 0.1194, 0.0639, 0.0930, 0.1737, 0.2528, 0.2865, 0.2528, 0.1737, 0.0930, 0.1737, 0.3246, 0.4723, 0.5352, 0.4723, 0.3246, 0.1737, 0.2528, 0.4723, 0.6872, 0.7788, 0.6872, 0.4723, 0.2528, 0.2865, 0.5352, 0.7788, 0.8824, 0.7788, 0.5352, 0.2865, 0.2528, 0.4723, 0.6872, 0.7788, 0.6872, 0.4723, 0.2528, 0.1737, 0.3246, 0.4723, 0.5352, 0.4723, 0.3246, 0.1737, 0.0930, 0.1737, 0.2528, 0.2865, 0.2528, 0.1737, 0.0930, 0.1053, 0.1969, 0.2865, 0.3246, 0.2865, 0.1969, 0.1053, 0.1969, 0.3678, 0.5352, 0.6065, 0.5352, 0.3678, 0.1969, 0.2865, 0.5352, 0.7788, 0.8824, 0.7788, 0.5352, 0.2865, 0.3246, 0.6065, 0.8824, 1.0, 0.8824, 0.6065, 0.3246, 0.2865, 0.5352, 0.7788, 0.8824, 0.7788, 0.5352, 0.2865, 0.1969, 0.3678, 0.5352, 0.6065, 0.5352, 0.3678, 0.1969, 0.1053, 0.1969, 0.2865, 0.3246, 0.2865, 0.1969, 0.1053, 0.0930, 0.1737, 0.2528, 0.2865, 0.2528, 0.1737, 0.0930, 0.1737, 0.3246, 0.4723, 0.5352, 0.4723, 0.3246, 0.1737, 0.2528, 0.4723, 0.6872, 0.7788, 0.6872, 0.4723, 0.2528, 0.2865, 0.5352, 0.7788, 0.8824, 0.7788, 0.5352, 0.2865, 0.2528, 0.4723, 0.6872, 0.7788, 0.6872, 0.4723, 0.2528, 0.1737, 0.3246, 0.4723, 0.5352, 0.4723, 0.3246, 0.1737, 0.0930, 0.1737, 0.2528, 0.2865, 0.2528, 0.1737, 0.0930, 0.0639, 0.1194, 0.1737, 0.1969, 0.1737, 0.1194, 0.0639, 0.1194, 0.2231, 0.3246, 0.3678, 0.3246, 0.2231, 0.1194, 0.1737, 0.3246, 0.4723, 0.5352, 0.4723, 0.3246, 0.1737, 0.1969, 0.3678, 0.5352, 0.6065, 0.5352, 0.3678, 0.1969, 0.1737, 0.3246, 0.4723, 0.5352, 0.4723, 0.3246, 0.1737, 0.1194, 0.2231, 0.3246, 0.3678, 0.3246, 0.2231, 0.1194, 0.0639, 0.1194, 0.1737, 0.1969, 0.1737, 0.1194, 0.0639, 0.0342, 0.0639, 0.0930, 0.1053, 0.0930, 0.0639, 0.0342, 0.0639, 0.1194, 0.1737, 0.1969, 0.1737, 0.1194, 0.0639, 0.0930, 0.1737, 0.2528, 0.2865, 0.2528, 0.1737, 0.0930, 0.1053, 0.1969, 0.2865, 0.3246, 0.2865, 0.1969, 0.1053, 0.0930, 0.1737, 0.2528, 0.2865, 0.2528, 0.1737, 0.0930, 0.0639, 0.1194, 0.1737, 0.1969, 0.1737, 0.1194, 0.0639, 0.0342, 0.0639, 0.0930, 0.1053, 0.0930, 0.0639, 0.0342])

    elif size == 5:
        r = np.array([0.0694, 0.1353, 0.1690, 0.1353, 0.0694, 0.1353, 0.2635, 0.3291, 0.2635, 0.1353, 0.1690, 0.3291, 0.4111, 0.3291, 0.1690, 0.1353, 0.2635, 0.3291, 0.2635, 0.1353, 0.0694, 0.1353, 0.1690, 0.1353, 0.0694, 0.1353, 0.2635, 0.3291, 0.2635, 0.1353, 0.2635, 0.5134, 0.6411, 0.5134, 0.2635, 0.3291, 0.6411, 0.8007, 0.6411, 0.3291, 0.2635, 0.5134, 0.6411, 0.5134, 0.2635, 0.1353, 0.2635, 0.3291, 0.2635, 0.1353, 0.1690, 0.3291, 0.4111, 0.3291, 0.1690, 0.3291, 0.6411, 0.8007, 0.6411, 0.3291, 0.4111, 0.8007, 1.0, 0.8007, 0.4111, 0.3291, 0.6411, 0.8007, 0.6411, 0.3291, 0.1690, 0.3291, 0.4111, 0.3291, 0.1690, 0.1353, 0.2635, 0.3291, 0.2635, 0.1353, 0.2635, 0.5134, 0.6411, 0.5134, 0.2635, 0.3291, 0.6411, 0.8007, 0.6411, 0.3291, 0.2635, 0.5134, 0.6411, 0.5134, 0.2635, 0.1353, 0.2635, 0.3291, 0.2635, 0.1353, 0.0694, 0.1353, 0.1690, 0.1353, 0.0694, 0.1353, 0.2635, 0.3291, 0.2635, 0.1353, 0.1690, 0.3291, 0.4111, 0.3291, 0.1690, 0.1353, 0.2635, 0.3291, 0.2635, 0.1353, 0.0694, 0.1353, 0.1690, 0.1353, 0.0694])
    else:
        raise Exception('Gaussian {} Not Implement'.format(size))
    return r.reshape(size,size,size)

def DrawGaussian3D(size, pt, g_size):
    tmpSize = int((g_size - 1) / 2)
    pt = np.array(pt)
    min_p = np.floor(pt-tmpSize).astype(np.int32)
    max_p = np.floor(pt+tmpSize).astype(np.int32)+1
    if pt[0] < 0 or pt[0] >= size or pt[1] < 0 or pt[1] >= size or pt[2] < 0 or pt[2] >= size:
        return None
    g = Gaussian3D(g_size)
    v = np.zeros((size,size,size), dtype=np.float32)
    g_x = [max(0, -min_p[0]), min(max_p[0], size) - max(0, min_p[0]) + max(0, -min_p[0])]
    g_y = [max(0, -min_p[1]), min(max_p[1], size) - max(0, min_p[1]) + max(0, -min_p[1])]
    g_z = [max(0, -min_p[2]), min(max_p[2], size) - max(0, min_p[2]) + max(0, -min_p[2])]
    v_x = [max(0, min_p[0]), min(max_p[0], size)]
    v_y = [max(0, min_p[1]), min(max_p[1], size)]
    v_z = [max(0, min_p[2]), min(max_p[2], size)]
    v[v_x[0]:v_x[1],v_y[0]:v_y[1],v_z[0]:v_z[1]] = g[g_x[0]:g_x[1], g_y[0]:g_y[1], g_z[0]:g_z[1]]
    return v

def DrawBoneGaussian3D(size, pt1, pt2, g_size):
    ps,pe = pt1, pt2
    v = DrawGaussian3D(size,pt1,g_size)
    while not np.array_equal(ps,pe):
        d = pe-ps
        d = d / np.linalg.norm(d, 2)
        ps = np.round(d + ps).astype(np.int)
        vt = DrawGaussian3D(size, ps, g_size)
        v = np.maximum(v,vt)
    return v

def generate_gaussian3D(size):
    s = -(size-1)/2
    e = (size-1)/2
    a = np.linspace(s, e, size)
    x, y, z = np.meshgrid(a,a,a,indexing='ij')
    d = np.sqrt(x * x + y * y+z*z)
    sigma, mu = 1.5, 0.0
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    return g

def generate_gaussian(size):
    s = -(size-1)/2
    e = (size-1)/2
    a = np.linspace(s, e, size)
    x, y= np.meshgrid(a,a)
    d = np.sqrt(x * x + y * y)
    sigma, mu = 1.3, 0.0
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    return g
# print Gaussian3D(7)
# np.set_printoptions(precision=4, suppress=True)
# g = generate_gaussian(5)
# for i in g:
#     for j in i:
#         print str(j)[:6]+',',
#     print

# p1 = np.array([1,2,3])
# p2 = np.array([1,2,4])
# d = p2-p1
# d = d/np.linalg.norm(d,2)
# print np.round(d + p1)
# print d