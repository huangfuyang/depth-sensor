import numpy as np
from params import *
import torch


def accuracy_portion(output, target, t=ERROR_THRESH):
    diff = np.abs(output - target).reshape(-1,3)
    sqr_sum = np.sum(np.square(diff),axis=1)
    out = np.zeros(sqr_sum.size)
    t = t**2
    out[sqr_sum < t] = 1
    good = np.sum(out) / out.size
    return good * 100


def accuracy_error_thresh_portion_batch(output, target, max_l, t=ERROR_THRESH):
    batch_size = target.size(0)
    sample_size = target.size(1)
    max_l = max_l.repeat(sample_size).view(batch_size,sample_size).cuda()
    diff = torch.abs(output-target)
    diff = torch.mul(diff,max_l).view(batch_size,-1,3)
    sqr_sum = torch.sum(torch.pow(diff,2),2)
    out = torch.zeros(sqr_sum.size()).cuda()
    t = t**2
    out[sqr_sum<t] = 1
    good = torch.sum(out)/(out.size(1)*batch_size)
    return good*100


def good_frame(output, target, max_l, t = ERROR_THRESH):
    batch_size = target.size(0)
    sample_size = target.size(1)
    max_l = max_l.repeat(sample_size).view(batch_size, sample_size).cuda()
    diff = torch.abs(output - target)
    diff = torch.mul(diff, max_l).view(batch_size, -1, 3)
    sqr_sum = torch.sum(torch.pow(diff, 2), 2)
    out = torch.zeros(sqr_sum.size()).cuda()
    t = t ** 2
    out[sqr_sum > t] = 1
    out = torch.sum(out, 1)
    out[out>0] = 1
    good = 1-torch.sum(out) / batch_size
    return good * 100


def mean_error(output,target, max_l):
    batch_size = target.size(0)
    sample_size = target.size(1)
    max_l = max_l.repeat(sample_size).view(batch_size, sample_size).cuda()
    diff = torch.abs(output - target)
    diff = torch.mul(diff, max_l).view(batch_size, -1, 3)
    sqr_sum = torch.sum(torch.pow(diff, 2), 2)
    sqrt_row = torch.sqrt(sqr_sum)
    # print torch.sum(sqrt_row,0)/batch_size
    # out = torch.sum(sqrt_row)
    if batch_size != 0:
        return torch.sum(sqrt_row,0)/batch_size
    else:
        return 0