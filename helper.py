import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_param(dataset):
    if dataset == 'icvl':
        return 240.99, 240.96, 160, 120
    elif dataset == 'nyu':
        return 588.03, -587.07, 320, 240
    elif dataset == 'msra':
        return 241.42, 241.42, 160, 120


def pixel2camera(x, fx, fy, ux, uy):
    x[:, :, 0] = (x[:, :, 0] - ux) * x[:, :, 2] / fx
    x[:, :, 1] = -(x[:, :, 1] - uy) * x[:, :, 2] / fy
    x[:, :, 2] = -x[:, :, 2]
    return x


def camera2pixel(x, fx, fy, ux, uy):
    out = np.empty(shape=x.shape, dtype=np.int32)
    if len(x.shape) == 3:
        out[:, :, 0] = np.round(-x[:, :, 0] * fx / x[:, :, 2] + ux)
        out[:, :, 1] = np.round(x[:, :, 1] * fy / x[:, :, 2] + uy)
        out[:, :, 2] = np.round(-x[:, :, 2])
    elif len(x.shape) == 2:
        out[:, 0] = np.round(-x[:, 0] * fx / x[:, 2] + ux)
        out[:, 1] = np.round(x[:, 1] * fy / x[:, 2] + uy)
        out[:, 2] = np.round(-x[:, 2])
    return out