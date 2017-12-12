import argparse
from network import *
import torch.optim as optim
import shutil
import datetime
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from params import *
from datareader import *
from metrics import *
import warnings


USE_SENSOR = False


def init_parser():
    parser = argparse.ArgumentParser(description='TSDF Fusion')
    parser.add_argument('-data',default=DATA_DIR, type=str, metavar='DIR',
                        help='path to dataset(default: {})'.format(DATA_DIR))

    parser.add_argument('-e', '--epochs',  default=EPOCH_COUNT, type=int,
                        help='number of total epochs to run (default: {})'.format(EPOCH_COUNT))

    parser.add_argument('-s', '--start-epoch',  default=0, type=int,
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('-b', '--batch-size', default=BATCH_SIZE, type=int,
                         help='mini-batch size (default: {})'.format(BATCH_SIZE))

    parser.add_argument('-lr', '--learning-rate', default=LEARNING_RATE, type=float,
                        metavar='LR', help='initial learning rate (default: {})'.format(LEARNING_RATE))

    parser.add_argument('-m', '--momentum', default=MOMENTUM, type=float, metavar='M',
                        help='momentum (default: {})'.format(MOMENTUM))

    parser.add_argument('-wd', '--weight-decay', default=WEIGHT_DECAY, type=float,
                        metavar='W', help='weight decay (default: {})'.format(WEIGHT_DECAY))

    parser.add_argument('-p', '--print-freq', default=PRINT_FREQ, type=int,
                         help='print frequency (default: {})'.format(PRINT_FREQ))

    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    global args
    args = parser.parse_args()


def warning_init():
    warnings.filterwarnings("error")
    np.seterr(all='warn')


def main(Model):
    warning_init()
    start_time = time()
    init_parser()
    net = Model()
    net.cuda()
    criterion = nn.MSELoss().cuda()
    best_acc = 0
    optimizer = optim.SGD(net.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # resume from checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    USE_SENSOR = True if type(net) is not HandNet else False
    dataset = MSRADataSet(args.data,USE_SENSOR, use_preprocessing=False)

    train_idx, valid_idx = dataset.get_train_test_indices()
    # train_idx = range(2)
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(valid_idx)

    # train_loader = torch.utils.data.DataLoader(dataset,
    #                                            batch_size=args.batch_size,
    #                                            num_workers=WORKER)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=args.batch_size, sampler=train_sampler,
                                               num_workers=WORKER)

    test_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=args.batch_size, sampler=test_sampler,
                                               num_workers=WORKER)

    for epoch in range(args.start_epoch, args.epochs):  # loop over the dataset multiple times
        epoch_start_time = time()
        adjust_learning_rate(optimizer, epoch+1)

        acc = train(train_loader,net,criterion,optimizer,epoch+1)
        # remember best acc and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        print('Epoch: [{0}/{1}]  Time [{2}/{3}]'.format(
            epoch+1, args.epochs, datetime.timedelta(seconds=(time() - epoch_start_time)),
                                datetime.timedelta(seconds=(time() - start_time))))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': '3dDNN',
            'state_dict': net.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best)
    print('Finished Training')
    print 'evaluating test dataset'
    result, acc = test(test_loader,net,criterion)
    print "final accuracy {:3f}".format(acc)
    print "total time: ",datetime.timedelta(seconds=(time()-start_time))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    # switch to train mode
    model.train()

    end = time()
    for i, s in enumerate(train_loader):
        if len(s) == 3:
            tsdf, target, max_l = s
        else:
            tsdf, target, max_l, mid_p = s
            mid_p = mid_p.unsqueeze(1)
        max_l = max_l.unsqueeze(1)
        # measure data loading time
        data_time.update(time() - end)
        if type(model) is not HandNet:
            tsdf, angles = tsdf
            input_sensor_var = torch.autograd.Variable(angles.cuda())
        batch_size = tsdf.size(0)
        tsdf = tsdf.unsqueeze_(1).cuda()

        # normalize target to [0,1]
        n_target = (target.view(batch_size, -1, 3) - mid_p).view(batch_size,-1) / max_l + 0.5
        n_target[n_target < 0] = 0
        n_target[n_target >= 1] = 1

        input_var = torch.autograd.Variable(tsdf)
        target_var = torch.autograd.Variable(n_target.cuda())

        # compute output
        if type(model) is HandSensorNet:
            output = model((input_var,input_sensor_var))
        elif type(model) is HandNet:
            output = model(input_var)
        elif type(model) is SensorNet:
            output = model(input_sensor_var)

        # record loss
        loss = criterion(output, target_var)
        losses.update(loss.data[0], batch_size)

        # measure accuracy
        # unnormalize output to original space
        output = ((output.data.cpu() - 0.5) * max_l).view(batch_size,-1,3) + mid_p
        output = output.view(batch_size,-1)
        err_t = accuracy_error_thresh_portion_batch(output, target)
        errors.update(err_t, batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time() - end)
        end = time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'acc_in_t {err_t.val:.3f} ({err_t.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, err_t=errors))
    return errors.avg


def test(test_loader, model, criterion, error = accuracy_error_thresh_portion_batch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time()

    result = np.empty(shape=(0,JOINT_POS_LEN),dtype=np.float32)
    for i, s in enumerate(test_loader):
        if len(s) == 3:
            tsdf, target, max_l = s
        else:
            tsdf, target, max_l, mid_p = s
            mid_p = mid_p.unsqueeze(1)
        max_l = max_l.unsqueeze(1)
        if USE_SENSOR:
            tsdf, angles = tsdf
            input_sensor_var = torch.autograd.Variable(angles.cuda())
        tsdf = tsdf.unsqueeze(1).cuda()
        batch_size = tsdf.size(0)

        # normalize target to [0,1]
        n_target = ((target.view(batch_size, -1, 3) - mid_p).view(batch_size,-1)) / max_l + 0.5
        n_target[n_target < 0] = 0
        n_target[n_target >= 1] = 1

        input_var = torch.autograd.Variable(tsdf)
        target_var = torch.autograd.Variable(n_target.cuda())


        # compute output
        if USE_SENSOR:
            output = model((input_var, input_sensor_var))
        else:
            output = model(input_var)

        # loss calcu
        loss = criterion(output, target_var)
        losses.update(loss.data[0], batch_size)

        # measure accuracy
        output = ((output.data.cpu() - 0.5) * max_l).view(batch_size,-1,3) + mid_p
        output = output.view(batch_size,-1)
        err_t = error(output, target)
        errors.update(err_t, batch_size)

        # measure elapsed time
        batch_time.update(time() - end)
        end = time()
        if i % PRINT_FREQ == 0:
            # print mean_errs.avg.cpu().numpy()
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'acc_in_t {err_t.val:.3f} ({err_t.avg:.3f})'.format(
                   i, len(test_loader), batch_time=batch_time,
                   loss=losses, err_t=errors))
        result = np.append(result, output.numpy(), axis=0)
    return result, errors.avg


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 0.3 every 5 epochs"""
    lr = args.learning_rate * (0.3 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def visualize_result(model_path, data_path):
    from visualization import plot_tsdf, plot_pointcloud
    net = HandNet()
    net.cuda()
    dataset = MSRADataSet(data_path)
    checkpoint = torch.load(model_path)
    # args.start_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    print "using model with acc [{:.2f}%]".format(best_acc)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    for i in range(100,110):
        tsdf, label, max_l, mid_p = dataset[i]
        tsdf = torch.from_numpy(tsdf)
        tsdf.unsqueeze_(0) # for channel
        tsdf.unsqueeze_(0) # for batch
        input_var = torch.autograd.Variable(tsdf).cuda()
        # compute output
        output = net(input_var)
        output = output.data
        label = torch.from_numpy(label).unsqueeze_(0).cuda()
        max_l = torch.from_numpy(np.array(max_l))
        print mean_error(output, label, max_l)
        # print accuracy_error_thresh_portion_batch(output, label, max_l)
        # print good_frame(output, label, max_l)
        # pc, label = dataset.get_point_cloud(i)
        # plot_pointcloud(pc, label)
        # plot_pointcloud(pc, (output.reshape(-1,3)-0.5)*max_l+mid_p)


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
    x[:, :, 2] = -x[:, :, 2]
    x[:, :, 0] = x[:, :, 0] * fx / x[:, :, 2] + ux
    x[:, :, 1] = -x[:, :, 1] * fy / x[:, :, 2] + uy
    return x


def test_only(model_path, error = accuracy_error_thresh_portion_batch, test_id = 0, save2file = False, world_coor=True):
    start_time = time()
    # warning_init()
    net = HandNet()
    net.cuda()
    dataset = MSRADataSet(DATA_DIR)
    criterion = nn.MSELoss().cuda()
    checkpoint = torch.load(model_path)
    best_acc = checkpoint['best_acc']
    print "using model with acc [{:.2f}%]".format(best_acc)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()

    if test_id != -1:
        train_idx, test_idx = dataset.get_train_test_indices(test_id)
        test_sampler = SubsetRandomSampler(test_idx)
        test_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=BATCH_SIZE, sampler=test_sampler,
                                               num_workers=0)
    else:
        test_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=BATCH_SIZE,
                                                  num_workers=0)
    r, acc = test(test_loader, net, criterion, error=error)
    print "final accuracy {:3f}".format(acc)
    print "total time: ", datetime.timedelta(seconds=(time() - start_time))
    if type(save2file) is str:
        if world_coor:
            np.savetxt(save2file, r,fmt = '%.3f')
        else:
            r = camera2pixel(r.reshape(r.shape[0],-1,3),*get_param('msra')).reshape(r.shape[0],-1)
            np.savetxt(save2file, r,fmt = '%.3f')
        print "result saved to ",save2file


def preprocessing():
    import tables
    warning_init()
    init_parser()
    start_time = time()
    dataset = MSRADataSet(args.data)
    # train_idx, valid_idx = dataset.get_train_test_indices()

    # train_sampler = SubsetRandomSampler(train_idx)
    #
    # train_loader = torch.utils.data.DataLoader(dataset,
    #                                            batch_size=args.batch_size,
    #                                            num_workers=0)

    f = tables.open_file("tsdf.h5", mode="w")
    atom = tables.Float32Atom()
    array_c = f.create_earray(f.root, 'data', atom, (0, dataset[0][0].size+dataset[0][1].size+dataset[0][2].size))
    for i in range(len(dataset)):
            s = dataset[i]
            tsdf, target, max_l, mid_p = s
            tsdf = np.append(tsdf.reshape(-1), np.append(target,max_l)).reshape(1,-1)
            array_c.append(tsdf)
            if i % 100 == 0:
                print ("saving [{}/{}]  time:{}".format(
                    i,len(dataset),datetime.timedelta(seconds=(time() - start_time))))
    f.close()
    print('Finished Preprocessing')


if __name__ == "__main__":

    # m = torch.FloatTensor([[[1, 2, 3], [4, 5, 6]],
    #                        [[1, 2, 3], [4, 5, 6]],
    #                       [[1, 2, 3], [4, 5, 6]],
    #                       [[7, 8, 9], [1,2,3]]])
    # c = torch.FloatTensor([[0, 0, 0],[1,1,1],[2,2,2],[0,0,0]])
    # c = c.unsqueeze(1)
    # m = m.numpy()
    # m = world2pixel(m.reshape(m.shape[0], -1, 3), *get_param('msra'))
    # print m
    # print m.size(),c.size()
    # print m-c
    # a = torch.FloatTensor([
    #     [1,1,1],
    #     [2,2,2]
    # ]).numpy()
    #
    # b = np.empty(shape=(0,3))
    # print b.shape,a.shape
    # b = np.append(b,a, axis=0)
    # b = np.append(b,a, axis=0)
    # print b
    # output = ((output - 1) * max_l + mid_p).reshape(batch,-1)
    # print output


    # preprocessing()
    # main(HandNet)
    # test_only('model_best90.pth.tar', mean_error, 0)
    path = '/home/hfy/code/awesome-hand-pose-estimation/evaluation/results/msra/result.txt'
    test_only('model_best_full.pth.tar',test_id=-1, error=mean_error, save2file=path, world_coor=False)

    # visualize_result('model_best90.pth.tar',DATA_DIR)
