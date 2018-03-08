import argparse
import datetime
import shutil
import warnings

import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

from dataset.MSRA import *
from models.hg import *
from helper import *
from metrics import *
from network import *

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


def main(Model, full = False):
    warning_init()
    start_time = time()
    init_parser()
    net = Model(nSTACK,2,256).cuda()
    criterion = nn.MSELoss().cuda()
    best_err = 9999
    optimizer = optim.RMSprop(net.parameters(),
    # optimizer=optim.SGD(net.parameters(),
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
            print("=> loaded checkpoint '{}' (epoch {}) err {}"
                  .format(args.resume, checkpoint['epoch'], best_acc))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    dataset = MSRADataSet(args.data)
    train_idx, valid_idx = dataset.get_train_test_indices()
    # train_idx = range(2)
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(valid_idx)

    if full:
        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=args.batch_size,
                                                   num_workers=WORKER)
        test_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=args.batch_size,
                                               num_workers=WORKER)

    else:
        train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=args.batch_size, sampler=train_sampler,
                                               num_workers=WORKER)

        test_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=args.batch_size, sampler=test_sampler,
                                               num_workers=WORKER)

    for epoch in range(args.start_epoch, args.epochs):  # loop over the dataset multiple times
        epoch_start_time = time()
        adjust_learning_rate(optimizer, epoch+1)

        err = train(train_loader,net,criterion,optimizer,epoch+1)
        # remember best acc and save checkpoint
        is_best = err < best_err
        best_err = min(err, best_err)

        print('Epoch: [{0}/{1}]  Time [{2}/{3}]'.format(
            epoch+1, args.epochs, datetime.timedelta(seconds=(time() - epoch_start_time)),
                                datetime.timedelta(seconds=(time() - start_time))))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': '3dDNN',
            'state_dict': net.state_dict(),
            'best_acc': best_err,
            'optimizer': optimizer.state_dict(),
        }, is_best)
    print('Finished Training')
    # print 'evaluating test dataset'
    # result, acc = test(test_loader,net,criterion)
    # print "final accuracy {:3f}".format(acc)
    # print "total time: ",datetime.timedelta(seconds=(time()-start_time))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    errors = AverageMeter()
    # switch to train mode
    model.train()

    end = time()
    for i, s in enumerate(train_loader):
        data, heatmaps, label, pix_length = s
        # measure data loading time
        data_time.update(time() - end)
        batch_size = data[0].size(0)
        # normalize target to [0,1]
        # n_target = (target.view(batch_size, -1, 3) - mid_p).view(batch_size,-1) / max_l + 0.5
        # n_target[n_target < 0] = 0
        # n_target[n_target >= 1] = 1

        input_var1 = torch.autograd.Variable(data[0].unsqueeze(1).cuda())
        input_var2 = torch.autograd.Variable(data[1].unsqueeze(1).cuda())
        input_var3 = torch.autograd.Variable(data[2].unsqueeze(1).cuda())
        target_var = torch.autograd.Variable(heatmaps.cuda())

        # compute output
        output = model(input_var1)

        # record loss
        loss = criterion(output[0], target_var)
        for k in range(1, nSTACK):
            loss += criterion(output[k], target_var)
        losses.update(loss.data[0], batch_size)

        # measure accuracy
        k = 5
        output = output[-1].data.view(batch_size, JOINT_LEN, HM_SIZE * HM_SIZE)
        d, indices = torch.topk(output, k, dim=2)
        indices = indices.unsqueeze(-1)
        indices_2d = torch.FloatTensor(batch_size, JOINT_LEN, k, 2)
        indices_2d[:, :, :, 0] = torch.remainder(indices, HM_SIZE)
        indices_2d[:, :, :, 1] = torch.div(indices, HM_SIZE)

        err_pixel, ind = mean_error_heatmap_topk(indices_2d, label)
        err_ = err_pixel*torch.mean(pix_length)
        err_pixel = err_pixel.mean()
        err_t =err_.mean()
        err_index = err_[4]
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
                  'acc_in_t {err_t.val:.3f} ({err_t.avg:.3f}) err pix {err:.3f} err in {ei:.3f}'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, err_t=errors, err=err_pixel, ei=err_index))
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
        batch_size = tsdf.size(0)
        tsdf = tsdf[:, 2, :, :, :].unsqueeze(1).cuda()

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
    lr = args.learning_rate * (DECAY_RATIO ** (epoch // DECAY_EPOCH))
    print 'Learning rate :',lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def visual_img(img):
    a = img*255
    Image.fromarray(a).show()

def label2img(label):
    d = np.zeros((HM_SIZE,HM_SIZE),dtype=np.float32)
    label = label.numpy().reshape((JOINT_LEN,2)).astype(int)
    d[label[:,1].reshape(-1),label[:,0].reshape(-1)] = 255
    Image.fromarray(d).show()


def visualize_result(model_path, data_path):
    # from visualization import plot_tsdf, plot_pointcloud
    net = HourglassNet(nSTACK,2,256)
    dataset = MSRADataSet(data_path)
    checkpoint = torch.load(model_path)
    # args.start_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    print "using model with acc [{:.2f}%]".format(best_acc)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    net.cuda()
    criterion = nn.MSELoss().cuda()
    batch_size= 1
    e = AverageMeter()
    for i in range(2001,2002):
        s = dataset[i]
        data, heatmaps, label, pix_length = s
        # measure data loading time
        # normalize target to [0,1]
        # n_target = (target.view(batch_size, -1, 3) - mid_p).view(batch_size,-1) / max_l + 0.5
        # n_target[n_target < 0] = 0
        # n_target[n_target >= 1] = 1

        input_var1 = torch.autograd.Variable(torch.from_numpy(data[0]).unsqueeze(0).unsqueeze(0).cuda())
        input_var2 = torch.autograd.Variable(torch.from_numpy(data[1]).unsqueeze(0).unsqueeze(0).cuda())
        input_var3 = torch.autograd.Variable(torch.from_numpy(data[2]).unsqueeze(0).unsqueeze(0).cuda())
        target_var = torch.autograd.Variable(torch.from_numpy(heatmaps).unsqueeze(0).cuda())

        # compute output
        output = net(input_var1)

        # record loss
        loss = criterion(output[0], target_var)
        for k in range(1, nSTACK):
            loss += criterion(output[k], target_var)

        # measure accuracy
        # print output[-1].data.shape
        # visual_img(output[-1].data[0,0].cpu().numpy())
        # visual_img(output[-1].data[0,3].cpu().numpy())
        # visual_img(heatmaps[0])
        palm =  output[-1].data[0,0].cpu().numpy()
        # for i in range(JOINT_LEN):
        #     visual_img(output[-1].data[0, i].cpu().numpy())


        k = 1
        output = output[-1].data.view(batch_size, JOINT_LEN, HM_SIZE * HM_SIZE)
        d, indices = torch.topk(output,k,dim=2)
        indices = indices.unsqueeze(-1)
        indices_2d = torch.FloatTensor(batch_size, JOINT_LEN, k,2)
        indices_2d[:, :,:, 0] = torch.remainder(indices, HM_SIZE)
        indices_2d[:, :,:, 1] = torch.div(indices, HM_SIZE)

        # indices_2d = indices_2d.view(batch_size, -1)
        label = torch.from_numpy(label).unsqueeze(0)

        err_t, ind = mean_error_heatmap_topk(indices_2d, label)
        err_t*=pix_length
        err_t = err_t.mean()
        # err_t = mean_error_heatmap(indices_2d, label)*pix_length
        heatmaps = np.sum(heatmaps.reshape((JOINT_LEN, -1)), axis=0).reshape((HM_SIZE, HM_SIZE))
        e.update(err_t,1)
        visual_img(data[0])
        # visual_img(heatmaps)
        label2img(label)
        label2img(ind)
        print err_t
        # print loss.data[0]
        # print accuracy_error_thresh_portion_batch(output, label, max_l)
        # print good_frame(output, label, max_l)
        # pc, label = dataset.get_point_cloud(i)
        # plot_pointcloud(pc, label)
        # plot_pointcloud(pc, (output.reshape(-1,3)-0.5)*max_l+mid_p)
    print "ave:",e.avg



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
    # test_only('model_best90.pth.tar', mean_error, 0)
    # path = '/home/hfy/code/awesome-hand-pose-estimation/evaluation/results/msra/result.txt'
    # test_only('model_best.pth.tar',test_id=-1, error=mean_error, save2file=path, world_coor=False)
    # test_only('model_best_full.pth.tar',test_id=-1, error=mean_error, save2file=path, world_coor=False)


    # main(HourglassNet, False)
    # visualize_result('model_best(3).pth.tar',DATA_DIR)
    # visualize_result('model_best(3).pth.tar',DATA_DIR)
    # visualize_result('checkpoint.pth.tar',DATA_DIR)
    # from torchvision.models.resnet import *
    # from models.v2vnet import V2VNet
    # model = HourglassNet(2,2,256)
    # model = V2VNet()
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # for i in model_parameters:
    #     print np.prod(i.size())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # print params
    # print np.prod(np.array([2,2,3]))