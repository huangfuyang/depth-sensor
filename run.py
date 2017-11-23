import argparse
from network import HandNet
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
import warnings


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
    np.seterr(all='warn')


def main():
    warning_init()
    start_time = time()
    init_parser()
    net = HandNet()
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

    dataset = MSRADataSet(args.data)

    train_idx, valid_idx = dataset.get_train_test_indices()

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=args.batch_size, sampler=train_sampler,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=args.batch_size, sampler=test_sampler,
                                               num_workers=0)

    for epoch in range(args.start_epoch, args.epochs):  # loop over the dataset multiple times
        epoch_start_time = time()
        adjust_learning_rate(optimizer, epoch)

        acc = train(train_loader,net,criterion,optimizer,epoch)
        # remember best acc and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        print('Epoch: [{0}/{1}]  Time [{2}/{3}]'.format(
            epoch, args.epochs, datetime.timedelta(seconds=(time() - epoch_start_time)),
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
    acc = test(test_loader,net,criterion)
    print "final accuracy {:3f}".format(acc)
    print "total time: ",datetime.timedelta(seconds=(time()-start_time))


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc_in_t = AverageMeter()

    # switch to train mode
    model.train()

    end = time()
    for i, (tsdf, target,(mid_p,max_l)) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time() - end)

        tsdf = tsdf.unsqueeze_(1).cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(tsdf)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        err_t = accuracy_error_thresh_portion_batch(output.data, target,max_l)
        losses.update(loss.data[0], tsdf.size(0))
        acc_in_t.update(err_t)

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
                   data_time=data_time, loss=losses, err_t=acc_in_t))
    return acc_in_t.avg


def test(test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc_in_t = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time()

    for i, (tsdf, target,(mid_p,max_l)) in enumerate(test_loader):
        target = target.cuda()
        tsdf = tsdf.unsqueeze_(1).cuda()
        input_var = torch.autograd.Variable(tsdf)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        err_t = accuracy_error_thresh_portion_batch(output.data, target, max_l)
        losses.update(loss.data[0], tsdf.size(0))
        acc_in_t.update(err_t)

        # measure elapsed time
        batch_time.update(time() - end)
        end = time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'acc_in_t {err_t.val:.3f} ({err_t.avg:.3f})'.format(
                   i, len(test_loader), batch_time=batch_time,
                   loss=losses, err_t=acc_in_t))
    return acc_in_t.avg


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 0.3 every 5 epochs"""
    lr = args.learning_rate * (0.3 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy_portion(output, target, t=ERROR_THRESH):
    diff = np.abs(output - target)
    out = np.zeros(target.size)
    out[diff < t] = 1
    good = np.sum(out) / out.size
    return good * 100

def accuracy_error_thresh_portion_batch(output, target, max_l, t=ERROR_THRESH):
    batch_size = target.size(0)
    sample_size = target.size(1)
    max_l = max_l.repeat(sample_size).view(batch_size,sample_size).cuda()
    diff = torch.abs(output-target)
    diff = torch.mul(diff,max_l)
    out = torch.zeros(target.size()).cuda()
    out[diff<t] = 1
    good = torch.sum(out)/(out.size(1)*batch_size)
    return good*100


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def visualize_result(model_path, data_path):
    from visualization import plot_tsdf, plot_pointcloud
    net = HandNet()
    # net.cuda()
    dataset = MSRADataSet(data_path)
    checkpoint = torch.load(model_path)
    # args.start_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    print "using model with acc [{:.2f}%]".format(best_acc)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    for i in range(100,200):
        tsdf, label,(mid_p,max_l) = dataset[i]
        tsdf = torch.from_numpy(tsdf)
        tsdf.unsqueeze_(0) # for channel
        tsdf.unsqueeze_(0) # for batch
        input_var = torch.autograd.Variable(tsdf)
        pc, label = dataset.get_point_cloud(i)
        # compute output
        output = net(input_var)
        output = output.data.squeeze(0).numpy().reshape(-1,3)
        output = ((output-0.5)*max_l+mid_p).reshape(-1)
        print accuracy_portion(output, label)
        # plot_pointcloud(pc, label)
        # plot_pointcloud(pc, output)


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


if __name__ == "__main__":
    # output = ((output - 1) * max_l + mid_p).reshape(batch,-1)
    # print output
    # main()
    visualize_result('model_best.pth.tar',DATA_DIR)
#
# ########################################################################
# # 5. Test the network on the test data
# # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# #
# # We have trained the network for 2 passes over the training dataset.
# # But we need to check if the network has learnt anything at all.
# #
# # We will check this by predicting the class label that the neural network
# # outputs, and checking it against the ground-truth. If the prediction is
# # correct, we add the sample to the list of correct predictions.
# #
# # Okay, first step. Let us display an image from the test set to get familiar.
#
# dataiter = iter(testloader)
# images, labels = dataiter.next()
#
# # print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
#
# ########################################################################
# # Okay, now let us see what the neural network thinks these examples above are:
#
# outputs = net(Variable(images.cuda()))
#
# ########################################################################
# # The outputs are energies for the 10 classes.
# # Higher the energy for a class, the more the network
# # thinks that the image is of the particular class.
# # So, let's get the index of the highest energy:
# _, predicted = torch.max(outputs.data, 1)
#
# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                               for j in range(4)))
#
# ########################################################################
# # The results seem pretty good.
# #
# # Let us look at how the network performs on the whole dataset.
#
# correct = 0
# total = 0
# for data in testloader:
#     images, labels = data
#     outputs = net(Variable(images.cuda()))
#     _, predicted = torch.max(outputs.data, 1)
#     total += labels.size(0)
#     correct += (predicted == labels.cuda()).sum()
#
# print('Accuracy of the network on the 10000 test images: %d %%' % (
#     100 * correct / total))
#
# ########################################################################
# # That looks waaay better than chance, which is 10% accuracy (randomly picking
# # a class out of 10 classes).
# # Seems like the network learnt something.
# #
# # Hmmm, what are the classes that performed well, and the classes that did
# # not perform well:
#
# class_correct = list(0. for i in range(10))
# class_total = list(0. for i in range(10))
# for data in testloader:
#     images, labels = data
#     outputs = net(Variable(images.cuda()))
#     _, predicted = torch.max(outputs.data, 1)
#     c = (predicted == labels.cuda()).squeeze()
#     for i in range(4):
#         label = labels[i]
#         class_correct[label] += c[i]
#         class_total[label] += 1
#
#
# for i in range(10):
#     print('Accuracy of %5s : %2d %%' % (
#         classes[i], 100 * class_correct[i] / class_total[i]))