from network import HandNet
import torch.optim as optim
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from params import *
from datareader import *

T_PRINT = 20
def main():
    net = HandNet()
    # net.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(),
                          lr=LEARNING_RATE,
                          momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY)
    dataset = MSRADataSet('/home/hfy/data/msra15/')

    train_loader = DataLoader(dataset=dataset,
                              batch_size=32,
                              shuffle=False,
                              num_workers=2)

    for epoch in range(EPOCH_COUNT):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (tsdf, labels) in enumerate(train_loader, 0):
            print "process:",i
            # wrap them in Variable
            inputs, labels = Variable(tsdf), Variable(labels)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, loss.data[0]))
            running_loss += loss.data[0]
            if i % T_PRINT == T_PRINT-1:    # print every 2000 mini-batches
                print('[%d, %5d] running loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / T_PRINT))
                running_loss = 0.0

    print('Finished Training')


if __name__ == "__main__":
    main()
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