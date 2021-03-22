import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F


def YOLO_loss(pred_confidence, pred_box, ann_confidence, ann_box):
    # input:
    # pred_confidence -- the predicted class labels from YOLO, [batch_size, 5,5, num_of_classes]
    # pred_box        -- the predicted bounding boxes from YOLO, [batch_size, 5,5, 4]
    # ann_confidence  -- the ground truth class labels, [batch_size, 5,5, num_of_classes]
    # ann_box         -- the ground truth bounding boxes, [batch_size, 5,5, 4]
    #
    # output:
    # loss -- a single number for the value of the loss function, [1]
    
    # TODO: write a loss function for YOLO
    #
    # For confidence (class labels), use cross entropy (F.cross_entropy)
    # You can try F.binary_cross_entropy and see which loss is better
    # For box (bounding boxes), use smooth L1 (F.smooth_l1_loss).
    # Note that you need to consider cells carrying objects and empty cells separately.
    # I suggest you to reshape confidence to [batch_size*5*5, num_of_classes]
    # and reshape box to [batch_size*5*5, 4].
    # Then you need to figure out how you can get the indices of all cells carrying objects,
    # and use confidence[indices], box[indices] to select those cells.
    ann_confidence = ann_confidence.reshape((-1, 4))
    pred_confidence = pred_confidence.reshape((-1, 4))
    ann_box = ann_box.reshape((-1, 4))
    pred_box = pred_box.reshape((-1, 4))

    # select cells
    obj = torch.where(ann_confidence[:, -1] == 1)
    no_obj = torch.where(ann_confidence[:, -1] == 0)
    #print('pred obj', pred_confidence[obj])
    #print('pred no_obj', pred_confidence[no_obj])
    loss_conf = F.binary_cross_entropy(pred_confidence[obj], ann_confidence[obj]) + 3 * F.binary_cross_entropy(pred_confidence[no_obj], ann_confidence[no_obj])
    loss_box = F.smooth_l1_loss(pred_box[obj], ann_box[obj])
    loss = loss_conf + loss_box

    return loss

'''
YOLO network
Please refer to the hand-out for better visualization

N is batch_size
bn is batch normalization layer
relu is ReLU activation layer
conv(cin,cout,ksize,stride) is convolution layer
  cin - the number of input channels
  cout - the number of output channels
  ksize - kernel size
  stride - stride
  padding - you need to figure this out by yourself

input -> [N,3,320,320]

conv(  3, 64, 3, 2),bn,relu -> [N,64,160,160]

conv( 64, 64, 3, 1),bn,relu -> [N,64,160,160]
conv( 64, 64, 3, 1),bn,relu -> [N,64,160,160]
conv( 64,128, 3, 2),bn,relu -> [N,128,80,80]

conv(128,128, 3, 1),bn,relu -> [N,128,80,80]
conv(128,128, 3, 1),bn,relu -> [N,128,80,80]
conv(128,256, 3, 2),bn,relu -> [N,256,40,40]

conv(256,256, 3, 1),bn,relu -> [N,256,40,40]
conv(256,256, 3, 1),bn,relu -> [N,256,40,40]
conv(256,512, 3, 2),bn,relu -> [N,512,20,20]

conv(512,512, 3, 1),bn,relu -> [N,512,20,20]
conv(512,512, 3, 1),bn,relu -> [N,512,20,20]
conv(512,256, 3, 2),bn,relu -> [N,256,10,10]

conv(256,256, 1, 1),bn,relu -> [N,256,10,10]
conv(256,256, 3, 2),bn,relu -> [N,256,5,5] (the last hidden layer)

output layer 1 - confidence
(from the last hidden layer)
conv(256,num_of_classes, 3, 1),softmax? -> [N,num_of_classes,5,5]
permute (or transpose) -> [N,5,5,num_of_classes]

output layer 2 - bounding boxes
(from the last hidden layer)
conv(256, 4, 3, 1) -> [N,4,5,5]
permute (or transpose) -> [N,5,5,4]
'''


class YOLO(nn.Module):

    def __init__(self, class_num):
        super(YOLO, self).__init__()
        
        self.class_num = class_num  # num_of_classes, in this assignment, 4: cat, dog, person, background
        
        #TODO: define layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.left = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=4, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.right = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Softmax(dim=2)
        )
        
    def forward(self, x):
        # input:
        # x -- images, [batch_size, 3, 320, 320]
        #print('x shape:', x.shape)
        x = x/255.0  # normalize image. If you already normalized your input image in the dataloader, remove this line.
        # TODO: define forward
        # should you apply softmax to confidence? (search the pytorch tutorial for F.cross_entropy.) If yes, which dimension should you apply softmax?
        x = self.conv1(x)
        #print(x.shape)
        x = self.conv2(x)
        #print(x.shape)
        x = self.conv3(x)
        #print(x.shape)
        x = self.conv4(x)
        #print(x.shape)
        x = self.conv5(x)
        #print(x.shape)

        x1 = self.left(x)
        #print('x1:', x1.shape)
        x2 = self.right(x)
        #print('x2:', x2.shape)

        # sanity check: print the size/shape of the confidence and bboxes, make sure they are as follows:
        # confidence - [batch_size,5,5,num_of_classes]
        # bboxes - [batch_size,5,5,4]
        confidence = x1.permute((0, 2, 3, 1))
        # print('x1:', confidence.shape)
        confidence = torch.sigmoid(confidence)
        bboxes = x2.permute((0, 2, 3, 1))
        # print('x2:', bboxes.shape)
        bboxes = torch.sigmoid(bboxes)
        
        return confidence, bboxes










