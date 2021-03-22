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
import numpy as np
import os
import cv2


def match(ann_box, ann_confidence, cat_id, x_min, y_min, x_max, y_max):
    # input:
    # ann_box                 -- [5,5,4], ground truth bounding boxes to be updated
    # ann_confidence          -- [5,5,number_of_classes], ground truth class labels to be updated
    # cat_id                  -- class id, 0-cat, 1-dog, 2-person
    # x_min,y_min,x_max,y_max -- bounding box
    
    size = 5  # the size of the output grid
    
    # update ann_box and ann_confidence
    w = (x_max-x_min)
    h = (y_max-y_min)
    center_x = (x_min + w/2)
    center_y = (y_min + h/2)
    x, y, idx_i, idx_j = 0, 0, 0, 0
    # find the corresponding cell
    for i in np.arange(0, 1, 0.2):  # 5*5
        if i < center_x < (i + 0.2):
            x = idx_i
        idx_i += 1
    for j in np.arange(0, 1, 0.2):
        if j < center_y < (j + 0.2):
            y = idx_j
        idx_j += 1

    ann_box[y, x, :] = [center_x, center_y, w, h]
    ann_confidence[y, x, 3] = 0  # not background
    ann_confidence[y, x, cat_id] = 1

    return ann_box, ann_confidence
    

class COCO(torch.utils.data.Dataset):
    def __init__(self, imgdir, anndir, class_num, train=True, image_size=320):
        self.train = train
        self.imgdir = imgdir
        self.anndir = anndir
        self.class_num = class_num
        
        self.img_names = os.listdir(self.imgdir)
        self.image_size = image_size
        
        # notice:
        # you can split the dataset into 90% training and 10% validation here, by slicing self.img_names with respect to self.train

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        size = 5  # the size of the output grid
        ann_box = np.zeros([5, 5, 4], np.float32)  # 5*5 bounding boxes
        ann_confidence = np.zeros([5, 5, self.class_num], np.float32)  # 5*5 one-hot vectors
        # one-hot vectors with four classes
        # [1,0,0,0] -> cat
        # [0,1,0,0] -> dog
        # [0,0,1,0] -> person
        # [0,0,0,1] -> background
        
        ann_confidence[:, :, -1] = 1  # the default class for all cells is set to "background"
    
        img_name = self.imgdir+self.img_names[index]
        ann_name = self.anndir+self.img_names[index][:-3]+"txt"
        
        # TODO:
        # 1. prepare the image [3,320,320], by reading image "img_name" first.

        # 2. prepare ann_box and ann_confidence, by reading txt file "ann_name" first.
        if self.train:  # train
            image = cv2.imread(img_name)
            height, width, _ = image.shape
            # resize
            image = cv2.resize(image, (self.image_size, self.image_size))  # (320, 320, 3)
            image = image.transpose(2, 0, 1)  # (3, 320, 320)
            ann_info = open(ann_name)
            info = ann_info.readlines()
            for object_info in info:
                obj = object_info.split()

                class_id, x_min, y_min, w, h = obj
                class_id = int(class_id)
                x_min = (float(x_min))
                y_min = (float(y_min))
                w = (float(w))
                h = (float(h))

                x_max = ((x_min + w) / width)
                y_max = ((y_min + h) / height)
                x_min = (x_min / width)
                y_min = (y_min / height)

                ann_box, ann_confidence = match(ann_box, ann_confidence, class_id, x_min, y_min, x_max, y_max)
            # 3. use the above function "match" to update ann_box and ann_confidence,
            # for each bounding box in "ann_name".
            # 4. Data augmentation. You need to implement random cropping first.
            # You can try adding other augmentations to get better results.
            # to use function "match":
            # match(ann_box,ann_confidence,class_id,x_min,y_min,x_max,y_max)
            # where [x_min,y_min,x_max,y_max] is from the ground truth bounding box,
            # normalized with respect to the width or height of the image.
            # you may wonder maybe it is better to input [x_center, y_center, box_width, box_height].
            # maybe it is better.
            # BUT please do not change the inputs.
            # Because you will need to input [x_min,y_min,x_max,y_max] for SSD.
            # It is better to keep function inputs consistent.
            # note: please make sure x_min,y_min,x_max,y_max
            # are normalized with respect to the width or height of the image.
            # For example, point (x=100, y=200) in a image with (width=1000, height=500)
            # will be normalized to (x/width=0.1,y/height=0.4)

            #print('ann box shape', ann_box.shape)
            #print(ann_box)
            #print('ann confidence shape', ann_confidence.shape)
            #print(ann_confidence)
        else:  # test, do not have ann dir
            image = cv2.imread(img_name)
            height, width, _ = image.shape
            # resize
            image = cv2.resize(image, (self.image_size, self.image_size))  # (320, 320, 3)
            image = image.transpose(2, 0, 1)  # (3, 320, 320)

        return image, ann_box, ann_confidence


'''
if __name__ == '__main__':
    dataset = COCO("data/train/images/", "data/train/annotations/", 4, train=True, image_size=320)

    a = 0

'''

