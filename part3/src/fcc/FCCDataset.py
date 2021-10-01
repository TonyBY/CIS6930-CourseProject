from PIL import Image
import os
import numpy as np
import cv2
import torch
import glob
# import the necessary packages
from FCC_utils import *

class MaskDataset(object):
    def __init__(self, transforms, imgs_path, labels_path, mode):
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = []
        for filename in glob.glob(imgs_path + '/*.png'):
            self.imgs.append(filename)
        # self.imgs = list(sorted(os.listdir(imgs_path)))
        self.imgs_path = imgs_path
        self.labels_path = labels_path
        self.mode = mode

    def __getitem__(self, idx):
        if self.mode == "eval":
            idx = idx + 768

        # load images ad masks
        file_image = 'maksssksksss' + str(idx) + '.png'
        file_label = 'maksssksksss' + str(idx) + '.xml'
        img_path = os.path.join(self.imgs_path, file_image)
        label_path = os.path.join(self.labels_path, file_label)
        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        width, height = 300, 300
        arr_img = resize(img, width, height)
        # Generate normalized labels
        width_scale = width / img.shape[2]
        height_scale = height / img.shape[1] 
        # print("scale:",width_scale,height_scale)
        target = generate_target(idx, label_path, width_scale,height_scale)
        # print(arr_img.size())
        return arr_img, target, img_path

    def __len__(self):
        return len(self.imgs)

# transform annotations into tensor(4*4*144*144)
def create_image_label(annotations, label_num, height, width, scale):
    result = torch.zeros(len(annotations),label_num, height, width)
    for i in range(len(annotations)): #each image
        #initial class3 = 1, background
        result[i,3] = torch.ones(height, width)
        # print(annotations[i]['labels'])
        for j in range(len(annotations[i]['labels'])):
            label = annotations[i]['labels'][j]
            [xmin, ymin, xmax, ymax] = annotations[i]['boxes'][j]
            xmin, ymin, xmax, ymax = int(xmin*scale), int(ymin*scale), min(int(xmax*scale),width-1), min(int(ymax*scale),height-1)
            bb = torch.ones(ymax+1-ymin, xmax+1-xmin)
            bg = torch.zeros(ymax+1-ymin, xmax+1-xmin)
            # print(bb.size(),result[i, label, ymin:ymax+1, xmin:xmax+1].size())
            result[i, label, ymin:ymax+1, xmin:xmax+1] = bb
            result[i,3, ymin:ymax+1, xmin:xmax+1] = bg
    return result
