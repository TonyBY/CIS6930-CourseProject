from PIL import Image
import os
import numpy as np
import cv2
import torch

from CNN_utils import generate_target

class MaskDataset(object):
    def __init__(self, transforms, imgs_path, labels_path, mode):
        self.transforms = transforms
        # load all image files, sorting them to ensure that they are aligned
        self.imgs = list(sorted(os.listdir(imgs_path)))
       # print(self.imgs)
        self.imgs_path = imgs_path
        self.labels_path = labels_path
        self.mode = mode

    def __getitem__(self, idx):
        if self.mode == "eval":
            idx = idx + 5032
        #print(idx)
        # load images ad masks
        file_image = 'maksssksksss' + str(idx) + '.png'
        file_label = 'maksssksksss' + str(idx) + '.xml'
        img_path = os.path.join(self.imgs_path, file_image)
        label_path = os.path.join(self.labels_path, file_label)
        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)

        #resize image
        arr_img = np.asarray(img).transpose((1, 2, 0))
        # print(idx, 'before Dimensions : ',arr_img.shape)
        width = 300
        height = 300
        dim = (width, height)
        arr_img = cv2.resize(arr_img, dim, interpolation = cv2.INTER_AREA)
        # print('Resized Dimensions : ',arr_img.shape)
        arr_img = torch.from_numpy(arr_img.transpose((2, 0, 1))).float()
        
        # Generate normalized labels
        width_scale = width / img.shape[2]
        height_scale = height / img.shape[1] 
        # print("scale:",width_scale,height_scale)
        target = generate_target(idx, label_path, width_scale,height_scale)
        # print(arr_img.size())
        return arr_img, target

    def __len__(self):
        return len(self.imgs)

# transform annotations into tensor(4*4*144*144)
def create_window_label(annotations):
    result = torch.zeros(len(annotations),4,144,144)
    scale = 144/300
    for i in range(len(annotations)): #each image
        #initial class'3' = 1, background
        result[i,3] = torch.ones(144,144)
        # print(annotations[i]['labels'])
        for j in range(len(annotations[i]['labels'])):
            label = annotations[i]['labels'][j]
            [xmin, ymin, xmax, ymax] = annotations[i]['boxes'][j]
            xmin, ymin, xmax, ymax = int(xmin*scale), int(ymin*scale), min(int(xmax*scale),143), min(int(ymax*scale),143)
            bb = torch.ones(ymax+1-ymin, xmax+1-xmin)
            bg = torch.zeros(ymax+1-ymin, xmax+1-xmin)
            # print(bb.size(),result[i, label, ymin:ymax+1, xmin:xmax+1].size())
            result[i, label, ymin:ymax+1, xmin:xmax+1] = bb
            result[i,3, ymin:ymax+1, xmin:xmax+1] = bg
    return result
