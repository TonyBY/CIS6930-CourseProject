from bs4 import BeautifulSoup
import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_normalized_box(obj,width_scale,height_scale):
    xmin = int(int(obj.find('xmin').text) * width_scale)
    ymin = int(int(obj.find('ymin').text) * height_scale)
    xmax = int(int(obj.find('xmax').text) * width_scale)
    ymax = int(int(obj.find('ymax').text) * height_scale)
    return [xmin, ymin, xmax, ymax]

def generate_label(obj):
    if obj.find('name').text == "with_mask":
        return 1
    elif obj.find('name').text == "mask_weared_incorrect":
        return 2
    return 3


def generate_target(image_id, file,width_scale,height_scale):
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, 'xml')
        objects = soup.find_all('object')

        num_objs = len(objects)

        # Bounding boxes for objects
        boxes = []
        labels = []
        for i in objects:
            boxes.append(generate_normalized_box(i,width_scale,height_scale))
            labels.append(generate_label(i))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([image_id])
        # Annotation is in dictionary format
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id

        return target


