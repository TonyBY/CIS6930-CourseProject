from bs4 import BeautifulSoup
import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class_list = ['without_mask', 'with_mask', 'mask_weared_incorrect']


def generate_box(obj):
    xmin = int(obj.find('xmin').text)
    ymin = int(obj.find('ymin').text)
    xmax = int(obj.find('xmax').text)
    ymax = int(obj.find('ymax').text)

    return [xmin, ymin, xmax, ymax]


def generate_label(obj):
    if 'without' in obj.find('name').text or 'no' in obj.find('name').text:
        return 0
    elif 'with_' in obj.find('name').text:
        return 1
    elif 'incorrect' in obj.find('name').text:
        return 2


def generate_target(image_id, file):
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, 'xml')
        objects = soup.find_all('object')

        num_objs = len(objects)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        for i in objects:
            boxes.append(generate_box(i))
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

def collate_fn(batch):
    return tuple(zip(*batch))

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def plot_image(img_tensor, annotation, file_name):
    fig, ax = plt.subplots(1)
    img = img_tensor.cpu().data

    # Display the image
    ax.imshow(img.permute(1, 2, 0))

    for i in range(len(annotation["boxes"])):
        boxes = annotation['boxes']
        labels = annotation['labels']

        box = boxes[i]
        xmin, ymin, xmax, ymax = box
        obj_name = class_list[labels[i]]
        # Create a Rectangle patch
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='r',
                                 facecolor='none', label=obj_name)
        # Add the patch to the Axes
        ax.add_patch(rect)

        # score = annotation['scores']
        try:
            score = annotation['scores']
            confidence = float(score[i])
            plt.text(xmax, ymin, obj_name + "_" + str(confidence), fontsize=12, color='red')
        except KeyError:
            plt.text(xmax, ymin, obj_name, fontsize=12, color='red')
    # plt.show()
    # plt.imsave(arr=img.permute(1, 2, 0), fname="../data/data2/FasterRCNN/outputs/%s.jpg" % file_name)
    plt.savefig("../data/data2/FasterRCNN/outputs/images/%s.jpg" % file_name, dpi=600, bbox_inches='tight', pad_inches=0)
