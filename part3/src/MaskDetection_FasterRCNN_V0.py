import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import torchvision
from torchvision import transforms, datasets, models
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import matplotlib.patches as patches

import os
import _pickle as pickle


def generate_box(obj):
    xmin = int(obj.find('xmin').text)
    ymin = int(obj.find('ymin').text)
    xmax = int(obj.find('xmax').text)
    ymax = int(obj.find('ymax').text)

    return [xmin, ymin, xmax, ymax]


def generate_label(obj):
    if obj.find('name').text == "with_mask":
        return 1
    elif obj.find('name').text == "mask_weared_incorrect":
        return 2
    return 0


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


class MaskDataset(object):
    def __init__(self, transforms):
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(imgs_path)))

    #         self.labels = list(sorted(os.listdir("/kaggle/input/face-mask-detection/annotations/")))

    def __getitem__(self, idx):
        # load images ad masks
        file_image = 'maksssksksss' + str(idx) + '.png'
        file_label = 'maksssksksss' + str(idx) + '.xml'
        img_path = os.path.join(imgs_path, file_image)
        label_path = os.path.join(labels_path, file_label)
        img = Image.open(img_path).convert("RGB")
        # Generate Label
        target = generate_target(idx, label_path)
        #         print ("target['boxes']: ", target["boxes"])
        #         print ("target['labels']: ", target["labels"])
        #         print ("target['image_id']", target["image_id"])

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


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


def plot_image(img_tensor, annotation):
    fig, ax = plt.subplots(1)
    img = img_tensor.cpu().data

    # Display the image
    ax.imshow(img.permute(1, 2, 0))

    for box in annotation["boxes"]:
        xmin, ymin, xmax, ymax = box

        # Create a Rectangle patch
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='r',
                                 facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    # plt.show()
    plt.imsave(arr=img.permute(1, 2, 0), fname="../data/data2/FasterRCNN/outputs")


if __name__ == "__main__":
    imgs_path = "../data/data2/images/"
    labels_path: str = "../data/data2/annotations/"
    # imgs = list(sorted(os.listdir(imgs_path)))
    # labels = list(sorted(os.listdir(labels_path)))

    data_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = MaskDataset(data_transform)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, collate_fn=collate_fn)

    model = get_model_instance_segmentation(3)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Train
    num_epochs = 1
    model.to(device)

    # parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    len_dataloader = len(data_loader)

    # Make folders and set parameters
    os.makedirs('../data/data2/FasterRCNN/outputs', exist_ok=True)
    os.makedirs('../data/data2/FasterRCNN/checkpoints', exist_ok=True)

    best_losses = 1e10

    for epoch in range(num_epochs):
        model.train()
        i = 0
        epoch_loss = 0
        for imgs, annotations in data_loader:
            i += 1
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            loss_dict = model([imgs[0]], [annotations[0]])
            losses = sum(loss for loss in loss_dict.values())

            if losses < best_losses:
                best_losses = losses
                torch.save(model.state_dict(),
                           '../data/data2/checkpoints/model-epoch-{}-losses-{:.3f}.pth'.format(epoch + 1, losses))
                print('done saving')

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            print(f'Iteration: {i}/{len_dataloader}, Loss: {losses}')
            epoch_loss += losses
        print(epoch_loss)
