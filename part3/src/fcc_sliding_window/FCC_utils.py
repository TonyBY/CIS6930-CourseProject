from bs4 import BeautifulSoup
import torchvision
import torch
import imutils
import cv2
import numpy as np

#output: tensor
def resize(img,width,height):
    arr_img = np.asarray(img.cpu()).transpose((1, 2, 0))
    # print(idx, 'before Dimensions : ',arr_img.shape)
    dim = (width, height)
    arr_img = cv2.resize(arr_img, dim, interpolation = cv2.INTER_AREA)
    # print('Resized Dimensions : ',arr_img.shape)
    arr_img = torch.from_numpy(arr_img.transpose((2, 0, 1))).float()
    return arr_img
        
def pyramid(images, scale=1.5, minSize=(30, 30)):
	# yield the original image
    yield images
    while True:
        #print(images.size()[2])
        w = int(images.size()[2] / scale)
        new = []
        for i in range(images.size()[0]):
            new.append(resize(images[i], w, w))
        images = torch.stack(new)
        if images.size()[2] < minSize[1] or images.size()[3] < minSize[0]:
            break
        yield images

        
def sliding_window(images, stepSize, windowSize):
	# slide a window across the image
    for y in range(0, images.size()[2], stepSize):
        for x in range(0, images.size()[3], stepSize):
        # yield the current window
            yield (x, y, images[:,:,y:y + windowSize[1], x:x + windowSize[0]])

def get_window_label(labels):
    result = []
    for i in range(labels.size()[0]):
        win_label = []
        for j in range(labels.size()[1]):
            if torch.equal(labels[i,j],torch.zeros_like(labels[i,j])):
                win_label.append(0.0)
            else:
                win_label.append(1.0)
        result.append(torch.tensor(win_label))
    result = torch.stack(result)
    return result


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
    return 0


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


