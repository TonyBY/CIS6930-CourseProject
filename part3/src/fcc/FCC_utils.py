from bs4 import BeautifulSoup
import torchvision
import torch
import imutils
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class_list = ['without_mask', 'with_mask', 'mask_weared_incorrect']
color_list = ['r','g','b']


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
        rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor=color_list[labels[i]],
                                 facecolor='none', label=obj_name)
        # Add the patch to the Axes
        ax.add_patch(rect)

        # score = annotation['scores']
        # try:
        #     score = annotation['scores']
        #     confidence = float(score[i])
        #     plt.text(xmax, ymin, obj_name + "_" + str(confidence), fontsize=12, color='red')
        # except KeyError:
        #     plt.text(xmax, ymin, obj_name, fontsize=12, color='red')
    # plt.show()
    # plt.imsave(arr=img.permute(1, 2, 0), fname="../data/data2/FasterRCNN/outputs/%s.jpg" % file_name)
    print(file_name)
    plt.savefig(file_name, dpi=600, bbox_inches='tight', pad_inches=0)


def save_prediction(prediction, tmp_file):
    boxes = prediction['boxes']
    labels = prediction['labels']
    score = prediction['scores']

    with open(tmp_file, "w") as new_f:
        for i in range(len(boxes)):
            ## split a line by spaces.
            ## "c" stands for center and "n" stands for normalized
            obj_name = class_list[labels[i]]
            box = boxes[i]
            left = float(box[0])
            bottom = float(box[1])
            right = float(box[2])
            top = float(box[3])
            # x_c_n = float(x_min + x_max)/2
            # y_c_n = float(y_min + y_max)/2
            # width_n = float(x_max - x_min)
            # height_n = float(y_max - y_min)/2
            confidence = float(score[i])

            # obj_name = obj_list[int(obj_id)]
            # left, top, right, bottom = convert_yolo_coordinates_to_voc(x_c_n, y_c_n, width_n, height_n, img_width,
            #                                                            img_height)
            ## add new line to file
            # print(obj_name + " " + str(left) + " " + str(top) + " " + str(right) + " " + str(bottom))
            new_f.write(obj_name + " " + str(confidence) + " " + str(left) + " " + str(bottom) + " " + str(right) + " " + str(top) + '\n')


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
    if 'without' in obj.find('name').text or 'no' in obj.find('name').text:
        return 0
    elif 'with_' in obj.find('name').text:
        return 1
    elif 'incorrect' in obj.find('name').text:
        return 2


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


