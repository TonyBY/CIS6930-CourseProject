import numpy as np
import cv2
import os, time, glob
import random
from AverageMeter import AverageMeter
import matplotlib.pyplot as plt
import torch
from skimage.color import lab2rgb
from torchvision import transforms
torch.set_default_tensor_type('torch.FloatTensor')

#load images in tensor
def load_images(data_path="../data/face_images/"):
    dirname = os.path.dirname(__file__)
    data_path = os.path.join(dirname, data_path)
    files = glob.glob(str(data_path)+'*.jpg')
    num_images = len(files)
    data =[]
    for f1 in files:
        img = cv2.imread(f1)
        data.append(img)
    data_tensor = torch.Tensor(data).permute(0,3,1,2)
    return data_tensor

#scale rgb
def scale_rgb(image):
    scale = random.uniform(0.6, 1)
    scaled_rgb_image = image*scale
    return scaled_rgb_image

#augment dataset n times
def augment_dataset(images,n):
    num_images = images.shape[0]
    augmented_data = torch.empty(n*num_images, 3, 128, 128)
    #random crop and random flip 
    train_transforms = transforms.Compose([transforms.RandomResizedCrop((128,128)), transforms.RandomHorizontalFlip()])
  
    for i in range(n):
        transformed_images = train_transforms(images)
        for j, image in enumerate(transformed_images):
            image = scale_rgb(image)
            augmented_data[j] = image
    return augmented_data

#convert to L a b color space
def convert_to_LAB(images):
    num_images = images.shape[0]
    LAB_data = torch.empty(num_images, 3, 128, 128)

    images = images.permute(0,2,3,1)

    for i, image in enumerate(images):
        imageLAB = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2LAB)
        imageLAB = torch.Tensor(imageLAB).permute(2,0,1)
        LAB_data[i] = imageLAB
    return LAB_data

    
# Move data into training and validation directories
def split_data_set(data_path='../data/face_images/', size_of_validation_set=50):
    os.makedirs(data_path + 'train/class/', exist_ok=True) # 700 images
    os.makedirs(data_path + 'val/class/', exist_ok=True)   #  50 images
    for i, file in enumerate(os.listdir(data_path)):
        if "jpg" in file:
            if i < size_of_validation_set: # first 50 will be val
                os.rename(data_path + file, data_path + 'val/class/' + file)
            else: # others will be train
                os.rename(data_path + file, data_path + 'train/class/' + file)


def to_rgb(grayscale_input, ab_input, save_path=None, save_name=None):
    '''Show/save rgb image from grayscale and ab channels
     Input save_path in the form {'grayscale': '/path/', 'colorized': '/path/'}'''
    plt.clf() # clear matplotlib
    color_image = torch.cat((grayscale_input, ab_input), 0).numpy() # combine channels
    color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
    color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
    color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
    color_image = lab2rgb(color_image.astype(np.float64))
    grayscale_input = grayscale_input.squeeze().numpy()
    if save_path is not None and save_name is not None:
        plt.imsave(arr=grayscale_input, fname='{}{}'.format(save_path['grayscale'], save_name), cmap='gray')
        plt.imsave(arr=color_image, fname='{}{}'.format(save_path['colorized'], save_name))


def train(train_loader, model, criterion, optimizer, epoch, use_gpu=False):
    print('Starting training epoch {}'.format(epoch))
    model.train()

    # Prepare value counters and timers
    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    for i, (input_gray, input_ab, target) in enumerate(train_loader):
        # Use GPU if available
        if use_gpu: input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()

        # Record time to load data (above)
        data_time.update(time.time() - end)

        # Run forward pass
        output_ab = model(input_gray)
        loss = criterion(output_ab, input_ab)
        losses.update(loss.item(), input_gray.size(0))

        # Compute gradient and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record time to do forward and backward passes
        batch_time.update(time.time() - end)
        end = time.time()

        # Print model accuracy -- in the code below, val refers to value, not validation
        if i % 1 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                  epoch, i, len(train_loader), batch_time=batch_time,
                 data_time=data_time, loss=losses))

    print('Finished training epoch {}'.format(epoch))


def validate(val_loader, model, criterion, save_images, epoch, use_gpu=False):
    model.eval()

    # Prepare value counters and timers
    batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    already_saved_images = False
    for i, (input_gray, input_ab, target) in enumerate(val_loader):
        data_time.update(time.time() - end)

    # Use GPU
    if use_gpu: input_gray, input_ab, target = input_gray.cuda(), input_ab.cuda(), target.cuda()

    # Run model and record loss
    output_ab = model(input_gray) # throw away class predictions
    loss = criterion(output_ab, input_ab)
    losses.update(loss.item(), input_gray.size(0))

    # Save images to file
    if save_images and not already_saved_images:
        already_saved_images = True
        for j in range(min(len(output_ab), 10)): # save at most 5 images
            save_path = {'grayscale': '../data/colorization/outputs/gray/', 'colorized': '../data/colorization/outputs/color/'}
            save_name = 'img-{}-epoch-{}.jpg'.format(i * val_loader.batch_size + j, epoch)
            to_rgb(input_gray[j].cpu(), ab_input=output_ab[j].detach().cpu(), save_path=save_path, save_name=save_name)

    # Record time to do forward passes and save images
    batch_time.update(time.time() - end)
    end = time.time()

    # Print model accuracy -- in the code below, val refers to both value and validation
    if i % 1 == 0:
        print('Validate: [{0}/{1}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
             i+1, len(val_loader), batch_time=batch_time, loss=losses))

    print('Finished validation.')
    return losses.avg