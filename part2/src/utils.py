import numpy as np
import cv2
import os, time, glob
from PIL import Image
import random
from AverageMeter import AverageMeter
import matplotlib.pyplot as plt
import torch
from skimage.color import lab2rgb
from torchvision import transforms
torch.set_default_tensor_type('torch.FloatTensor')

#split data
def split(images):
    train_index = int(0.9 * images.shape[0])
    train, test = images[:train_index,:,:,:], images[train_index:,:,:,:]
    return train, test

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
    # for image in data_tensor:
    #     # print(image)
    #     # print(a)
    #     print(image.shape)
    #     new_image = np.array(image.permute(1,2,0))
    #     plt.imshow(cv2.cvtColor(new_image.astype('uint8'), cv2.COLOR_BGR2RGB))
    #     plt.show()
    #     print(a)
    return data_tensor

#scale rgb
def scale_rgb(image):
    sample = image.permute(1,2,0)
   
   
    scale = random.uniform(0.6, 1)
    scaled_rgb_image = image*scale

    return scaled_rgb_image

#augment dataset n times
def augment_dataset(images,n):
    num_images = images.shape[0]
    augmented_data = torch.empty(n*num_images, 3, 128, 128)
    
    #random crop and random flip 
    train_transforms = transforms.Compose([transforms.RandomResizedCrop((128,128)), transforms.RandomHorizontalFlip()])
  
    image_num = 0
    for i in range(n):
        transformed_images = train_transforms(images)
        for image in transformed_images:
            image = scale_rgb(image)
            augmented_data[image_num] = image
            image_num+=1
    return augmented_data

#convert to L a b color space and normalize
def convert_to_LAB(images):
    num_images = images.shape[0]
    LAB_data = torch.empty(num_images, 3, 128, 128)

    images = images.permute(0,2,3,1)

    for i, image in enumerate(images):
        image = (np.array(image).astype("float32")) / 255      
        imageLAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        imageLAB = torch.Tensor(imageLAB).permute(2,0,1)
       
        LAB_data[i] = imageLAB
    return LAB_data

def train(train_loader, model, criterion, optimizer, epoch, model_type, use_gpu=False):
    print('Starting to train. At epoch: {}'.format(epoch))
    losses = AverageMeter()
    model.train()
    for i, image in enumerate(train_loader):
        L_channel = image[:,0,:,:]/100
        L_channel = L_channel.unsqueeze(1)
        ab_channel = image[:,1:3,:,:]
        if use_gpu: 
            L_channel, ab_channel = L_channel.cuda(), ab_channel.cuda()
        out_ab = model(L_channel)
        if model_type == 'regressor':
            ab_channel = ab_channel.mean([2,3],True)
        loss = criterion(out_ab, ab_channel)
        losses.update(loss.item(), L_channel.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 1 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'.format(epoch,i,len(train_loader)))
            print('Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(loss=losses))

    print('Finished Training.')


def validate(test_loader, model, criterion, epoch, model_type, use_gpu=False):
    model.eval()

    losses = AverageMeter()

    already_saved_images = False
    for i, image in enumerate(test_loader):
        L_channel = image[:,0,:,:]/100
        L_channel = L_channel.unsqueeze(1)
        ab_channel = image[:,1:3,:,:]
        if use_gpu: 
            L_channel, ab_channel = L_channel.cuda(), ab_channel.cuda()
        if model_type == 'regressor':
            ab_channel = ab_channel.mean([2,3],True)
        out_ab = model(L_channel)
        loss = criterion(out_ab, ab_channel)
        losses.update(loss.item(), L_channel.size(0))

    # Save images to file
    if model_type == 'colorized' and save_images and not already_saved_images:
        already_saved_images = True
        for j in range(min(len(output_ab), 10)): # save at most 5 images
            save_path = {'grayscale': '../data/colorization/outputs/gray/', 'colorized': '../data/colorization/outputs/color/'}
            save_name = 'img-{}-epoch-{}.jpg'.format(i * val_loader.batch_size + j, epoch)
            to_rgb(input_gray[j].cpu(), ab_input=output_ab[j].detach().cpu(), save_path=save_path, save_name=save_name)

    if i % 1 == 0:
        print('Validate: [{0}/{1}]\t'.format(i+1,len(test_loader)))
        print('Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(loss=losses))
    print('Finished validation.')
    return losses.avg