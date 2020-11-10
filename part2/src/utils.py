import numpy as np
import cv2
import os, glob
from PIL import Image
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import tensorflow as tf

import torch
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
    scale = random.uniform(0.6, 1)
    scaled_rgb_image = image*scale
    return scaled_rgb_image

def flip_coin():
    return random.choice([True,False])

#augment dataset n times
def augment_dataset(images,n):
    num_images = images.shape[0]
    augmented_data = torch.empty((n*num_images)+num_images, 3, 128, 128)
    
    #random crop and random flip 
    # crop_transform = transforms.Compose([transforms.RandomResizedCrop((128,128)),transforms.ToTensor()])
    # flip_transforms = transforms.Compose([transforms.RandomHorizontalFlip(p=.7)])
  
    image_num = 0

    #store original training set
    for image in images:
        augmented_data[image_num] = torch.Tensor(np.array(image).astype(np.uint8))
        image_num+=1

    #augment training set and store (iterate over 10 times)
    for i in range(n):
        for image in images:
            # if flip_coin:
            #     transformed_image = np.fliplr(np.array(image))
            # else:
            #     transformed_image = image
            # if flip_coin:
            #     transformed_image = np.fliplr(np.array(transformed_image))
            tf.random_crop(image, image)
            
            transformed_image = cv2.resize(np.array(transformed_image), (128,128), interpolation = cv2.INTER_AREA) 
            transformed_image = scale_rgb(transformed_image)
            print(transformed_image)
            print(a)
            augmented_data[image_num] = torch.Tensor(np.array(transformed_image).astype(np.uint8))
            image_num+=1
    return augmented_data

#convert to L a b color space and normalize
def convert_to_LAB(images,use_tanh=False):
    num_images = images.shape[0]
    LAB_data = torch.empty(num_images, 3, 128, 128)
    
    images = images.permute(0,2,3,1)
    for i, image in enumerate(images):
        image = np.array(image).astype(np.uint8)
        # image = (np.array(image).astype("float32")) / 255      
        imageLAB = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        imageLAB = torch.Tensor(imageLAB).permute(2,0,1)
        if use_tanh:
            imageLAB = 2*(imageLAB/255)-1
        else:
            imageLAB = imageLAB/255
        LAB_data[i] = torch.Tensor(imageLAB)
    return LAB_data

def convert_to_rgb(L_input, ab_input, savefolder=None, savefile=None, use_tanh=False):
    #concat channels
    gray = L_input.squeeze().numpy()
    LAB_image = torch.cat((L_input, ab_input), 0).numpy() 
    LAB_image = LAB_image.transpose((1, 2, 0))
    if use_tanh:
        LAB_image = ((LAB_image+1)/2)*255
    else:
        LAB_image = LAB_image*255
    LAB_image = np.array(LAB_image).astype(np.uint8)
    rgb_image = cv2.cvtColor(LAB_image, cv2.COLOR_LAB2RGB)
    plt.imsave(arr=gray, fname=savefolder+'/gray/'+savefile, cmap='gray')
    plt.imsave(arr=rgb_image, fname=savefolder+'/color/'+savefile,)
    plt.clf()

def train(model, opt, crit, train_data, model_type, use_tanh=False, use_gpu=False):
    total_loss = 0
    num_samps = 0
  
    model.train()
    for i, image in enumerate(train_data):
        if i % 100 == 49:
            print('Sample: ' + str(i+1) + '/' + str(len(train_data)))
            print('Current Average Loss: ' + str(total_loss/num_samps))
        L_channel = image[:,0,:,:]
        L_channel = L_channel.unsqueeze(1)
        ab_channel = image[:,1:3,:,:]
        if use_gpu: 
            L_channel, ab_channel = L_channel.cuda(), ab_channel.cuda()
        out_ab = model(L_channel,use_tanh)
        
        if model_type == 'regressor':
            ab_channel = ab_channel.mean([2,3],True)
        loss = crit(out_ab, ab_channel)
        total_loss += loss.item()*L_channel.size(0)
        num_samps += L_channel.size(0)

        opt.zero_grad()
        loss.backward()
        opt.step()

def test(model, opt, crit, test_data, store, model_type, use_tanh=False, use_gpu=False):
    total_loss = 0
    num_samps = 0

    model.eval()

    for i, image in enumerate(test_data):
        if i % 4 == 0 and i != 0:
            print('Sample: ' + str(i+1) + '/' + str(len(test_data)))
            print('Current Average Loss: ' + str(total_loss/num_samps))
        L_channel = image[:,0,:,:]
        L_channel = L_channel.unsqueeze(1)
        ab_channel = image[:,1:3,:,:]
        if use_gpu: 
            L_channel, ab_channel = L_channel.cuda(), ab_channel.cuda()
        if model_type == 'regressor':
            ab_channel = ab_channel.mean([2,3],True)
        out_ab = model(L_channel,use_tanh)
        loss = crit(out_ab, ab_channel)
        total_loss += loss.item()*L_channel.size(0)
        num_samps += L_channel.size(0)
        if model_type == 'colorize' and store:
            savefolder = '../data/out/images/colorize/'
            savefile = 'image' + str(i * L_channel.shape[0] + i)+'.jpg'
            convert_to_rgb(L_channel[i].cpu(), out_ab[i].detach().cpu(), savefolder,savefile,use_tanh)

        #     for j in range(L_channel.shape[0]): 
        #         print(L_channel.shape[0])
        #         # print(len(test_data))
        #         # print(a)
        #         savefolder = '../data/out/images/colorize/'
        #         savefile = 'image' + str(i * test_data.batch_size + j)+'.jpg'
        #         convert_to_rgb(L_channel[j].cpu(), out_ab[j].detach().cpu(), savefolder,savefile,use_tanh)
    return (total_loss/num_samps)