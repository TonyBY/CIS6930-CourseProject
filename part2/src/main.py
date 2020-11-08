import argparse
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from IPython.display import Image, display
from os import path
import torch.nn as nn
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from CNNRegressor import CNNRegressor
from ColorizationNet import ColorizationNet
from utils import load_images, split, augment_dataset, convert_to_LAB, train, validate


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='A CNN based image colorizer.'
    )
    parser.add_argument('-data_path', default='../data/face_images/',
                        help='data path of file containing the face images')
    parser.add_argument('-model_path', default='../data/colorization/checkpoints/model-epoch-1-losses-0.004.pth/',
                        help='Pre-trained model path of ColorizationNet')
    parser.add_argument('-save_images', action='store_true',
                        help='Cache grey image and colorized images of validation set.')
    parser.add_argument('-mode', default='train', help='Option: train/eval')
    parser.add_argument('-use_gpu', action='store_true',
                        help='Use gup if available.')
    parser.add_argument('-model', default='regressor', help='Option: regressor/colorize')
    return parser.parse_args(args)


def main(args):
    ###Data Preprocessing####
    # Check if GPU is available
    use_gpu = torch.cuda.is_available() and args.use_gpu

    data_path = args.data_path
    #creates tensor of all images and splits
    images = load_images(data_path=data_path)
    train_images, test_images = split(images)
   
    n = 10
    print(train_images.shape)
    #augments images by a factor of n
    train_images = augment_dataset(train_images,n)
    #Convert images to L∗a∗b∗ color space and normalize between 0 and 1
    train_images_LAB = convert_to_LAB(train_images)
    test_images_LAB = convert_to_LAB(test_images)
    print(train_images_LAB[-1,:,:,:])
    print(a)
    # output_test = test_images_LAB[:,1:3,:,:]

    train_loader = torch.utils.data.DataLoader(train_images_LAB, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_images_LAB, batch_size=16, shuffle=True)
    for i in train_loader:
        print(i)
    print(a)
    ###Model Init###
    if args.model == "regressor":
        model = CNNRegressor()
    elif args.model == 'colorize':
        model = ColorizationNet()
    else:
        raise ValueError('Invalid Model.')

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)

    if use_gpu:
        model = model.cuda()
        criterion = criterion.cuda()

    save_images = args.save_images

 
    if args.mode == 'train':
        # Make folders and set parameters
        os.makedirs('../data/colorization/outputs/color', exist_ok=True)
        os.makedirs('../data/colorization/outputs/gray', exist_ok=True)
        os.makedirs('../data/regressor/ckpts/', exist_ok=True)
        best_loss = 100000000
        epochs = 10

        # Train model
        for epoch in range(epochs):
            # Train for one epoch, then validate
            train(train_loader, model, criterion, optimizer, epoch, args.model, use_gpu=args.use_gpu)
            with torch.no_grad():
                losses = validate(test_loader, model, criterion, epoch, args.model, use_gpu=args.use_gpu)
            # Save checkpoint and replace old best model if current model is better
            if losses < best_loss:
                best_loss = losses
                torch.save(model.state_dict(),
                           '../data/regressor/ckpts/model-epoch-{}-losses-{:.3f}.pth'.format(epoch + 1, losses))
    
    elif args.mode == 'eval':
        # Load model
        pretrained = torch.load(
            args.model_path,
            map_location=lambda storage, loc: storage)


        model.load_state_dict(pretrained)

        # Validate
        with torch.no_grad():
            validate(val_loader, model, criterion, save_images, 0, use_gpu=use_gpu)   
    
    
    elif args.mode == 'demo':
        # Show images
        image_pairs = [
            ('../data/colorization/outputs/color/img-0-epoch-0.jpg',
             '../data/colorization/outputs/gray/img-0-epoch-0.jpg'),
            ('../data/colorization/outputs/color/img-1-epoch-0.jpg',
             '../data/colorization/outputs/gray/img-1-epoch-0.jpg')]
        for c, g in image_pairs:
            color = mpimg.imread(c)
            gray = mpimg.imread(g)
            f, axarr = plt.subplots(1, 2)
            f.set_size_inches(15, 15)
            axarr[0].imshow(gray, cmap='gray')
            axarr[1].imshow(color)
            axarr[0].axis('off'), axarr[1].axis('off')
            plt.show()
    else:
        raise ValueError('mode can only be demo, eval or train.')


if __name__ == '__main__':
    main(parse_args())