import argparse
import torch
from torchvision import transforms
from IPython.display import Image, display
from os import path
import torch.nn as nn
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from Colorize import Colorize
from Regressor import Regressor

from GrayscaleImageFolder import GrayscaleImageFolder
from colorization_utils import train, validate, split_data_set


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
    parser.add_argument('-split_dataset', action='store_true',
                        help='Split the dataset into training set and validation set.')
    parser.add_argument('-validation_size', default='50', help='the number of images in the validation set.')
    parser.add_argument('-mode', default='train', help='Option: train/eval')
    parser.add_argument('-use_gpu', action='store_true',
                        help='Use gup if available.')
    return parser.parse_args(args)


def main(args):
    # Check if GPU is available
    use_gpu = torch.cuda.is_available() and args.use_gpu

    data_path = args.data_path
    if args.split_dataset and path.exists(data_path + "image00000.jpg"):
        split_data_set(data_path=data_path, size_of_validation_set=int(args.validation_size))


    # Make sure the images are there
    display(Image(filename=data_path + 'train/class/image00000.jpg'))

    # Training
    train_transforms = transforms.Compose([transforms.RandomResizedCrop((128,128)), transforms.RandomHorizontalFlip()])
    train_imagefolder = GrayscaleImageFolder(data_path + 'train', train_transforms)
    train_loader = torch.utils.data.DataLoader(train_imagefolder, batch_size=64, shuffle=True)

    # Validation
    val_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
    val_imagefolder = GrayscaleImageFolder(data_path + 'val', val_transforms)
    val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=64, shuffle=False)

    model = Colorize()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0)

    save_images = args.save_images

    if args.mode == 'eval':
        # Load model
        pretrained = torch.load(
            args.model_path,
            map_location=lambda storage, loc: storage)


        model.load_state_dict(pretrained)
        if use_gpu:
            model.cuda()

        # Validate
        with torch.no_grad():
            validate(val_loader, model, criterion, save_images, 0, use_gpu=use_gpu)
    elif args.mode == 'train':
        if use_gpu:
            criterion = criterion.cuda()
            model = model.cuda()

        # Make folders and set parameters
        os.makedirs('../data/colorization/outputs/color', exist_ok=True)
        os.makedirs('../data/colorization/outputs/gray', exist_ok=True)
        os.makedirs('../data/colorization/checkpoints', exist_ok=True)
        best_losses = 1e10
        epochs = 20

        # Train model
        for epoch in range(epochs):
            # Train for one epoch, then validate
            train(train_loader, model, criterion, optimizer, epoch,use_gpu=use_gpu)
            with torch.no_grad():
                losses = validate(val_loader, model, criterion, save_images, epoch,use_gpu=use_gpu)
            # Save checkpoint and replace old best model if current model is better
            if losses < best_losses:
                best_losses = losses
                torch.save(model.state_dict(),
                           '../data/colorization/checkpoints/model-epoch-{}-losses-{:.3f}.pth'.format(epoch + 1, losses))
                print('done saving')
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