import argparse
import torch
import torch.nn as nn
import os
import shutil

from CNNRegressor import CNNRegressor
from Colorize import Colorize

from utils import load_images, split, augment_dataset, convert_to_LAB, train, test


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='A CNN based image colorizer.'
    )
    parser.add_argument('-data_path', default='../data/face_images/',
                        help='data path of file containing the face images')
    parser.add_argument('-model_path', default='../data/out/ckpts/colorize/model-epoch-30-losses-0.0005.pth',
                        help='Load model path of Colorize')
    parser.add_argument('-save_path', default='../data/out/',
                        help='Pre-trained model path of Colorize')
    parser.add_argument('-store', action='store_true',
                        help='Cache grey image and colorized images of validation set.')
    parser.add_argument('-mode', default='train', help='Option: train/test')
    parser.add_argument('-use_gpu', action='store_true',
                        help='Use gup if available.')
    parser.add_argument('-model', default='colorize', help='Option: regressor/colorize')
    parser.add_argument('-use_tanh', action='store_true')
    return parser.parse_args(args)


def main(args):
    ###Data Preprocessing####
    # Check if GPU is available
    use_gpu = torch.cuda.is_available() and args.use_gpu

    data_path = args.data_path
    print("Preparing Data.")
    #creates tensor of all images and splits
    images = load_images(data_path=data_path)
    train_images, test_images = split(images)
   
    if args.mode == 'train':
        n = 10
        #augments images by a factor of n
        train_images = augment_dataset(train_images,n)
        #Convert images to L∗a∗b∗ color space and normalize between 0 and 1
        train_images_LAB = convert_to_LAB(train_images,args.use_tanh)
        train_data = torch.utils.data.DataLoader(train_images_LAB, batch_size=16, shuffle=True)


    test_images_LAB = convert_to_LAB(test_images,args.use_tanh)

    test_data = torch.utils.data.DataLoader(test_images_LAB, batch_size=16, shuffle=True)
    
    ###Model Init###
    if args.model == "regressor":
        model = CNNRegressor()
    elif args.model == 'colorize':
        model = Colorize()
    else:
        raise ValueError('Invalid Model.')

    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.01)

    if use_gpu:
        model = model.cuda()
        crit = crit.cuda()
 
    if args.mode == 'train':
        if os.path.isdir(args.save_path):
            shutil.rmtree(args.save_path)
        os.makedirs(args.save_path)
        os.makedirs(args.save_path+'images/colorize/color')
        os.makedirs(args.save_path+'images/colorize/gray')
        os.makedirs(args.save_path+'ckpts/regressor/')
        os.makedirs(args.save_path+'ckpts/colorize/')
        best_loss = 100000000
        epochs = 30

        # Train model
        for epoch in range(epochs):
            # Train for one epoch, then test
            print('Training... Epoch:' + str(epoch) )
            train(model, opt, crit, train_data, args.model, use_tanh=args.use_tanh, use_gpu=args.use_gpu)
            with torch.no_grad():
                tloss = test(model, opt, crit, test_data, args.store, args.model, use_tanh=args.use_tanh, use_gpu=args.use_gpu)
            
            # Save checkpoint and replace old best model if current model is better
            if tloss < best_loss:
                best_loss = tloss
                if args.model == "colorize":
                    save_path_model = args.save_path+'ckpts/colorize/'
                    torch.save(model.state_dict(),save_path_model + str(epoch) + 'with_tloss' + str(tloss) + 'colorizer.pth')
                elif args.model == "regressor":
                    save_path_model = args.save_path+'ckpts/regressor/'
                    torch.save(model.state_dict(),save_path_model + str(epoch) + 'with_tloss' + str(tloss) + 'regressor.pth')
                else:
                    raise ValueError('Invalid model.')

    elif args.mode == 'test':
        model.load_state_dict(torch.load(args.model_path))
        if use_gpu:
            model.cuda()
        with torch.no_grad():
            # print(len(test_data))
            # print(a)
            test(model, opt, crit, test_data, args.store, args.model, use_tanh=args.use_tanh, use_gpu=args.use_gpu)
    else:
        raise ValueError('mode can only be demo, test or train.')


if __name__ == '__main__':
    main(parse_args())