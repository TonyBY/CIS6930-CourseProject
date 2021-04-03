import torch
# import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import glob
import argparse
import torch.nn as nn
import numpy as np
import sys
import matplotlib.patches as patches
import time


from CNNNet_withoutBG import *
from CNNDataset_withoutBG import MaskDataset
from CNN_utils import *

np.set_printoptions(threshold=sys.maxsize)

class_list = ['without_mask', 'with_mask', 'mask_weared_incorrect']
BATCH_SIZE = 4
LEARNING_RATE = 0.001
TEST_SET_START_IDX = 768

def collate_fn(batch):
    return tuple(zip(*batch))
def getAnnotationInDir(dir_path):
    annotation_list = []
    for filename in glob.glob(dir_path + '/*.txt'):
        annotation_list.append(filename.split('/')[-1])
    return annotation_list


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if args.mode == 'train':
        imgs_path = args.data_path + "training/images/"
        labels_path = args.data_path + "training/annotations/"
    elif args.mode == "eval":
        imgs_path = args.data_path + "testing/images/"
        labels_path = args.data_path + "testing/annotations/"
    # Make folders and set parameters
    os.makedirs('../../data/data2/CNNNet/withoutBG/outputs/images', exist_ok=True)
    os.makedirs('../../data/data2/CNNNet/withoutBG/outputs/text', exist_ok=True)
    os.makedirs('../../data/data2/CNNNet/withoutBG/checkpoints', exist_ok=True)

    data_transform = transforms.Compose([transforms.ToTensor()])
    dataset = MaskDataset(data_transform, imgs_path, labels_path, args.mode)
    if args.mode == 'train':
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=False)
        net = CnnNet()
        net.to(device)
        # Loss function and optimizer
        # parameters
        params = [p for p in net.parameters() if p.requires_grad]
        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(params, lr=LEARNING_RATE,
                                    momentum=0.9, weight_decay=0.0005)
        len_dataloader = len(trainloader)

        #train the model
        CNN_taining(trainloader, net, device, criterion, optimizer)
        
    
    elif args.mode == 'eval':
        txt_list = getAnnotationInDir(imgs_path)
        testloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)
        # load the trained model
        net = CnnNet()
        
        #without bg small boxes
        # model_path = "./data/data2/CNNNet/withoutBG/checkpoints/model-epoch-100-losses-0.32230151.pth"
     
        #large net
        model_path = args.model_path
        net.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))

        # network performs on the whole dataset.
        correct = 0
        total = 0
        i = 0
        with torch.no_grad():
            for images, labels, imgPath in testloader:
                # print("###################################")
                # print("imgPath: ", imgPath)
                # print("i + %s: " % TEST_SET_START_IDX, i + TEST_SET_START_IDX)

                try:
                    # print(imgPath)
                    images = torch.stack(images)
                    # large: 0.8 small:0.6
                    obj_thred = 0.6
                    outputs = CNN_testing(net, images, obj_thred)

                except RuntimeError:
                    continue

                #compute accuracy
                try:
                    # print("Saving generated annotations...")
                    tmp_file = "../../data/data2/CNNNet/withoutBG/outputs/text/" + imgPath[0].split('/')[-1][:-3] + "txt"
                    save_prediction(outputs, tmp_file)
                    # print("Done.")

                    img_path = "../../data/data2/CNNNet/withoutBG/outputs/images/"
                    # print("Prediction")
                    plot_image(images[0], outputs, img_path+"prediction_%s" % str(i+TEST_SET_START_IDX))

                    # print("Target")
                    plot_image(images[0], labels[0], img_path+"target_%s" % str(i+TEST_SET_START_IDX))

                    i += 1
                except RuntimeError:
                    continue


       
def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='A CNN based image colorizer.'
    )
    parser.add_argument('-data_path', default='../../data/data2/',
                        help='data path of file containing the data')
    # parser.add_argument('-labels_path', default='../data/data2/annotations/',
    #                     help='data path of file containing the annotations')
    parser.add_argument('-model_path', default="../../data/data2/CNNNet/withoutBG/checkpoints/largemodel-epoch-100-losses-0.17985973.pth", 
                        help='Pre-trained model path of FasterRCNN')
    parser.add_argument('-mode', default='eval', help='Option: train/eval')

    return parser.parse_args(args)

if __name__ == '__main__':
    main(parse_args())
    print("BATCH_SIZE: ", BATCH_SIZE)
    print("LEARNING_RATE: ", LEARNING_RATE)
    begin_time = time.time()
    main(parse_args())
    end_time = time.time()
    total_time = end_time - begin_time
    print("Total time: ", total_time)
    print("Total time in hours: ", total_time/3600)