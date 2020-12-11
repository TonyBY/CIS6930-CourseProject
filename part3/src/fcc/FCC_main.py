import torch
# import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
# import matplotlib.pyplot as plt
import os
import argparse
import torch.nn as nn
import glob

from FCCNet import *
from FCCDataset import MaskDataset
from FCC_utils import *

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
    os.makedirs('../../data/data2/FCCNet/outputs/images', exist_ok=True)
    os.makedirs('../../data/data2/FCCNet/outputs/text', exist_ok=True)
    os.makedirs('../../data/data2/FCCNet/checkpoints', exist_ok=True)

    if args.mode == 'train':
        imgs_path = args.data_path + "training/images/"
        labels_path = args.data_path + "training/annotations/"
    elif args.mode == "eval":
        imgs_path = args.data_path + "testing/images/"
        labels_path = args.data_path + "testing/annotations/"
    
    data_transform = transforms.Compose([transforms.ToTensor()])
    dataset = MaskDataset(data_transform, imgs_path, labels_path, args.mode)
    if args.mode == 'train':
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fn, shuffle=False)
        net = FccNet()
        net.to(device)
        # Loss function and optimizer
        # parameters
        params = [p for p in net.parameters() if p.requires_grad]
        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(params, lr=0.001,
                                    momentum=0.9, weight_decay=0.0005)
        len_dataloader = len(trainloader)

        # Make folders and set parameters
        
        #train the model
        FCC_taining(trainloader, net, device, criterion, optimizer)
    
    
    elif args.mode == 'eval':
        txt_list = getAnnotationInDir(imgs_path)   
        testloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)
        # load the trained model
        net = FccNet()

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
                    # obj_thred, bg_thred, win_size = 0.5, 1, 50
                    obj_thred, bg_thred, win_size = 0.4, 1, 50
                    outputs = FCC_testing(net, images, obj_thred, bg_thred, win_size)
                except RuntimeError:
                    continue

                #compute accuracy
                try:  
                    # print("Saving generated annotations...")
                    tmp_file = "../../data/data2/FCCNet/outputs/text/" + imgPath[0].split('/')[-1][:-3] + "txt"
                    save_prediction(outputs, tmp_file)
                    # print("Done.")

                    img_path = "../../data/data2/FCCNet/outputs/images/"
                    # print("Prediction")
                    plot_image(images[0], outputs, img_path+"prediction_%s" % str(i+TEST_SET_START_IDX))

                    # print("Target")
                    plot_image(images[0], labels[0], img_path+"target_%s" % str(i+TEST_SET_START_IDX))

                    i += 1
                except RuntimeError:
                    continue
  



def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='A FCC based image colorizer.'
    )
    parser.add_argument('-data_path', default='../../data/data2/',
                        help='data path of file containing the data')
    # parser.add_argument('-labels_path', default='../data/data2/annotations/',
    #                     help='data path of file containing the annotations')
    parser.add_argument('-model_path', default="../../data/data2/FCCNet/checkpoints/FCCmodel-epoch-98-losses-0.19462039.pth",
                        help='Pre-trained model path of FCC')
    parser.add_argument('-mode', default='train', help='Option: train/eval')
    
    return parser.parse_args(args)


if __name__ == '__main__':
    main(parse_args())
