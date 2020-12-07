import torch
# import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
# import matplotlib.pyplot as plt
import os
import argparse
import torch.nn as nn

from FCCNet import *
from FCCDataset import MaskDataset
from FCC_utils import *

def collate_fn(batch):
    return tuple(zip(*batch))

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
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
        os.makedirs('../data/data2/FasterRCNN/outputs', exist_ok=True)
        os.makedirs('../data/data2/FasterRCNN/checkpoints', exist_ok=True)

        #train the model
        FCC_taining(trainloader, net, device, criterion, optimizer)
        
        # save trained model
        PATH = 'cnn_net.pth'
        torch.save(net.state_dict(), PATH)
    
    elif args.mode == 'eval':
        testloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)
        # load the trained model
        net = CnnNet()
        net.load_state_dict(torch.load(PATH))

        # network performs on the whole dataset.
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

        #  the classes that performed well, and the classes that did not perform well
        # class_correct = list(0. for i in range(10))
        # class_total = list(0. for i in range(10))
        # with torch.no_grad():
        #     for data in testloader:
        #         images, labels = data
        #         outputs = net(images)
        #         _, predicted = torch.max(outputs, 1)
        #         c = (predicted == labels).squeeze()
        #         for i in range(4):
        #             label = labels[i]
        #             class_correct[label] += c[i].item()
        #             class_total[label] += 1

        # for i in range(10):
        #     print('Accuracy of %5s : %2d %%' % (
        #         classes[i], 100 * class_correct[i] / class_total[i]))

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='A CNN based image colorizer.'
    )
    parser.add_argument('-data_path', default='./data/data2/',
                        help='data path of file containing the data')
    # parser.add_argument('-labels_path', default='../data/data2/annotations/',
    #                     help='data path of file containing the annotations')
    parser.add_argument('-model_path', default='./data/data2/FasterRCNN/checkpoints/model-epoch-253-losses-0.0000.pth',
                        help='Pre-trained model path of FasterRCNN')
    parser.add_argument('-mode', default='train', help='Option: train/eval')

    return parser.parse_args(args)

if __name__ == '__main__':
    main(parse_args())