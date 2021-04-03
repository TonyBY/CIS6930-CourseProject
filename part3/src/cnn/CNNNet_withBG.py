import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from CNNDataset_withBG import create_window_label 

class_list = ['without_mask', 'with_mask', 'mask_weared_incorrect']
BATCH_SIZE = 4
LEARNING_RATE = 0.001
BEST_LOSS = 100

# CNN network
class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)
        self.conv2 = nn.Conv2d(64, 16, 5)
        self.conv3 = nn.Conv2d(16, 16, 5)
        self.conv4 = nn.Conv2d(16, 16, 5)
        self.conv4 = nn.Conv2d(16, 16, 5)
        
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.pool4 = nn.MaxPool2d(2,2)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(16)
        self.bn4 = nn.BatchNorm2d(16)
        
        self.conv5 = nn.Conv2d(16, 5*5*16, 5)
        self.conv6 = nn.Conv2d(5*5*16, 5*5*16, 1)
        self.conv7 = nn.Conv2d(5*5*16, 4, 1)

        self.bn5 = nn.BatchNorm2d(5*5*16)
        self.bn6 = nn.BatchNorm2d(5*5*16)
        self.bn7 = nn.BatchNorm2d(4)
        
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fcbb = nn.Linear(84, 4)
        # self.fcclass = nn.Linear(84,2)

    def forward(self, x):
        # print(x.size())
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        # print(x.size())
        x = F.relu(self.bn5(self.conv5(x)))
        # print(x.size())
        x = F.relu(self.bn6(self.conv6(x)))
        # print(x.size())
        result = nn.Softmax(dim = 1)(F.relu(self.bn7(self.conv7(x))))
        # print(result.shape)
        return result

def CNN_taining(trainloader,net, device, criterion, optimizer):
    len_dataloader = len(trainloader)
    best_losses = BEST_LOSS

    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (imgs, annotations, _) in enumerate(trainloader):
            # 4 imgs and 4 annotations:{boxes, label, image_id}
            #print()
            imgs = list(img.to(device) for img in imgs)
            imgs = torch.stack(imgs)
            #annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            # image shape: torch.Size([3, 300, 300])
            optimizer.zero_grad()
            output = net(imgs)
            gt = create_window_label(annotations,output.size()[2],imgs.size()[2]).to(device)
            loss = criterion(output,gt)
            if loss < best_losses:
                best_losses = loss
                torch.save(net.state_dict(),
                            '../../data/data2/CNNNet/withBG/checkpoints/CNNwithBGmodel-epoch-{}-losses-{:.8f}.pth'.format(epoch + 1, loss))
                # print('done saving')

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # print(i)
            if i % 100 == 99:
                print(f'Epoch: {epoch}, Batch: {i}/{len_dataloader}, loss: {running_loss/100}')
                running_loss = 0.0
            print(i)

    print('Finished Training')

def CNN_testing(net, images, obj_thred, bg_thred):
    result = {'boxes':[],'labels':[],'scores':[]}
    output = net(images)
    print("output:",output.shape)
    # image shape: torch.Size([3, 300, 300])
    # result shape: 
    count = output.size()[3]
    scale = images.size()[3]/count
    for i in range(count):
        for j in range(count):
            # check if it's obj
            xmin, xmax, ymin, ymax = int(i*scale), int(i*scale+scale), int(j*scale), int(j*scale+scale)
            bg = 3
            for k in range(3):
                if output[0,k,i,j]>=obj_thred and output[0,bg,i,j]<=bg_thred :
                # if output[0,k,i,j]>=obj_thred:
                    result['boxes'].append([xmin, ymin, xmax, ymax])
                    result['labels'].append(k)
                    result['scores'].append(output[0,k,i,j])
    # print('^^^^^^^^')
    # print(output[0,3,:,:].numpy())
    # print('----')
    # np.savetxt('test_1.out', output[0,1,:,:].numpy(), delimiter=',')     
    return result