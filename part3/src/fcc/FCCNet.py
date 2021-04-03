import torch
import torch.nn as nn
import torch.nn.functional as F

from FCCDataset import create_image_label 
from FCC_utils import *

class_list = ['without_mask', 'with_mask', 'mask_weared_incorrect']
CLASS_NUM = 4
BATCH_SIZE = 4
LEARNING_RATE = 0.001
BEST_LOSS = 0.001

# FCC network
class FccNet(nn.Module):
    def __init__(self):
        super(FccNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fcc1 = nn.Linear(16 * 23 * 23,16 * 23 * 23)
        self.fcc2 = nn.Linear(16 * 23 * 23, 4)
        # self.fc2 = nn.Linear(120, 84)
        # self.fcbb = nn.Linear(84, 4)
        # self.fcclass = nn.Linear(84,2)

    def forward(self, x):
        # print(x.size())
        x_copy = x.detach().clone()
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # print(x.size())
        x = x.view(x_copy.shape[0],16*23*23)
        # print(x.size())
        x = F.relu(self.fcc1(x))
        # print(x.size())
        result = nn.Softmax(dim = 1)(self.fcc2(x))
        # print(result.size())
        return result

def FCC_taining(trainloader,net, device, criterion, optimizer):
    len_dataloader = len(trainloader)
    best_losses = BEST_LOSS

    for epoch in range(100):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (imgs, annotations,_) in enumerate(trainloader):
            # 4 imgs and 4 annotations:{boxes, label, image_id}
            imgs = list(img.to(device) for img in imgs)
            imgs = torch.stack(imgs)
            #annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            # batch:  torch.Size([4, 3, 300, 300])
            # loop over the sliding windows
            (winW, winH) = (50, 50)
            for resized_imgs in pyramid(imgs, 2, (winW, winH)):
                print(resized_imgs.size())
                scale = resized_imgs.size()[2]/imgs.size()[2]
                normalized_label = create_image_label(annotations, CLASS_NUM, resized_imgs.size()[2], resized_imgs.size()[3], scale)
                # loop over the sliding window for each layer of the pyramid
                for (x, y, windows) in sliding_window(resized_imgs, 50, (winW, winH)):
                    # print(windows.shape)
                    # if the window does not meet our desired window size, ignore it
                    if windows.size()[3] != winH or windows.size()[2] != winW:
                        continue
                    windows = windows.to(device)
                    optimizer.zero_grad()
                    output = net(windows)
                    gt_win = get_window_label(normalized_label[:,:,y:y+winH, x:x+winW]).to(device)  
                    loss = criterion(output,gt_win)
                    print(loss)
                     # avg_batch_loss = losses/BATCH_SIZE
                    if loss < best_losses:
                        best_losses = loss
                        torch.save(net.state_dict(), '../../data/data2/FCCNet/checkpoints/model-epoch-{}-losses-{:.8f}.pth'.format(epoch + 1, loss))
                        print('done saving')
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    
            # print("batch ",i)
            if i % 100 == 99:
                print(f'Epoch: {epoch}, Batch: {i}/{len_dataloader}, loss: {running_loss/100}')
                running_loss = 0.0

    print('Finished Training')


def FCC_testing(net, imgs, obj_thred, bg_thred,win_size):
    result = {'boxes':[],'labels':[],'scores':[]}
    for resized_imgs in pyramid(imgs, 1.5, (win_size, win_size)):
                # print(resized_imgs.size())
                # 0.5
                scale = imgs.size()[2]/resized_imgs.size()[2]
                # loop over the sliding window for each layer of the pyramid
                step_size = 25
                for (x, y, windows) in sliding_window(resized_imgs, step_size, (win_size, win_size)):
                    # print(windows.shape)
                    # if the window does not meet our desired window size, ignore it
                    if windows.size()[3] != win_size or windows.size()[2] != win_size:
                        continue
                    output = net(windows)
                    # print("output",output)
                    xmin, xmax, ymin, ymax = scale*x, scale*(x+win_size), scale*y , scale*(y+win_size)
                    #consider background
                    bg = 3
                    for k in range(3):
                        if output[0,k]>=obj_thred and output[0,bg]<=bg_thred :
                            result['boxes'].append([xmin, ymin, xmax, ymax])
                            result['labels'].append(k)
                            result['scores'].append(output[0,k])
    return result



    output = net(images)
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
                    result['boxes'].append([xmin, ymin, xmax, ymax])
                    result['labels'].append(k)
                    result['scores'].append(output[0,k,i,j])
    print('^^^^^^^^')
    print(output[0,3,:,:].numpy())
    print('----')
    np.savetxt('test_1.out', output[0,1,:,:].numpy(), delimiter=',')     
    