import torch
import torch.nn as nn
import torch.nn.functional as F

from CNNDataset import create_window_label 

class_list = ['without_mask', 'with_mask', 'mask_weared_incorrect']
BATCH_SIZE = 4
LEARNING_RATE = 0.001
BEST_LOSS = 0.001

# CNN network
class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 5*5*16, 5)
        self.conv3 = nn.Conv2d(5*5*16, 5*5*16, 1)
        self.conv4 = nn.Conv2d(5*5*16, 4, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(5*5*16)
        self.bn3 = nn.BatchNorm2d(5*5*16)
        self.bn4 = nn.BatchNorm2d(4)
        
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fcbb = nn.Linear(84, 4)
        # self.fcclass = nn.Linear(84,2)

    def forward(self, x):
        # print(x.size())
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # print(x.size())
        x = F.relu(self.bn2(self.conv2(x)))
        # print(x.size())
        x = F.relu(self.bn3(self.conv3(x)))
        # print(x.size())
        result = nn.Softmax(dim = 1)(F.relu(self.bn4(self.conv4(x))))
        return result

def CNN_taining(trainloader,net, device, criterion, optimizer):
    len_dataloader = len(trainloader)
    best_losses = BEST_LOSS

    for epoch in range(100):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, (imgs, annotations) in enumerate(trainloader):
            # 4 imgs and 4 annotations:{boxes, label, image_id}
            #print()
            imgs = list(img.to(device) for img in imgs)
            imgs = torch.stack(imgs)
            #annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            gt = create_window_label(annotations).to(device)
            # image shape: torch.Size([3, 300, 300])
            optimizer.zero_grad()
            output = net(imgs)
            loss = criterion(output,gt)
            # avg_batch_loss = losses/BATCH_SIZE
            if loss < best_losses:
                best_losses = loss
                torch.save(net.state_dict(),
                            '../../data/data2/random_sample/CNNNet/checkpoints/model-epoch-{}-losses-{:.8f}.pth'.format(epoch + 1, loss))
                print('done saving')

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(i)
            if i % 100 == 99:
                print(f'Epoch: {epoch}, Batch: {i}/{len_dataloader}, loss: {running_loss/100}')
                running_loss = 0.0

    print('Finished Training')

