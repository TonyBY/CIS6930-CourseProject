import torch.nn as nn
import torchvision.models as models


class Colorize(nn.Module):
    def __init__(self):
        super(Colorize, self).__init__()
        self.downsampling = nn.Sequential(
          nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
          nn.BatchNorm2d(64),
          nn.ReLU(),

          nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
          nn.BatchNorm2d(128),
          nn.ReLU(),

          nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
          nn.BatchNorm2d(128),
          nn.ReLU(),

          nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
          nn.BatchNorm2d(256),
          nn.ReLU(),

          nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
          nn.BatchNorm2d(256),
          nn.ReLU(),

          nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
          nn.BatchNorm2d(512),
          nn.ReLU()
       )

       
        self.upsampling = nn.Sequential(
          # nn.ConvTranspose2d(3, 3,kernel_size=4, stride=2, padding=1),
          # nn.BatchNorm2d(3),
          # nn.ReLU(True),
          # nn.ConvTranspose2d(3, 3,kernel_size=4, stride=2, padding=1),
          # nn.BatchNorm2d(3),
          # nn.ReLU(True),
          # nn.ConvTranspose2d(3, 3,kernel_size=4, stride=2, padding=1),
          # nn.BatchNorm2d(3),
          # nn.ReLU(True),
          # nn.ConvTranspose2d(3, 3,kernel_size=4, stride=2, padding=1),
          # nn.BatchNorm2d(3),
          # nn.ReLU(True),
          # nn.ConvTranspose2d(3, 2,kernel_size=4, stride=2, padding=1),
          # nn.BatchNorm2d(2),
          # nn.ReLU(True)
          nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(256),
          nn.ReLU(),
          nn.UpsamplingNearest2d(scale_factor=2),

          nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(128),
          nn.ReLU(),
          nn.UpsamplingNearest2d(scale_factor=2),

          nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.UpsamplingNearest2d(scale_factor=2),

          nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.UpsamplingNearest2d(scale_factor=2),

          nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(32),
          nn.ReLU(),
          nn.UpsamplingNearest2d(scale_factor=2),
         

          nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(2),
          nn.ReLU(),
          nn.UpsamplingNearest2d(scale_factor=2)
        )

    def forward(self, input):
        downsample = self.downsampling(input)
        out = self.upsampling(downsample)
        return out