import torch.nn as nn
import torchvision.models as models


class Colorize(nn.Module):
    def __init__(self, input_size=128):
        super(Colorize, self).__init__()
        self.downsampling = nn.Sequential(
          nn.Conv2d(1, 3, kernel_size=3, stride=2, padding=1),
          nn.LeakyReLU(0.1),

          nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
          nn.LeakyReLU(0.1),

          nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
          nn.LeakyReLU(0.1),

          nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
          nn.LeakyReLU(0.1),

          nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
          nn.LeakyReLU(0.1)
       )

       
        self.upsampling = nn.Sequential(
          nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(3),
          nn.LeakyReLU(0.1),
          nn.Upsample(scale_factor=2),
          nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(3),
          nn.LeakyReLU(0.1),
          nn.Upsample(scale_factor=2),
          nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(3),
          nn.LeakyReLU(0.1),
          nn.Upsample(scale_factor=2),
          nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(3),
          nn.LeakyReLU(0.1),
          nn.Upsample(scale_factor=2),
          nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(2),
          nn.LeakyReLU(0.1),
          nn.Upsample(scale_factor=2),
          nn.LeakyReLU(0.1)
          # nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
          # nn.BatchNorm2d(3),
          # nn.LeakyReLU(0.1),
          # nn.Upsample(scale_factor=2),
          # nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
          # nn.BatchNorm2d(3),
          # nn.LeakyReLU(0.1),
          # nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1),
          # nn.Upsample(scale_factor=2)
        )

    def forward(self, input):
        # Pass input through ResNet-gray to extract features
        downsample = self.downsampling(input)
        out = self.upsampling(downsample)
        return out