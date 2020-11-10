import torch.nn as nn
import torchvision.models as models


class Colorize(nn.Module):
    def __init__(self):
        super(Colorize, self).__init__()
        self.downsample1 = nn.Sequential(
          nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
          nn.BatchNorm2d(64),
          nn.ReLU())
        
        self.downsample2 = nn.Sequential(
          nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
          nn.BatchNorm2d(128),
          nn.ReLU())

        self.downsample3 = nn.Sequential(
          nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
          nn.BatchNorm2d(128),
          nn.ReLU())
        
        self.downsample4 = nn.Sequential(
          nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
          nn.BatchNorm2d(256),
          nn.ReLU())
        
        self.downsample5 = nn.Sequential(
          nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
          nn.BatchNorm2d(256),
          nn.ReLU())

        self.downsample6 = nn.Sequential(
          nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
          nn.BatchNorm2d(512),
          nn.ReLU())

       
        self.upsampling1 = nn.Sequential(
          nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(256),
          nn.ReLU(),
          nn.UpsamplingNearest2d(scale_factor=2))
        
        self.upsampling2 = nn.Sequential(
          nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(128),
          nn.ReLU(),
          nn.UpsamplingNearest2d(scale_factor=2))
        
        self.upsampling3 = nn.Sequential(
          nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.UpsamplingNearest2d(scale_factor=2))
        
        self.upsampling4 = nn.Sequential(
          nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(64),
          nn.ReLU(),
          nn.UpsamplingNearest2d(scale_factor=2))
        
        self.upsampling5 = nn.Sequential(
          nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(32),
          nn.ReLU(),
          nn.UpsamplingNearest2d(scale_factor=2))

        self.upsampling6 = nn.Sequential(
          nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(2),
          nn.ReLU(),
          nn.UpsamplingNearest2d(scale_factor=2))

        self.tanh_upsampling6 = nn.Sequential(
          nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(2),
          nn.Tanh(),
          nn.UpsamplingNearest2d(scale_factor=2))

    def forward(self, input_,use_tanh):
        downsample = self.downsample1(input_)
        downsample = self.downsample2(downsample)
        downsample = self.downsample3(downsample)
        downsample = self.downsample4(downsample)
        downsample = self.downsample5(downsample)
        downsample = self.downsample6(downsample)

        upsample = self.upsampling1(downsample)
        upsample = self.upsampling2(upsample)
        upsample = self.upsampling3(upsample)
        upsample = self.upsampling4(upsample)
        upsample = self.upsampling5(upsample)
        if use_tanh:
          out = self.tanh_upsampling6(upsample)
        else:
          out = self.upsampling6(upsample)
        return out

# class Colorize(nn.Module):
#     def __init__(self):
#         super(Colorize, self).__init__()
#         self.downsampling = nn.Sequential(
#           nn.Conv2d(1, 3, kernel_size=3, stride=2, padding=1),
#           nn.BatchNorm2d(3),
#           nn.ReLU(),
          
#           nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
#           nn.BatchNorm2d(3),
#           nn.ReLU(),

#           nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
#           nn.BatchNorm2d(3),
#           nn.ReLU(),

#           nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
#           nn.BatchNorm2d(3),
#           nn.ReLU(),

#           nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
#           nn.BatchNorm2d(3),
#           nn.ReLU()
#        )

       
#         self.upsampling = nn.Sequential(
#           nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
#           nn.BatchNorm2d(3),
#           nn.ReLU(),
#           nn.UpsamplingNearest2d(scale_factor=2),

#           nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
#           nn.BatchNorm2d(3),
#           nn.ReLU(),
#           nn.UpsamplingNearest2d(scale_factor=2),

#           nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
#           nn.BatchNorm2d(3),
#           nn.ReLU(),
#           nn.UpsamplingNearest2d(scale_factor=2),

#           nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
#           nn.BatchNorm2d(3),
#           nn.ReLU(),
#           nn.UpsamplingNearest2d(scale_factor=2),

#           nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1),
#           nn.BatchNorm2d(2),
#           nn.Tanh(),
#           nn.UpsamplingNearest2d(scale_factor=2)
#         )

#     def forward(self, input, input_,use_tanh=None):
#         # Pass input through ResNet-gray to extract features
#         downsample = self.downsampling(input)
#         out = self.upsampling(downsample)
#         return out