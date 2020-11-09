import torch.nn as nn
import torchvision.models as models


class Regressor(nn.Module):
    def __init__(self, input_size=128):
      super(Regressor, self).__init__()
      self.module7regressor = nn.Sequential(
        nn.Conv2d(1, 3, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),

        nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),

        nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),

        nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),

        nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),

        nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),

        nn.Conv2d(3, 2, kernel_size=3, stride=2, padding=1),
        nn.ReLU()
      )

    def forward(self, input):
        output = self.module7regressor(input)
        return output