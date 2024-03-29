import torchvision.models as models
import torch.nn as nn


class Modified_ShufflenetV2(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.bw2col = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 10, 1, padding=0), nn.ReLU(),
            nn.Conv2d(10, 3, 1, padding=0), nn.ReLU())

        self.mv2 = models.shufflenet_v2_x0_5(pretrained=True) #pre-trained shufflenet_v2_x0_5 on ImageNet

        self.mv2.conv5 = nn.Sequential(
            nn.Conv2d(192, 512, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),) # change the output_channels from 1024 to 512

        self.mv2.fc = nn.Linear(512, num_classes) #change the fully connect layer

    def forward(self, x):
        x = self.bw2col(x)
        x = self.mv2(x)
        return x