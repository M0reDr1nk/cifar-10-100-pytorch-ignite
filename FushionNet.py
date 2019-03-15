from torch import nn
import torch.nn.functional as F
from blocks.DepthwiseConv import DepthwiseConv
from blocks.CReLu import CReLU


class FushionNet(nn.Module):
    def __init__(self):
        super(FushionNet, self).__init__()

        self.conv0_0 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 64, 1, 1, 0),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ELU()
        )
        self.conv0_1 = nn.Sequential(
            nn.Conv2d(64, 128, 1, 1, 0),
            nn.BatchNorm2d(128),
            nn.ELU(),  # ,
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 128, 3, 1, 1),#stride=2
            nn.BatchNorm2d(128),
            nn.ELU()
        )
        self.conv1_0 = nn.Sequential(
            DepthwiseConv(128, 1, 1, 1),
            DepthwiseConv(128, 1, 1, 1),
            DepthwiseConv(128, 2, 1, 1),
            DepthwiseConv(128, 2, 1, 1)
        )
        self.conv1_0_up = nn.Sequential(
            nn.Conv2d(128, 256, 1, 1, 0),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ELU()

        )
        self.conv1_1 = nn.Sequential(
            DepthwiseConv(256, 1, 1, 1),
            DepthwiseConv(256, 1, 1, 1),
            DepthwiseConv(256, 2, 1, 1),
            DepthwiseConv(256, 2, 1, 1),
            DepthwiseConv(256, 2, 1, 1),
            DepthwiseConv(256, 2, 1, 1),
            DepthwiseConv(256, 4, 1, 1),
            DepthwiseConv(256, 4, 1, 1),
            DepthwiseConv(256, 4, 1, 1),
            DepthwiseConv(256, 4, 1, 1)
        )
        self.conv1_1_up = nn.Sequential(
            nn.Conv2d(256, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.ELU()
        )
        self.conv1_2 = nn.Sequential(
            DepthwiseConv(512, 2, 1, 1),
            DepthwiseConv(512, 2, 1, 1)#S2
        )
        self.conv2_0 = nn.Sequential(
            nn.Conv2d(512, 1024, 1, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ELU(),
            nn.Conv2d(1024, 1024, 3, 1, 1, 1, 32),
            nn.BatchNorm2d(1024),
            nn.ELU(),
            nn.Conv2d(1024, 1024, 1, 1, 0),
            nn.BatchNorm2d(1024),
            nn.ELU()
        )
        self.drop_out = nn.Sequential(
            nn.Dropout(0.2)#0.4
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 10),
            nn.ELU(),
            nn.Linear(10, 10)
        )

        self.initialize_weight(self.conv0_0)
        self.initialize_weight(self.conv0_1)
        self.initialize_weight(self.conv1_0)
        self.initialize_weight(self.conv1_0_up)
        self.initialize_weight(self.conv1_1)
        self.initialize_weight(self.conv1_1_up)
        self.initialize_weight(self.conv1_2)
        self.initialize_weight(self.conv2_0)
        self.initialize_weight(self.fc)

    def forward(self, x):
        x = self.conv0_0(x)  # 24
        result_0 = self.conv0_1(x)  # 12
        x = self.conv1_0(result_0)+result_0
        x = self.conv1_0_up(x)
        x = self.conv1_1(x)+x
        x = self.conv1_1_up(x)
        result_1 = self.conv1_2(x)  # 6
        x=self.conv2_0(result_1)
        x = F.avg_pool2d(x, 8)
        x = self.drop_out(x)
        # x = F.max_pool2d(x, 2)
        x = x.view(-1, 1024)
        # x = self.drop_out(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

    def initialize_weight(self, sequential):
        for module in sequential:
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.001)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, DepthwiseConv):
                for submodule in module.sequential:
                    if isinstance(submodule, nn.Conv2d):
                        nn.init.xavier_normal_(submodule.weight)
                        nn.init.constant_(submodule.bias, 0)
                    elif isinstance(submodule, nn.BatchNorm2d):
                        nn.init.constant_(submodule.weight, 1)
                        nn.init.constant_(submodule.bias, 0)
                for submodule in module.shortcut:
                    if isinstance(submodule, nn.Conv2d):
                        nn.init.xavier_normal_(submodule.weight)
                        nn.init.constant_(submodule.bias, 0)
                    elif isinstance(submodule, nn.BatchNorm2d):
                        nn.init.constant_(submodule.weight, 1)
                        nn.init.constant_(submodule.bias, 0)
