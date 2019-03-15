import torch.nn as nn
import torch
import torch.nn.functional as F


class DepthwiseConv(nn.Module):

    def __init__(self, channels, inner_mul, down_sample, groups_div=1):
        self.down_sample = down_sample
        super(DepthwiseConv, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(channels, inner_mul * channels, 1, 1, 0),
            nn.BatchNorm2d(inner_mul * channels),
            nn.ELU(),
            nn.Conv2d(inner_mul * channels, inner_mul * channels, 3, down_sample, 1, groups=round(inner_mul * channels / groups_div), bias=True),
            nn.BatchNorm2d(inner_mul * channels),
            nn.ELU(),
            nn.Conv2d(channels * inner_mul, channels, 1, 1, 0, bias=True),
            nn.BatchNorm2d(channels)
        )
        self.shortcut = nn.Sequential()
        if down_sample != 1 :
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=down_sample,padding=1, bias=True),
                nn.BatchNorm2d(channels),
                nn.ELU()
            )


    def forward(self, input):
        x = self.sequential(input) + self.shortcut(input)
        return x
