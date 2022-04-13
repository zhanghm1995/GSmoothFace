'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-03-01 14:32:12
Email: haimingzhang@link.cuhk.edu.cn
Description: ResetNet embedding network
'''

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import resnet34, resnet18


class ResNetEmbedding(nn.Module):

    def __init__(self, arch='resnet34', pretrained=True, output_ch=128) -> None:
        super().__init__()

        if arch == "resnet34":
            net = resnet34(pretrained)
        elif arch == "resnet18":
            net = resnet18(pretrained)
        else:
            raise ValueError("Please choose the correct ResNet Arch. Options: [resnet34|resnet18]")
        
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        
        self.avgpool = net.avgpool

        self.out_fc = nn.Linear(512, output_ch)
    
    def forward(self, x: Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
    
        x = x.reshape(x.shape[0], -1) # TODO: view?
        
        x = self.out_fc(x)
        return x
