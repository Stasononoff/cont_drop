import torch.nn.functional as F

import pickle
import numpy as np
from skimage import io

import torch.nn as nn
from torchvision import models


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F



class ResNet_DropOut(nn.Module):
  def __init__(self):
    super(ResNet_DropOut, self).__init__()
    model_resnet18 = models.resnet18(pretrained=True)
    # for param in model_resnet18.parameters():
    #     param.requires_grad = False

    self.conv1 = model_resnet18.conv1
    self.bn1 = model_resnet18.bn1
    self.relu = model_resnet18.relu
    self.maxpool = model_resnet18.maxpool
    self.layer11 = model_resnet18.layer1[0]
    self.drop11 = nn.Dropout2d(p=0.5)
    self.layer12 = model_resnet18.layer1[1]
    self.drop12 = nn.Dropout2d(p=0.5)
    self.layer21 = model_resnet18.layer2[0]
    self.drop21 = nn.Dropout2d(p=0.5)
    self.layer22 = model_resnet18.layer2[1]
    self.drop22 = nn.Dropout2d(p=0.5)
    self.layer31 = model_resnet18.layer3[0]
    self.drop31 = nn.Dropout2d(p=0.5)
    self.layer32 = model_resnet18.layer3[1]
    self.drop32 = nn.Dropout2d(p=0.5)
    self.layer41 = model_resnet18.layer4[0]
    self.drop41 = nn.Dropout2d(p=0.5)
    self.layer42 = model_resnet18.layer4[1]
    # self.drop42 = nn.Dropout2D((p=0.5))
    self.avgpool = model_resnet18.avgpool
    # self.__in_features = model_resnet18.fc.in_features
    self.fc = nn.Linear(512, 10)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer11(x)
    x = self.drop11(x)
    x = self.layer12(x)
    x = self.drop12(x)
    x = self.layer21(x)
    x = self.drop21(x)
    x = self.layer22(x)
    x = self.drop22(x)
    x = self.layer31(x)
    x = self.drop31(x)
    x = self.layer32(x)
    x = self.drop32(x)
    x = self.layer41(x)
    x = self.drop41(x)
    x = self.layer42(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x