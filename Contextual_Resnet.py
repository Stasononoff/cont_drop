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

torch.pi = torch.acos(torch.zeros(1)).item() * 2


def prior_probs(sigma, S, type_ = 'normal'): # считает среднюю вероятность получения маски S для нормального распр с дисперсией sigma (параметр)
  if type_ == 'normal': 
    probs = 1/(torch.sqrt(2*torch.pi)*sigma)*torch.exp(-(1 - S)**2/(2*sigma**2))
  return torch.mean(probs)

def prob_estimation(tensors_list): # усредняем вероятности по всем слоям
  prob_tensor = torch.cat(tensors_list, axis = 0).mean()
  return prob_tensor


def calc_loss(Out_U,z,params, alpha): # части функции правдоподобия для оенки градиентов
  
  L_theta = torch.log(Out_U)
  L_ny = torch.log(prior_probs(params, z))
  L_fi = torch.log(Out_U) - torch.log(alpha)/torch.log(prior_probs(params, z))



def Sigma(X,t): # Сигмоида с весом t
  Y = 1/(1 + torch.exp(-t*X))
  return Y

class Encoderblock(nn.Module): # энкодер выхода слоя в маску для выхода

    def __init__(self, in_channels, H, gamma = 1):
        super(Encoderblock, self).__init__()

        # print(np.floor(in_channels*gamma), in_channels)

        self.avg1 = nn.AvgPool2d(kernel_size=(H, H))
        self.lin1 = nn.Linear(in_channels, int(np.floor(in_channels*gamma))) 
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear( int(np.floor(in_channels*gamma)), in_channels)

    def forward(self, x):
        alpha = self.lin2(self.relu(self.lin1(self.avg1(U))))
        return F.softmax(alpha, inplace=True)


class Context_Dropout(nn.Module): # Блок контекстного дропаута
    def __init__(self, in_channels, H, gamma = 1, t = 0.01):
        super(Context_Dropout, self).__init__()
        self.gamma = gamma
        self.t = t
        self.H = H

        self.enc_lay = Encoderblock(in_channels, H, gamma)

    def forward(self, U):
        alpha = self.enc_lay(U) 
        eps = torch.normal(0, 1, size=(self.H, self.H))
        tau = torch.sqrt((1 - Sigma(alpha, self.t))/Sigma(alpha, self.t))
        z = 1 + tau*eps
        x = U * z
        return x, z, alpha
        
        
        
class ResNet_Contextual_DropOut(nn.Module): # архитектура предобученной ResNet с наличием ContextualDropaut блоков
  def __init__(self, gamma =1, t = 0.1):
    super(ResNet_Contextual_DropOut, self).__init__()
    model_resnet18 = models.resnet18(pretrained=True)
    # for param in model_resnet18.parameters():
    #     param.requires_grad = False

    self.gamma = gamma
    self.t = t
    

    self.conv1 = model_resnet18.conv1
    self.bn1 = model_resnet18.bn1
    self.relu = model_resnet18.relu
    self.maxpool = model_resnet18.maxpool
    self.layer11 = model_resnet18.layer1[0]
    self.drop11 = Context_Dropout(in_channels = 64 , H = 56,  gamma = self.gamma, t = self.t)
    self.layer12 = model_resnet18.layer1[1]
    self.drop12 = Context_Dropout(in_channels = 64 , H = 56,  gamma = self.gamma, t = self.t)
    self.layer21 = model_resnet18.layer2[0]
    self.drop21 = Context_Dropout(in_channels = 128 , H = 28,  gamma = self.gamma, t = self.t)
    self.layer22 = model_resnet18.layer2[1]
    self.drop22 = Context_Dropout(in_channels = 128 , H = 28,  gamma = self.gamma, t = self.t)
    self.layer31 = model_resnet18.layer3[0]
    self.drop31 = Context_Dropout(in_channels = 256, H = 14,  gamma = self.gamma, t = self.t)
    self.layer32 = model_resnet18.layer3[1]
    self.drop32 = Context_Dropout(in_channels = 256 , H = 14,  gamma = self.gamma, t = self.t)
    self.layer41 = model_resnet18.layer4[0]
    self.drop41 = Context_Dropout(in_channels = 512 , H = 7,  gamma = self.gamma, t = self.t)
    self.layer42 = model_resnet18.layer4[1]
    # self.drop42 = Context_Dropout(in_channels = 512 , H = 7,  gamma = self.gamma, t = self.t)
    self.avgpool = model_resnet18.avgpool
    # self.__in_features = model_resnet18.fc.in_features
    self.fc = nn.Linear(512, 10)

  def forward(self, x):
    p_ny = []
    alpha_list = []
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.layer11(x)
    x,z,alpha = self.drop11(x)
    p_ny.append(prior_probs(sigma, z))
    alpha_list.append(alpha)
    x = self.layer12(x)
    x,z,alpha = self.drop12(x)
    p_ny.append(prior_probs(sigma, z))
    alpha_list.append(alpha)
    x = self.layer21(x)
    x,z,alpha = self.drop21(x)
    p_ny.append(prior_probs(sigma, z))
    alpha_list.append(alpha)
    x = self.layer22(x)
    x,z,alpha = self.drop22(x)
    p_ny.append(prior_probs(sigma, z))
    alpha_list.append(alpha)
    x = self.layer31(x)
    x,z,alpha = self.drop31(x)
    p_ny.append(prior_probs(sigma, z))
    alpha_list.append(alpha)
    x = self.layer32(x)
    x,z,alpha = self.drop32(x)
    p_ny.append(prior_probs(sigma, z))
    alpha_list.append(alpha)
    x = self.layer41(x)
    x,z,alpha = self.drop41(x)
    p_ny.append(prior_probs(sigma, z))
    alpha_list.append(alpha)
    x = self.layer42(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    ny_prob = torch.log(prob_estimation(p_ny))
    return x, ny_prob