import copy

import pandas as pd

import tarfile


import os
import urllib

import pickle
import numpy as np
from skimage import io

from tqdm import tqdm, tqdm_notebook
from PIL import Image
from pathlib import Path

from torchvision import transforms
from multiprocessing.pool import ThreadPool
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from matplotlib import colors, pyplot as plt

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F


from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)

from Train_process import *






def plot_res(loss, val_loss, acc):
  
    fig1, ax1 = plt.subplots(figsize=(15, 9))
    ax1.plot(loss, label="train_loss")
    ax1.plot(val_loss, label="val_loss")
    plt.legend(loc='best')
    ax1.set_xlabel("epochs")
    ax1.set_ylabel("loss")
    ax1.set_title("Loss")
    plt.show()


    fig2, ax2 = plt.subplots(figsize=(15, 9))
    ax2.plot(acc, label="accuracy")
    plt.legend(loc='best')
    ax2.set_xlabel("epochs")
    ax2.set_ylabel("accuracy")
    ax2.set_title("Accuracy score")
    plt.show()

    return 0



def predict_one_sample(model, inputs, device='gpu'):
    with torch.no_grad():
        inputs = inputs.to(device)
        model.eval()
        logit = model(inputs).cpu()
        probs = torch.nn.functional.softmax(logit, dim=-1).numpy()
    return probs
    
    
    
def calc_metrics(model, val_dataset, DEVICE = 'cpu', el_num = 400, cert_lev = 0.9):

    idxs = list(map(int, np.random.uniform(0,2000, el_num)))
    imgs = [val_dataset[id][0].unsqueeze(0) for id in idxs]

    probs_ims = predict(model, imgs, DEVICE = DEVICE)
    y_pred = np.argmax(probs_ims,-1)

    actual_labels = [val_dataset[id][1] for id in idxs]

    top1_acc = accuracy_score(actual_labels, y_pred, )
    f1_score_multi = f1_score(actual_labels, y_pred, average = None)

    top1_prob, top1_catid = torch.topk(torch.tensor(probs_ims), 1)


    acc_list = []
    cert_list = []

    for y_real, y_pred, prob_pred in zip(actual_labels, top1_catid, top1_prob):
        if y_pred == y_real:
            acc_list.append(1)
        else:
            acc_list.append(0)

        if prob_pred > cert_lev:
            cert_list.append(1)
        else:
            cert_list.append(0)

    acc = np.array(acc_list)
    cert = np.array(cert_list)

    PAvPU = (sum(acc*cert) + sum((1-acc)*(1-cert)))/(sum(acc*cert) + sum((1-acc)*(1-cert)) + sum((1-acc)*cert) + sum(acc*(1-cert)))

    print('Top1 acc: ' + str(top1_acc))
    print('Multi-lable F1_score: ' + str(f1_score_multi))
    print('PAvPU:' + str(PAvPU))

    return 'done'




import copy

def save_model(model, name = 'ResNet_no_dropout' ):
    model_weights = copy.deepcopy(model.state_dict())
    torch.save(model_weights, 'Models/'+name+'_weights.pth') 
    torch.save(model, 'Models/'+name+'.pth') 
    return 'saved'

def load_model(name = 'ResNet_no_dropout' ):
    model = torch.load('Models/'+name+'.pth') 
    return model