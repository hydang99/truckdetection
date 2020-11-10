import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import sys
import random
from tqdm import tqdm_notebook
from skimage.io import imread, imshow
from skimage.transform import resize
#from sklearn.metrics import jaccard_similarity_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
from torch.autograd import Variable
import import_ipynb
from dataset import TruckDataset
#Show example for data testing
def show_sample(ids,train_path_images,train_path_masks):
    plt.figure(figsize=(20,10))
    for j, img_name in enumerate(ids):
        q = j+1
        img = imread(train_path_images + "/" + img_name + '.png')
        img_mask = imread(train_path_masks + "/" + img_name + '.png')

        plt.subplot(1,2*(1+len(ids)),q*2-1)
        plt.imshow(img)
        plt.subplot(1,2*(1+len(ids)),q*2)
        plt.imshow(img_mask)
    plt.show()
#Return ids of the dataset of training set
def list_ids(train_path_images):
    train_ids = next(os.walk(train_path_images))[2]
    return train_ids

# Get and resize train images and masks
#Return the set of X_train and Y_train

def train_ds(path_train,train_ids,im_height,im_width,im_chan=3):
    X_train = np.zeros((len(train_ids), im_height, im_width, im_chan), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), im_height, im_width, 1), dtype=np.bool_)
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()
    for n, id_ in tqdm_notebook(enumerate(train_ids), total=len(train_ids)):
        img = imread(path_train + '/images/' + id_)
        x = resize(img, (im_height, im_width, 1), mode='constant', preserve_range=True)
        X_train[n] = x
        mask = imread(path_train + '/masks/' + id_)
        Y_train[n] = resize(mask, (im_height, im_width, 1), 
                          mode='constant', 
                          preserve_range=True)

    print('Done!')
    return X_train, Y_train

#Normalize the image and reshape it to be one dimensional
def normalize_reshape(X_train,Y_train):
    X_train_shaped = X_train.reshape(-1, 1, 128, 128)/255
    Y_train_shaped = Y_train.reshape(-1, 1, 128, 128)
    X_train_shaped = X_train_shaped.astype(np.float32)
    Y_train_shaped = Y_train_shaped.astype(np.float32)
    return X_train_shaped,Y_train_shaped

#Shuffle the dataset into training and validation
def shuffle(X_train_shaped, val_size = 0.1):
    indices = list(range(len(X_train_shaped)))
    np.random.shuffle(indices)
    split = np.int_(np.floor(val_size * len(X_train_shaped)))
    train_idxs = indices[split:]
    val_idxs = indices[:split]
    return train_idxs, val_idxs 

from torchvision.utils import make_grid

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        break