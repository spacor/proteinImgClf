
# coding: utf-8

# In[23]:


import os
import json
import time
import copy
from copy import deepcopy
from collections import defaultdict

import numpy as np
import math
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models

from skimage import io

import matplotlib.pyplot as plt
from matplotlib import patches, patheffects

import imgaug as ia
from imgaug import augmenters as iaa

from sklearn.model_selection import train_test_split

from tqdm import tqdm
from pprint import pprint


# In[24]:


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# In[25]:


from tensorboardX import SummaryWriter


# In[26]:


# from torchsummary import summary


# In[27]:


# base_path = r'../input'
base_path = r'input'
PATH_TRAIN_ANNO = os.path.join(base_path, 'train.csv')
PATH_TRAIN_IMG = os.path.join(base_path, 'train')


# In[28]:


os.listdir(PATH_TRAIN_IMG)[:10]


# In[29]:


NUM_CLASSES = 28
MAX_TAGS = 5
IMG_SIZE = 224
BATCH_SIZE = 16
VAL_SIZE =0.33
THRESHOLD = 0.5
SAMPLES = 1.0
NUM_WORKER = 8
USE_TENSORBOARD = True
# DEVICE = torch.device("cpu")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[30]:


if USE_TENSORBOARD is True:
    writer = SummaryWriter()


# In[60]:


def record_tb(phase, tag, value, global_step):
    if USE_TENSORBOARD is True:
        writer.add_scalar('{phase}/{tag}'.format(phase = phase, tag = tag), value, global_step)


# In[32]:


def get_transform_anno(annotation_path, img_path):
    df = pd.read_csv(annotation_path)
    annotations = []
    for i, row in df.iterrows():
        rcd_id = row['Id']
        rcd_cate =  [int(j) for j in row['Target'].split()]
        annotations.append((rcd_id, rcd_cate))
    return annotations


# In[33]:


#get annotations
annotations = get_transform_anno(PATH_TRAIN_ANNO, PATH_TRAIN_IMG)
sample_size = int(len(annotations) * SAMPLES)
print('sample size: {}'.format(sample_size))
annotations = annotations[:sample_size]
pprint(annotations[0])


# In[34]:


#find out max tags, which is 5
# MAX_TAGS = 0
# for i in annotations:
#     num_tags = len(i[1])
#     if num_tags > MAX_TAGS:
#         MAX_TAGS = num_tags
# print('max num of tags: {}'.format(MAX_TAGS))


# In[35]:


#Test augmentation
seq = iaa.Sequential([
    iaa.Scale({"height": 224, "width": 224}),
    iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Affine(
            rotate=(-20, 20),
        )
    ], random_order=True) # apply augmenters in random order
], random_order=False)


# In[36]:


#read raw data
ix = 26
tmp = annotations[ix]
tmp_id = tmp[0]
tmp_img_tags = tmp[1]

tmp_ch = []
channels = ['red', 'blue', 'yellow', 'green']
img_file_template = '{}_{}.png'
for c in channels:
    tmp_ch.append(io.imread(os.path.join(PATH_TRAIN_IMG, img_file_template.format(tmp_id, c))))
tmp_img = np.stack(tmp_ch)


# In[37]:


tmp_img.shape


# In[38]:


def show_img(im, figsize=None, ax=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax

def show_batch_img_per_channel(imgs):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ix, ax in enumerate(axes.flat):
        tmp_img = imgs[ix]
        ax = show_img(tmp_img, ax=ax)


# In[39]:


#display each channel before aug
# show_batch_img_per_channel(tmp_img)  


# In[40]:


seq_det = seq.to_deterministic()


# In[41]:


#augmentation
tmp_aug_img=tmp_img.transpose((1, 2, 0))
tmp_aug_img = seq_det.augment_images([tmp_aug_img.copy()])[0]
tmp_aug_img=tmp_aug_img.transpose((2, 1, 0))


# In[42]:


#display each channel after aug
# show_batch_img_per_channel(tmp_aug_img)  


# In[43]:


class ProteinDataset(Dataset):
    def __init__(self, img_meta, max_tags, img_path, transform = None):
        self.img_meta = img_meta
        self.transform = transform
        self.max_tags = max_tags
        self.channels = ['red', 'blue', 'yellow', 'green']
        self.dummy_value = 28 #for padding
        self.img_path = img_path
        
    def __len__(self):
        return len(self.img_meta)

    def __getitem__(self, idx):
        img_id, img_tags= self.img_meta[idx]
        ch = []
        img_file_template = '{}_{}.png'
        for c in channels:
            ch.append(io.imread(os.path.join(self.img_path, img_file_template.format(img_id, c))))
        img = np.stack(tmp_ch)

        #augmentation
        if bool(self.transform) is True:
            img = self.transform(img)
            
        #pad
        img_tags = np.pad(np.array(img_tags), pad_width = (0, self.max_tags), mode = 'constant', constant_values=(self.dummy_value,self.dummy_value))[:self.max_tags]
        
        #transform to tensor
        img = torch.from_numpy(img).float()
        img_tags = torch.from_numpy(img_tags)
        
        output = (img, img_tags)
        return output


# In[44]:


class ImgTfm:
    def __init__(self, aug_pipline = None):
        self.seq = aug_pipline
    
    def __call__(self, img):
        
        seq_det = seq.to_deterministic()
        
        #augmentation
        aug_img=img.copy().transpose((1, 2, 0))
        aug_img = seq_det.augment_images([aug_img])[0]
        aug_img=aug_img.transpose((2, 1, 0))
        
        #normalize
        aug_img=aug_img/255
        
        return aug_img


# In[45]:


def get_aug_pipline(img_size, mode = 'train'):
    if mode == 'train':
        seq = iaa.Sequential([
            iaa.Scale({"height": IMG_SIZE, "width": IMG_SIZE}),
            iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Affine(
                    rotate=(-20, 20),
                )
            ], random_order=True) # apply augmenters in random order
        ], random_order=False)
    else: #ie.val
        seq = iaa.Sequential([
            iaa.Scale({"height": IMG_SIZE, "width": IMG_SIZE}),
        ], random_order=False)
    return seq


# In[46]:


train_set, val_set = train_test_split(annotations, test_size=VAL_SIZE, random_state=42)

composed = {}
composed['train'] = transforms.Compose([ImgTfm(aug_pipline=get_aug_pipline(img_size=IMG_SIZE, mode = 'train'))])
composed['val'] = transforms.Compose([ImgTfm(aug_pipline=get_aug_pipline(img_size=IMG_SIZE, mode = 'val'))])

image_datasets = {'train': ProteinDataset(train_set, max_tags = MAX_TAGS, img_path = PATH_TRAIN_IMG, transform=composed['train']),
                 'val': ProteinDataset(val_set, max_tags = MAX_TAGS, img_path = PATH_TRAIN_IMG, transform=composed['val'])}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER, drop_last=True)
              for x in ['train', 'val']}


# In[47]:


dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print(dataset_sizes)


# In[48]:


#test dataset
ix = 10
tmp_img, tmp_tags  = image_datasets['train'][ix]


# In[49]:


#test dataloader
tmp_img, tmp_tags = next(iter(dataloaders['train']))
print('tmp_img shape: {}\ntmp_tags: shape {}'.format(tmp_img.shape, tmp_tags.shape))


# In[50]:


def inverse_transform(img_torch):
    """denormalize and inverse transform img"""
#     inv_normalize = transforms.Normalize(
#         mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
#         std=[1/0.229, 1/0.224, 1/0.255]
#     )
    tmp = deepcopy(img_torch)
#     inv_normalize(tmp)
    tmp = np.clip((tmp.numpy().transpose((1,2,0)) * 255), a_min=0, a_max=255).astype(np.int)
    return tmp


# In[51]:


def show_batch_img(imgs):
    fig, axes = plt.subplots(3, 4, figsize=(12, 8))
    for ix, ax in enumerate(axes.flat):
        tmp_img = imgs[ix][:3] #showing first 3 channel only
        tmp_img = inverse_transform(tmp_img)
        ax = show_img(tmp_img, ax=ax)


# In[52]:


# show_batch_img(tmp_img)


# In[53]:


class Flatten(nn.Module):
    def __init__(self): 
        super().__init__()
    def forward(self, x): 
        return x.view(x.size(0), -1)

class RnetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = self._prep_backbone()
        
    def _prep_backbone(self):     
        base_model = models.resnet34(pretrained=False)
        removed = list(base_model.children())[1:-2]
        backbone = nn.Sequential(*removed)
#         for param in backbone.parameters():
#             param.require_grad = False
        return backbone
    
    def forward(self, x):
        x = self.backbone(x)
        return x

class CustomHead(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.num_class = num_class
        
        self.flatten = Flatten()
        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(p=0.5)
        self.fc_2 = nn.Linear(512 * 7 * 7, 256)
        self.relu_2 = nn.ReLU()
        self.batchnorm_2 = nn.BatchNorm1d(256)
        self.dropout_2 = nn.Dropout(p=0.5)
        self.fc_3 = nn.Linear(256, self.num_class)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu_1(x)
        x = self.dropout_1(x)
        x = self.fc_2(x)
        x = self.relu_2(x)
        x = self.batchnorm_2(x)
        x = self.dropout_2(x)
        x = self.fc_3(x)
        return x

class CustomEntry(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    def forward(self, x):
        x = self.conv_1(x)
        return x
    
class CustomNet(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.custom_entry = CustomEntry()
        self.backbone = RnetBackbone()
        self.custom_head = CustomHead(num_class)
        
    def forward(self, x):
        x = self.custom_entry(x)
        x = self.backbone(x)
        x = self.custom_head(x)
        return x


# In[54]:


def k_hot_embedding(labels, num_classes):
    khot = torch.eye(num_classes)[labels.data.cpu()]
    khot = khot.sum(1).clamp(0,1)
    return khot
    
# def criterion(y_pred, y_true):
#     #prep y_true
#     y_true_khot = k_hot_embedding(y_true, num_classes = NUM_CLASSES + 1)
#     y_true_khot = y_true_khot[:, :-1] #last element is dummy
#     y_true_khot = y_true_khot.to(DEVICE)
    
#     #calculate loss
    
#     loss = F.binary_cross_entropy_with_logits(y_pred, y_true_khot)
#     return loss
class F1Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input, target):
        #f1 loss
        #prep y_true
        y_pred_raw = input
        y_true_raw = target
        y_true = k_hot_embedding(y_true_raw, num_classes = NUM_CLASSES + 1)
        y_true = y_true[:, :-1] #last element is dummy
        y_true = y_true.to(DEVICE)

        #prep y_pred
        y_pred = torch.tensor(data = (torch.sigmoid(y_pred_raw).ge(THRESHOLD)), dtype=torch.float, device=DEVICE, requires_grad=True)

        #calculate loss
        tp = (y_true * y_pred).sum(0).float()
        # tn = ((1-y_true) * (1-y_pred)).sum(0).float()
        fp = ((1-y_true) * y_pred).sum(0).float()
        fn = (y_true * (1-y_pred)).sum(0).float()

        p = tp / (tp + fp)
        r = tp / (tp + fn)

        f1 = 2*p*r / (p+r)
        f1[torch.isnan(f1)] = 0
        f1_loss = 1-f1.mean()
#         print(f1_loss)
        return f1_loss
    
# def criterion(y_pred_raw, y_true_raw):
#     #f1 loss
#     #prep y_true
#     y_true = k_hot_embedding(y_true_raw, num_classes = NUM_CLASSES + 1)
#     y_true = y_true[:, :-1] #last element is dummy
#     y_true = y_true.to(DEVICE)
    
#     #prep y_pred
#     y_pred = (torch.sigmoid(y_pred_raw) > THRESHOLD).float()

#     #calculate loss
#     tp = (y_true * y_pred).sum(0).float()
#     # tn = ((1-y_true) * (1-y_pred)).sum(0).float()
#     fp = ((1-y_true) * y_pred).sum(0).float()
#     fn = (y_true * (1-y_pred)).sum(0).float()

#     p = tp / (tp + fp)
#     r = tp / (tp + fn)

#     f1 = 2*p*r / (p+r)
#     f1[torch.isnan(f1)] = 0
#     f1_loss = 1-f1.mean()
#     print(f1_loss)
#     return f1_loss


# In[55]:


def prep_stats(y_pred, y_true):
    #prep y_true
    y_true_khot = k_hot_embedding(y_true, num_classes = NUM_CLASSES + 1)
    y_true_khot = y_true_khot[:, :-1].cpu().numpy().astype('uint8') #last element is dummy
    
    #prep y_pred khot
    y_pred_khot = (torch.sigmoid(y_pred) > THRESHOLD).cpu().numpy().astype('uint8')
    
    return y_pred_khot, y_true_khot


# In[56]:


def calc_stats(y_pred, y_true, stats = 'accurancy'):
    if stats == 'accuracy':
        stat_value = accuracy_score(y_true, y_pred)
    elif stats == 'precision':
        stat_value = precision_score(y_true, y_pred, average = 'macro')
    elif stats == 'recall':
        stat_value = recall_score(y_true, y_pred, average = 'macro')
    elif stats == 'f1':
        stat_value = f1_score(y_true, y_pred, average = 'macro')
    else:
        stat_value = 0
    return stat_value


# In[57]:


def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    steps = 0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            running_y_true = []
            running_y_pred = []
            
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, targets in dataloaders[phase]:
                inputs = inputs.to(DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    y_pred_khot, y_true_khot = prep_stats(outputs, targets)
                    
                    running_y_pred.append(y_pred_khot)
                    running_y_true.append(y_true_khot)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == categories.data)

            epoch_loss = running_loss / dataset_sizes[phase]
#             epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_acc = accuracy_score(np.vstack(running_y_true), np.vstack(running_y_pred))
            epoch_precision = precision_score(np.vstack(running_y_true), np.vstack(running_y_pred), average = 'macro')
            epoch_recall = recall_score(np.vstack(running_y_true), np.vstack(running_y_pred), average = 'macro')
            epoch_f1 = f1_score(np.vstack(running_y_true), np.vstack(running_y_pred), average = 'macro')
            record_tb(phase, 'loss', epoch_loss, epoch)
            record_tb(phase, 'accuracy', epoch_acc, epoch)
            record_tb(phase, 'precision', epoch_precision, epoch)
            record_tb(phase, 'recall', epoch_recall, epoch)
            record_tb(phase, 'f1', epoch_f1, epoch)
            
            print('{} Loss: {:.4f} Acc: {:.4f} Percision: {:.4f} Recall {:.4f} F1 {:.4f}'.format(
                phase, epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1))

            # deep copy the model
#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
#     print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
#     model.load_state_dict(best_model_wts)
    return model


# In[58]:


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = CustomNet(num_class=NUM_CLASSES)
model_ft = model_ft.to(DEVICE)

criterion = F1Loss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# In[61]:


model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=40)


# In[ ]:


# tmp_img, tmp_tags = next(iter(dataloaders['val']))


# In[ ]:


# tmp_y_pred = model_ft(tmp_img)
# tmp_y_pred.shape
# tmp_tags.shape

# y_pred_khot, y_true_khot = prep_stats(tmp_y_pred, tmp_tags)

# np.vstack([y_pred_khot, y_true_khot]).shape

# precision_score(y_true_khot, y_pred_khot, average = 'macro')


# In[ ]:


# tmp_img = tmp_img.to(DEVICE)
# tmp_y_pred = model_ft(tmp_img)

# y_pred_khot, y_true_khot = prep_stats(tmp_y_pred, tmp_tags)

# precision_score(y_true_khot, y_pred_khot, average='macro')

# recall_score(y_true_khot, y_pred_khot, average='macro')

# f1_score(y_true_khot, y_pred_khot, average='macro')

# accuracy_score(y_true_khot, y_pred_khot)

# y_true_khot[1]

# y_pred_khot[1]


# In[ ]:


# tmp_img, tmp_tags = next(iter(dataloaders['train']))
# tmp_img = tmp_img.to(DEVICE)
# tmp_tags = tmp_tags.to(DEVICE)

# tmp_y_pred = model_ft(tmp_img)

# y_pred = (torch.sigmoid(tmp_y_pred) > THRESHOLD).float()

# torch.sigmoid(tmp_y_pred).requires_grad

# torch.tensor(data = (torch.sigmoid(tmp_y_pred).ge(THRESHOLD)), dtype=torch.float, device=DEVICE, requires_grad=True)

