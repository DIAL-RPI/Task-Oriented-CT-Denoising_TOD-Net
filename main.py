import sys
import os
import numpy as np
import random
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils import data

from dataset import scan_path, create_folds, Dataset, DatasetStk
# from dataset_apex import scan_path, create_folds, Dataset, DatasetStk
from model import UNet
from loss import DiceLoss

from utils import *
from metric import eval
import math

from TOD_Net import Generator, Discriminator # TOD nets
from tod_train import WGAN_De_Att
from tod_eval import WGAN_Eval
from model import UNet, VNet, VoxResNet, ResUNet, DenseUNet # Segmentation nets


'''--- configuration ---'''
if __name__ == '__main__':
    cfg = {}
    cfg['cls_num'] = 1
    cfg['gpu'] = '0' # to use multiple gpu: cfg['gpu'] = '0,1,2,3'
    cfg['fold_num'] = 5 # corss validation
    cfg['epoch_num'] = 100 # training epochs
    cfg['batch_size'] = 3 # batch size
    cfg['lr'] = 0.001 # learning rate
    cfg['Unet_path'] = 'unet path'
    cfg['model_path'] = 'tod-net path'
    cfg['rs_size'] = [256,256,32] # resample size: [x, y, z]
    cfg['rs_spacing'] = [1.5,1.5,3.0] # resample spacing: [x, y, z]. non-positive value means adaptive spacing fit the physical size: rs_size * rs_spacing = origin_size * origin_spacing
    cfg['rs_intensity'] = [-200.0, 200.0] # rescale intensity from [min, max] to [0, 1].
    cfg['cpu_thread'] = 8 # multi-thread for data loading. zero means single thread.

    # list of dataset names and paths
    cfg['data_path_train'] = [
        ['KiTS', '/data_kits']
#         ['KiTS', '/zion/data_new2']
    ]
    cfg['label_map'] = {
        'KiTS':{1:1},
        'BTCV':{2:1}
    }

    # exclude any samples in the form of '[dataset_name, case_name]'
    cfg['exclude_case'] = [
        #['KiTS', 'case_00133'],
        #['LiTS', 'volume-102']
    ]

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu']


'''--- prepare datasets ---'''
train_start_time = time.localtime()
time_stamp = time.strftime("%Y%m%d%H%M%S", train_start_time)
# acc_time = 0
    
# create directory for results storage
store_dir = '{}/model_{}'.format(cfg['model_path'], time_stamp)
os.makedirs(store_dir, exist_ok=True)

best_model_fn = '{}/epoch_{}.pth.tar'.format(store_dir, 1)
loss_fn = '{}/loss.txt'.format(store_dir)
log_fn = '{}/log.txt'.format(store_dir)

val_result_path = '{}/results_val'.format(store_dir)
os.makedirs(val_result_path, exist_ok=True)

test_result_path = '{}/results_test'.format(store_dir)
os.makedirs(test_result_path, exist_ok=True)

#Dataloader
folds, _ = create_folds(data_path=cfg['data_path_train'], fold_num=cfg['fold_num'], exclude_case=cfg['exclude_case'])

'''create training and validation fold'''
train_fold = []
for i in range(cfg['fold_num']-2):
    train_fold.extend(folds[i])

#d_train = Dataset(train_fold, rs_size=cfg['rs_size'], rs_spacing=cfg['rs_spacing'], rs_intensity=cfg['rs_intensity'], label_map=cfg['label_map'], cls_num=cfg['cls_num'])
d_train = DatasetStk(train_fold, rs_size=cfg['rs_size'], rs_spacing=cfg['rs_spacing'], rs_intensity=cfg['rs_intensity'], label_map=cfg['label_map'], cls_num=cfg['cls_num'], perturb=True)
dl_train = data.DataLoader(dataset=d_train, batch_size=cfg['batch_size'], shuffle=True, pin_memory=True, drop_last=True, num_workers=cfg['cpu_thread'])
    
# create validaion fold
val_fold = folds[cfg['fold_num']-2]
#d_val = Dataset(val_fold, rs_size=cfg['rs_size'], rs_spacing=cfg['rs_spacing'], rs_intensity=cfg['rs_intensity'], label_map=cfg['label_map'], cls_num=cfg['cls_num'])
d_val = DatasetStk(val_fold, rs_size=cfg['rs_size'], rs_spacing=cfg['rs_spacing'], rs_intensity=cfg['rs_intensity'], label_map=cfg['label_map'], cls_num=cfg['cls_num'], perturb=False)
dl_val = data.DataLoader(dataset=d_val, batch_size=cfg['batch_size'], shuffle=False, pin_memory=True, drop_last=False, num_workers=cfg['cpu_thread'])

# create test fold
test_fold = folds[cfg['fold_num']-1]
#d_val = Dataset(val_fold, rs_size=cfg['rs_size'], rs_spacing=cfg['rs_spacing'], rs_intensity=cfg['rs_intensity'], label_map=cfg['label_map'], cls_num=cfg['cls_num'])
d_test = DatasetStk(test_fold, rs_size=cfg['rs_size'], rs_spacing=cfg['rs_spacing'], rs_intensity=cfg['rs_intensity'], label_map=cfg['label_map'], cls_num=cfg['cls_num'], perturb=False)
dl_test = data.DataLoader(dataset=d_test, batch_size=cfg['batch_size'], shuffle=False, pin_memory=True, drop_last=False, num_workers=cfg['cpu_thread'])


################### training #######################
'''Load pretrained UNet'''
# model
model = UNet(in_ch=1, cls_num=cfg['cls_num'], base_ch=64)
model.cuda()
netU = nn.DataParallel(module=model)
netU.load_state_dict(torch.load(cfg['Unet_path'])['model_state_dict'])
netU.eval()

'''Load pretrained UNet'''
Solver = WGAN_De_Att(netU.module)
Solver.Net_train(cfg, 
                 dl_train, 
                 dl_val, 
                 loss_fn, 
                 store_dir, 
                 if_eval=True)

################### evaluation #######################
# model
model = UNet(in_ch=1, cls_num=cfg['cls_num'], base_ch=64)
model.cuda()
netU = nn.DataParallel(module=model)
netU.load_state_dict(torch.load(cfg['Unet_path'])['model_state_dict'])

ii_index = [] # fill in your model serial number
file_path = 'your model location'
for ii in ii_index:
    print('------------- model #',ii,'-------------\n')
    model_path = os.path.join(file_path,'G_epoch'+str(ii)+'.pth')
    model = Generator()
    model.cuda()
    model.load_state_dict(torch.load(model_path))
    gan = nn.DataParallel(model)
    gan.eval()
    Solver = WGAN_Eval(netU=netU.module, netG=gan)
    Solver.batch_eval(cfg=cfg, dl_val=dl_test, epoch_id=0, loss_fn=loss_fn, net_G=True, eval_test=True)