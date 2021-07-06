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

from dataset import *
from model import *
from loss import DiceLoss

from utils import *
from metric import eval
import math


class WGAN_De_Att:
    def __init__(self, netV=None):
        # initialize networks
        if netV is None:
             self.netV = init_model(VNet())
        else:
            self.netV = netV
        # initialize optimizers
        self.optimizer_V = optim.SGD(self.netV.parameters(), lr=1e-2, momentum=0.99, weight_decay=1e-8)
    
    def scheduler(self, epoch, lr0=0.01):
        if epoch>=80:
            lr = lr0*0.1
            for param_group in self.optimizer_V.param_groups:
                param_group['lr'] = lr
        if epoch>100:
            lr = lr0*0.01    
            for param_group in self.optimizer_V.param_groups:
                param_group['lr'] = lr
    
    def batch_train(self, cfg, dl_train, epoch_id):
        self.netV.train()
        criterion = DiceLoss()

        # for this epoch
        epoch_loss = np.zeros(cfg['cls_num'], dtype=np.float)
        epoch_loss_num = np.zeros(cfg['cls_num'], dtype=np.int64)

        for batch_id, batch in enumerate(dl_train):
            image = batch['data']
            label = batch['label']
            flag = batch['label_exist']
            n = len(image)

            image, label = image.cuda(), label.cuda()
            
            print_line = 'Epoch {0:d}/{1:d} (train) --- Progress {2:5.2f}% (+{3:d})'.format(
                    epoch_id+1, cfg['epoch_num'], 100.0 * batch_id * cfg['batch_size'] / len(d_train), n)
#             occupy_mem_allgpus()
            ### UNet loss
            '''---Dice loss---'''
            loss = 0
            cls_loss = np.zeros(cfg['cls_num'], dtype=np.float)
            pred = self.netV(image)
            for c in range(cfg['cls_num']):
                if torch.sum(flag[:,c]) > 0:
                    l = criterion(pred[:,c*2:c*2+2], label[:,c*2:c*2+2], flag[:,c])
                    loss += l
                    cls_loss[c] = l.item()
                    epoch_loss[c] += cls_loss[c]
                    epoch_loss_num[c] += 1
                else:
                    cls_loss[c] = 0

            print_line += ' -- Dice Loss: {0:.4f}'.format(loss.item())
            print(print_line)
            
            self.netV.zero_grad()
            loss.backward()
            self.optimizer_V.step()
            del image, label, pred, loss

        train_loss = np.sum(epoch_loss)/np.sum(epoch_loss_num)
        epoch_loss = epoch_loss / epoch_loss_num
        print_line = 'Epoch {0:d}/{1:d} (train) --- Loss: {2:.6f} ({3:s})\n'.format(
                epoch_id+1, cfg['epoch_num'], train_loss, '/'.join(['%.6f']*len(epoch_loss)) % tuple(epoch_loss))
        print(print_line)
        torch.cuda.empty_cache()
    
    
    def batch_eval(self, cfg, dl_val, epoch_id, loss_fn, eval_test=False):
        self.netV.eval()
        criterion = DiceLoss()
        output_mask = []

        for c in range(cfg['cls_num']):
            output_mask.append(None)
        for batch_id, batch in enumerate(dl_val):
            image = batch['data']
            label = batch['label']
            flag = batch['label_exist']
            n = len(image)
            image = image.cuda()
            label = label.cuda()
            pred = self.netV(image)
            
#             print('------------------\n')
#             for ii in range(n):
#                 print(torch.max(label[ii,1,:]), torch.min(label[ii,1,:]))

            print_line = 'Epoch {0:d}/{1:d} (val) --- Progress {2:5.2f}% (+{3:d})'.format(
                    epoch_id+1, cfg['epoch_num'], 100.0 * batch_id * cfg['batch_size'] / len(d_val), n)
            print(print_line)
#             occupy_mem_allgpus()
            
            for c in range(cfg['cls_num']):
                pred_bin = torch.argmax(pred[:,c*2:c*2+2], dim=1, keepdim=True)
                for i in range(n):
                    if flag[i, c] > 0:
                        mask = pred_bin[i,:].contiguous().cpu().numpy().copy().astype(dtype=np.uint8)
                        mask = np.squeeze(mask)
#                         print('-----------\n')
#                         print(np.max(mask), np.min(mask))
                        mask = resample_array(
                                mask, batch['size'][i].numpy(), batch['spacing'][i].numpy(), batch['origin'][i].numpy(), 
                                batch['org_size'][i].numpy(), batch['org_spacing'][i].numpy(), batch['org_origin'][i].numpy())
                        
                        if output_mask[c] is None:
                            output_mask[c] = mask
                        else:
                            output_mask[c] = output_mask[c] + mask

                        if batch['eof'][i]:
                            output_mask[c][output_mask[c] > 0] = 1
                            output2file(
                                output_mask[c], batch['org_size'][i].numpy(), batch['org_spacing'][i].numpy(), batch['org_origin'][i].numpy(), 
                                '{}/{}@{}@{}.nii.gz'.format(val_result_path, batch['dataset'][i], batch['case'][i], c+1))
                            output_mask[c] = None
                del pred_bin
            del image, pred
        
        if eval_test is True:
            dsc, asd, dsc_m, asd_m = eval(
                pd_path=val_result_path, gt_entries=test_fold, label_map=cfg['label_map'], cls_num=cfg['cls_num'], 
                metric_fn='metric_{0:04d}'.format(epoch_id), calc_asd=False)
        else:
            dsc, asd, dsc_m, asd_m = eval(
                pd_path=val_result_path, gt_entries=val_fold, label_map=cfg['label_map'], cls_num=cfg['cls_num'], 
                metric_fn='metric_{0:04d}'.format(epoch_id), calc_asd=False)

        print_line = 'Epoch {0:d}/{1:d} (val) --- DSC {2:.2f} ({3:s})% --- ASD {4:.2f} ({5:s})mm'.format(
            epoch_id+1, cfg['epoch_num'], 
            dsc_m*100.0, '/'.join(['%.2f']*len(dsc[:,0])) % tuple(dsc[:,0]*100.0), 
            asd_m, '/'.join(['%.2f']*len(asd[:,0])) % tuple(asd[:,0]))
        print(print_line)
        
        

    def Net_train(self, cfg, dl_train, dl_val, loss_fn, save_dir, if_eval=False):
        # check
#         self.batch_eval(cfg, dl_val, 0, loss_fn)
        
        for epoch_id in range(cfg['epoch_num']):
            t0 = time.perf_counter()
            self.scheduler(epoch_id)
            # train
            self.batch_train(cfg, dl_train, epoch_id)
            # evaluation
            if if_eval is True: self.batch_eval(cfg, dl_val, epoch_id, loss_fn)
            # save model
            file_name_V = 'V_epoch{}'.format(epoch_id)+'.pth'
            save_model(self.netV, save_dir, file_name_V)
            # eval
            if (epoch_id+1)%1==0:
                self.batch_eval(cfg, dl_val, epoch_id, loss_fn)
            


if __name__ == '__main__':
    cfg = {}
    cfg['cls_num'] = 1
    cfg['gpu'] = '7,5,1' # to use multiple gpu: cfg['gpu'] = '0,1,2,3'
    cfg['fold_num'] = 5
    cfg['epoch_num'] = 120
    cfg['batch_size'] = 6
    cfg['lr'] = 0.001
    cfg['Unet_path'] = ' '
    cfg['model_path'] = ' '
    cfg['rs_size'] = [256,256,32] # resample size: [x, y, z]
    cfg['rs_spacing'] = [1.5,1.5,3.0] # resample spacing: [x, y, z]. non-positive value means adaptive spacing fit the physical size: rs_size * rs_spacing = origin_size * origin_spacing
    cfg['rs_intensity'] = [-200.0, 200.0] # rescale intensity from [min, max] to [0, 1].
    cfg['cpu_thread'] = 8 # multi-thread for data loading. zero means single thread.

    # list of dataset names and paths
    cfg['data_path_train'] = [
        ['KiTS', ' ']
#         ['KiTS', '/home/zhangj41/RobDA_MICCAI/data/kits19/data_new2']
    ]
    cfg['data_path_test'] = [        
        ['BTCV', ' ']
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

'''--- train downstream tasks ---'''
Solver = WGAN_De_Att()
Solver.Net_train(cfg, 
                 dl_train, 
                 dl_val, 
                 loss_fn, 
                 store_dir)


'''--- eval downstream tasks ---'''
ii = 119
file_path = ' '
model_path = os.path.join(file_path,'V_epoch'+str(ii)+'.pth')
model = VNet()
model.cuda()
model.load_state_dict(torch.load(model_path))
netU = nn.DataParallel(module=model)
netU.eval()
Solver = WGAN_De_Att(netU)
Solver.batch_eval(cfg=cfg, 
                  dl_val=dl_test, 
                  epoch_id=0, 
                  loss_fn=loss_fn, 
                  eval_test=True)

