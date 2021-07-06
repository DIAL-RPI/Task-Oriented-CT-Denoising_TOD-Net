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

from utils import resample_array, output2file
from metric import eval
import math

from TOD_Net import Generator, Discriminator

class WGAN_De_Att:
    def __init__(self, netU):
        # initialize networks
        self.netG = Generator().cuda()
        self.netD = Discriminator().cuda()
        self.netU = netU.cuda()
        
        self.netG = nn.DataParallel(self.netG)
        self.netD = nn.DataParallel(self.netD)
        self.netU = nn.DataParallel(self.netU)
        
        # initialize optimizers
        self.optimizer_G = torch.optim.RMSprop(self.netG.parameters(),lr=0.0005)
        self.optimizer_D = torch.optim.RMSprop(self.netD.parameters(),lr=0.0005)
        
    
    def scheduler(self, epoch, lr0=0.0005):
        if epoch>=49:
            lr = lr0*0.1
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = lr
            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = lr
        if epoch>79:
            lr = lr0*0.01    
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = lr
            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = lr
    
    def batch_train(self, cfg, dl_train, epoch_id):
        ###
        self.netU.eval()
        self.netG.train()
        self.netD.train()
        criterion = DiceLoss()

        # for this epoch
        epoch_loss = np.zeros(cfg['cls_num'], dtype=np.float)
        epoch_loss_num = np.zeros(cfg['cls_num'], dtype=np.int64)

        for batch_id, batch in enumerate(dl_train):
            image, image_ld, label, flag = batch['data'], batch['data_ld'], batch['label'], batch['label_exist']
            n = len(image)
            image, image_ld, label = image.cuda(), image_ld.cuda(), label.cuda()
            print_line = 'Epoch {0:d}/{1:d} (train) --- Progress {2:5.2f}% (+{3:d})'.format(
                    epoch_id+1, cfg['epoch_num'], 100.0 * batch_id * cfg['batch_size'] / len(d_train), n)

            ### WGAN loss
            '''---update netD---'''
            x_outputs = self.netD(image)
            image_enh = self.netG(image_ld).detach()
            z_outputs = self.netD(image_enh)
            D_x_loss = torch.mean(x_outputs)
            D_z_loss = torch.mean(z_outputs)
            D_loss = D_z_loss - D_x_loss
            
            self.optimizer_D.zero_grad()
            D_loss.backward()
            self.optimizer_D.step()
            for p in self.netD.parameters():
                p.data.clamp_(-0.01, 0.01)
            
            '''---update netG w D and mse---'''
            image_enh = self.netG(image_ld)
            mse_loss = F.mse_loss(image_enh, image)
            z_outputs = self.netD(image_enh)
            gan_loss = -torch.mean(z_outputs)
            print_line += ' -- MSE: {0:.4f}'.format(mse_loss.item())
            
            ### Task-driven loss
            '''---Dice loss---'''
            dice_loss = 0
            cls_loss = np.zeros(cfg['cls_num'], dtype=np.float)
            pred = self.netU(image_enh)
            for c in range(cfg['cls_num']):
                if torch.sum(flag[:,c]) > 0:
                    l = criterion(pred[:,c*2:c*2+2], label[:,c*2:c*2+2], flag[:,c])
                    dice_loss += l
                    cls_loss[c] = l.item()
                    epoch_loss[c] += cls_loss[c]
                    epoch_loss_num[c] += 1
                else:
                    cls_loss[c] = 0

            print_line += ' -- Dice Loss: {0:.4f}'.format(dice_loss.item())
            print(print_line)
            
            G_loss = gan_loss + mse_loss/2 #+ dice_loss
            self.optimizer_D.zero_grad()
            self.optimizer_G.zero_grad()
            G_loss.backward()
            self.optimizer_G.step()
    
            del image, image_ld, image_enh, label, pred, G_loss
            torch.cuda.empty_cache()

    def batch_eval(self, cfg, dl_val, epoch_id, loss_fn, net_G=False):
        self.netU.eval()
        if net_G is not False: self.netG.eval()
        criterion = DiceLoss()
        output_mask = []

        for c in range(cfg['cls_num']):
            output_mask.append(None)
        for batch_id, batch in enumerate(dl_val):
            image = batch['data']
            image_ld = batch['data_ld']
            flag = batch['label_exist']
            n = len(image)
            image = image.cuda()
            image_ld = image_ld.cuda()
            
            if net_G is False:
                pred = self.netU(image_ld)
            else:
                pred = self.netU(self.netG(image_ld))

            print_line = 'Epoch {0:d}/{1:d} (val) --- Progress {2:5.2f}% (+{3:d})'.format(
                    epoch_id+1, cfg['epoch_num'], 100.0 * batch_id * cfg['batch_size'] / len(d_val), n)
            print(print_line)
            
            for c in range(cfg['cls_num']):
                pred_bin = torch.argmax(pred[:,c*2:c*2+2], dim=1, keepdim=True)
                for i in range(n):
                    if flag[i, c] > 0:
                        mask = pred_bin[i,:].contiguous().cpu().numpy().copy().astype(dtype=np.uint8)
                        mask = np.squeeze(mask)
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
                torch.cuda.empty_cache()
            del image, image_ld, pred
            torch.cuda.empty_cache()

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
        self.batch_eval(cfg, dl_val, 0, loss_fn, False)
        
        for epoch_id in range(cfg['epoch_num']):
            t0 = time.perf_counter()
            self.scheduler(epoch_id)
            # train
            self.batch_train(cfg, dl_train, epoch_id)
            # evaluation
            if if_eval is True: self.batch_eval(cfg, dl_val, epoch_id, loss_fn, True)
            # save model
            file_name_G = 'G_epoch{}'.format(epoch_id)+'.pth'
            save_model(self.netG, save_dir, file_name_G)
            file_name_D = 'D_epoch{}'.format(epoch_id)+'.pth'
            save_model(self.netD, save_dir, file_name_D)
            