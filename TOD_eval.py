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


class WGAN_Eval:
    def __init__(self, netU, netG):
        # initialize networks
        self.netG = netG.cuda()
        self.netU = netU.cuda()
#         self.netU = nn.DataParallel(self.netU)
    
    def batch_eval(self, cfg, dl_val, epoch_id, loss_fn, net_G=False, eval_test=True):
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
            