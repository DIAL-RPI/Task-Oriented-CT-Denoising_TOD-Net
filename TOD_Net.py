# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
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

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm3d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out



'''TOD generator'''
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.deConv1_1 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.deConv1 = nn.Sequential(
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.deConv2_1 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.deConv2 = nn.Sequential(
            nn.InstanceNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.deConv3_1 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.deConv3 = nn.Sequential(
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.deConv4_1 = nn.Conv3d(32, 1, kernel_size=3, padding=1)
        self.deConv4 = nn.Tanh() #nn.ReLU()

    def forward(self, input):
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        
        x = self.deConv1_1(conv4)
        x = x + conv3
        deConv1 = self.deConv1(x)
        x = self.deConv2_1(deConv1)
        x += conv2
        deConv2 = self.deConv2(x)
        x = self.deConv3_1(deConv2)
        x += conv1
        deConv3 = self.deConv3(x)
        x = self.deConv4_1(deConv3)
        x += input
        output = self.deConv4(x)
        return output



'''TOD discriminator'''
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.out_shape = 32
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool3d(2)
        )
        self.linear = nn.Sequential(
            torch.nn.Linear(128 * 4 * self.out_shape ** 2, 1),
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 128 * 4 * self.out_shape ** 2)
        x = self.linear(x)
        return x
