import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import torch.utils.data.sampler as sampler

from torch.autograd import Variable

import numpy as np
from torch.autograd import Variable

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y = self.conv(x)
        return y

# Encoding block in U-Net
class enc_block(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_prob=0.0):
        super(enc_block, self).__init__()
        self.conv = double_conv(in_ch, out_ch)
        self.down = nn.MaxPool3d(2)
        self.dropout_prob = dropout_prob
        if dropout_prob > 0:
            self.dropout = nn.Dropout3d(p=dropout_prob)

    def forward(self, x):
        y_conv = self.conv(x)
        y = self.down(y_conv)
        if self.dropout_prob > 0:
            y = self.dropout(y)
        return y, y_conv

# Decoding block in U-Net
class dec_block(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, dropout_prob=0.0):
        super(dec_block, self).__init__()
        self.conv = double_conv(in_ch, out_ch)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(out_ch, out_ch, 2, stride=2)
        self.dropout_prob = dropout_prob
        if dropout_prob > 0:
            self.dropout = nn.Dropout3d(p=dropout_prob)

    def forward(self, x):
        y_conv = self.conv(x)
        y = self.up(y_conv)
        if self.dropout_prob > 0:
            y = self.dropout(y)
        return y, y_conv

def concatenate(x1, x2):
    # input is CHW
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]
    x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                    diffY // 2, diffY - diffY//2))        
    y = torch.cat([x2, x1], dim=1)
    return y

class softmax(nn.Module):
    def __init__(self, cls_num):
        super(softmax, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.cls_num = cls_num

    def forward(self, x):
        y = torch.zeros_like(x)
        for i in range(self.cls_num):
            y[:,i*2:i*2+2] = self.softmax(x[:,i*2:i*2+2])
        return y

class fuse_conv(nn.Module):
    def __init__(self, ch):
        super(fuse_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch, ch, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        y = self.conv(x)
        return y

# U-Net (baseline)
class UNet(nn.Module):
    def __init__(self, in_ch, cls_num, base_ch=64):
        super(UNet, self).__init__()
        self.in_ch = in_ch
        self.cls_num = cls_num
        self.base_ch = base_ch

        self.enc1 = enc_block(in_ch, base_ch)
        self.enc2 = enc_block(base_ch, base_ch*2)
        self.enc3 = enc_block(base_ch*2, base_ch*4)
        self.enc4 = enc_block(base_ch*4, base_ch*8)

        self.dec1 = dec_block(base_ch*8, base_ch*8, bilinear=False)
        self.dec2 = dec_block(base_ch*8+base_ch*8, base_ch*4, bilinear=False)
        self.dec3 = dec_block(base_ch*4+base_ch*4, base_ch*2, bilinear=False)
        self.dec4 = dec_block(base_ch*2+base_ch*2, base_ch, bilinear=False)
        self.lastconv = double_conv(base_ch+base_ch, base_ch)

        self.outconv = nn.Conv3d(base_ch, cls_num*2, 1)
        self.softmax = softmax(cls_num)

    def forward(self, x):
        enc1, enc1_conv = self.enc1(x)
        enc2, enc2_conv = self.enc2(enc1)
        enc3, enc3_conv = self.enc3(enc2)
        enc4, enc4_conv = self.enc4(enc3)
        dec1, _ = self.dec1(enc4)
        dec2, _ = self.dec2(concatenate(dec1, enc4_conv))
        dec3, _ = self.dec3(concatenate(dec2, enc3_conv))
        dec4, _ = self.dec4(concatenate(dec3, enc2_conv))
        lastconv = self.lastconv(concatenate(dec4, enc1_conv))
        output = self.outconv(lastconv)
        output = self.softmax(output)

        return output

    def description(self):
        return 'U-Net with {0:d}-ch input for {1:d}-class segmentation (base channel = {2:d})'.format(self.in_ch, self.cls_num, self.base_ch)

# U-Net with three decoding paths sharing one encoding path
# each decoding path predicts a binary mask of one target organ
class UNet2(nn.Module):
    def __init__(self, in_ch, cls_num, base_ch=64):
        super(UNet2, self).__init__()
        self.in_ch = in_ch
        self.cls_num = cls_num
        self.base_ch = base_ch

        self.enc1 = enc_block(in_ch, base_ch)
        self.enc2 = enc_block(base_ch, base_ch*2)
        self.enc3 = enc_block(base_ch*2, base_ch*4)
        self.enc4 = enc_block(base_ch*4, base_ch*8)

        self.dec11 = dec_block(base_ch*8, base_ch*4, bilinear=False)
        self.dec12 = dec_block(base_ch*4+base_ch*8, base_ch*2, bilinear=False)
        self.dec13 = dec_block(base_ch*2+base_ch*4, base_ch, bilinear=False)
        self.dec14 = dec_block(base_ch+base_ch*2, base_ch//2, bilinear=False)
        self.lastconv1 = double_conv(base_ch//2+base_ch, base_ch//2)
        self.outconv1 = nn.Conv3d(base_ch//2, 2, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.dec21 = dec_block(base_ch*8, base_ch*4, bilinear=False)
        self.dec22 = dec_block(base_ch*4+base_ch*8, base_ch*2, bilinear=False)
        self.dec23 = dec_block(base_ch*2+base_ch*4, base_ch, bilinear=False)
        self.dec24 = dec_block(base_ch+base_ch*2, base_ch//2, bilinear=False)
        self.lastconv2 = double_conv(base_ch//2+base_ch, base_ch//2)
        self.outconv2 = nn.Conv3d(base_ch//2, 2, 1)
        self.softmax2 = nn.Softmax(dim=1)

        self.dec31 = dec_block(base_ch*8, base_ch*4, bilinear=False)
        self.dec32 = dec_block(base_ch*4+base_ch*8, base_ch*2, bilinear=False)
        self.dec33 = dec_block(base_ch*2+base_ch*4, base_ch, bilinear=False)
        self.dec34 = dec_block(base_ch+base_ch*2, base_ch//2, bilinear=False)
        self.lastconv3 = double_conv(base_ch//2+base_ch, base_ch//2)
        self.outconv3 = nn.Conv3d(base_ch//2, 2, 1)
        self.softmax3 = nn.Softmax(dim=1)

    def forward(self, x):
        enc1, enc1_conv = self.enc1(x)
        enc2, enc2_conv = self.enc2(enc1)
        enc3, enc3_conv = self.enc3(enc2)
        enc4, enc4_conv = self.enc4(enc3)

        dec11, _ = self.dec11(enc4)
        dec12, _ = self.dec12(concatenate(dec11, enc4_conv))
        dec13, _ = self.dec13(concatenate(dec12, enc3_conv))
        dec14, _ = self.dec14(concatenate(dec13, enc2_conv))
        lastconv1 = self.lastconv1(concatenate(dec14, enc1_conv))
        output1 = self.outconv1(lastconv1)
        output1 = self.softmax1(output1)

        dec21, _ = self.dec21(enc4)
        dec22, _ = self.dec22(concatenate(dec21, enc4_conv))
        dec23, _ = self.dec23(concatenate(dec22, enc3_conv))
        dec24, _ = self.dec24(concatenate(dec23, enc2_conv))
        lastconv2 = self.lastconv2(concatenate(dec24, enc1_conv))
        output2 = self.outconv2(lastconv2)
        output2 = self.softmax2(output2)

        dec31, _ = self.dec31(enc4)
        dec32, _ = self.dec32(concatenate(dec31, enc4_conv))
        dec33, _ = self.dec33(concatenate(dec32, enc3_conv))
        dec34, _ = self.dec34(concatenate(dec33, enc2_conv))
        lastconv3 = self.lastconv3(concatenate(dec34, enc1_conv))
        output3 = self.outconv3(lastconv3)
        output3 = self.softmax3(output3)

        output = torch.cat([output1, output2, output3], dim=1)

        return output

    def description(self):
        return 'U-Net with three decoding paths sharing one encoding path [{0:d}-ch input][{1:d}-class seg][base ch={2:d}]'.format(self.in_ch, self.cls_num, self.base_ch)

# model 'UNet2' with intermediate 1x1 conv layers re-weighting features maps from different organ branches (decoding path)
class UNet2_ch1x1(nn.Module):
    def __init__(self, in_ch, cls_num, base_ch=64):
        super(UNet2_ch1x1, self).__init__()
        self.in_ch = in_ch
        self.cls_num = cls_num
        self.base_ch = base_ch

        self.enc1 = enc_block(in_ch, base_ch)
        self.enc2 = enc_block(base_ch, base_ch*2)
        self.enc3 = enc_block(base_ch*2, base_ch*4)
        self.enc4 = enc_block(base_ch*4, base_ch*8)

        self.dec11 = dec_block(base_ch*8, base_ch*4, bilinear=False)
        self.dec12 = dec_block(base_ch*4+base_ch*8, base_ch*2, bilinear=False)
        self.dec13 = dec_block(base_ch*2+base_ch*4, base_ch, bilinear=False)
        self.dec14 = dec_block(base_ch+base_ch*2, base_ch//2, bilinear=False)

        self.dec21 = dec_block(base_ch*8, base_ch*4, bilinear=False)
        self.dec22 = dec_block(base_ch*4+base_ch*8, base_ch*2, bilinear=False)
        self.dec23 = dec_block(base_ch*2+base_ch*4, base_ch, bilinear=False)
        self.dec24 = dec_block(base_ch+base_ch*2, base_ch//2, bilinear=False)

        self.dec31 = dec_block(base_ch*8, base_ch*4, bilinear=False)
        self.dec32 = dec_block(base_ch*4+base_ch*8, base_ch*2, bilinear=False)
        self.dec33 = dec_block(base_ch*2+base_ch*4, base_ch, bilinear=False)
        self.dec34 = dec_block(base_ch+base_ch*2, base_ch//2, bilinear=False)

        self.fuse1 = fuse_conv(base_ch*4*3)
        self.fuse2 = fuse_conv(base_ch*2*3)
        self.fuse3 = fuse_conv(base_ch*3)
        self.fuse4 = fuse_conv((base_ch//2)*3)

        self.lastconv1 = double_conv(base_ch//2+base_ch, base_ch//2)
        self.outconv1 = nn.Conv3d(base_ch//2, 2, 1)
        self.softmax1 = nn.Softmax(dim=1)

        self.lastconv2 = double_conv(base_ch//2+base_ch, base_ch//2)
        self.outconv2 = nn.Conv3d(base_ch//2, 2, 1)
        self.softmax2 = nn.Softmax(dim=1)

        self.lastconv3 = double_conv(base_ch//2+base_ch, base_ch//2)
        self.outconv3 = nn.Conv3d(base_ch//2, 2, 1)
        self.softmax3 = nn.Softmax(dim=1)

    def forward(self, x):
        enc1, enc1_conv = self.enc1(x)
        enc2, enc2_conv = self.enc2(enc1)
        enc3, enc3_conv = self.enc3(enc2)
        enc4, enc4_conv = self.enc4(enc3)

        dec11, _ = self.dec11(enc4)
        dec21, _ = self.dec21(enc4)
        dec31, _ = self.dec31(enc4)

        f1 = self.fuse1(torch.cat([dec11, dec21, dec31], dim=1))
        [dec11_, dec21_, dec31_] = torch.split(f1, self.base_ch*4, dim=1)

        dec12, _ = self.dec12(concatenate(dec11_, enc4_conv))
        dec22, _ = self.dec22(concatenate(dec21_, enc4_conv))
        dec32, _ = self.dec32(concatenate(dec31_, enc4_conv))

        f2 = self.fuse2(torch.cat([dec12, dec22, dec32], dim=1))
        [dec12_, dec22_, dec32_] = torch.split(f2, self.base_ch*2, dim=1)

        dec13, _ = self.dec13(concatenate(dec12_, enc3_conv))
        dec23, _ = self.dec23(concatenate(dec22_, enc3_conv))
        dec33, _ = self.dec33(concatenate(dec32_, enc3_conv))

        f3 = self.fuse3(torch.cat([dec13, dec23, dec33], dim=1))
        [dec13_, dec23_, dec33_] = torch.split(f3, self.base_ch, dim=1)
        
        dec14, _ = self.dec14(concatenate(dec13_, enc2_conv))
        dec24, _ = self.dec24(concatenate(dec23_, enc2_conv))
        dec34, _ = self.dec34(concatenate(dec33_, enc2_conv))

        f4 = self.fuse4(torch.cat([dec14, dec24, dec34], dim=1))
        [dec14_, dec24_, dec34_] = torch.split(f4, self.base_ch//2, dim=1)
        
        lastconv1 = self.lastconv1(concatenate(dec14_, enc1_conv))
        output1 = self.outconv1(lastconv1)
        output1 = self.softmax1(output1)

        lastconv2 = self.lastconv2(concatenate(dec24_, enc1_conv))
        output2 = self.outconv2(lastconv2)
        output2 = self.softmax2(output2)

        lastconv3 = self.lastconv3(concatenate(dec34_, enc1_conv))
        output3 = self.outconv3(lastconv3)
        output3 = self.softmax3(output3)

        output = torch.cat([output1, output2, output3], dim=1)

        return output

    def description(self):
        return '<UNet2> with intermediate 1x1 conv layers re-weighting features maps from different organ branches (decoding path) [{0:d}-ch input][{1:d}-class seg][base ch={2:d}]'.format(self.in_ch, self.cls_num, self.base_ch)


#################################### TOD Net  #####################################
# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
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

'''WGAN discriminator'''
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
        ############################################################
        self.linear = nn.Sequential(
            torch.nn.Linear(128 * 4 * self.out_shape ** 2, 1),
        )


    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 128 * 4 * self.out_shape ** 2)
        x = self.linear(x)
        return x



#################################### WGAN  #####################################
'''WGAN generator'''
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.InstanceNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # torch.nn.init.
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



'''WGAN discriminator'''
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
        ############################################################
        self.linear = nn.Sequential(
            torch.nn.Linear(128 * 4 * self.out_shape ** 2, 1),
#             nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Flatten and apply sigmoid
        x = x.view(-1, 128 * 4 * self.out_shape ** 2)
        x = self.linear(x)
        return x


    
    
#################################### VoxResNet  #####################################
import torch
import torch.nn as nn

class softmax(nn.Module):
    def __init__(self, cls_num):
        super(softmax, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.cls_num = cls_num

    def forward(self, x):
        y = torch.zeros_like(x)
        for i in range(self.cls_num):
            y[:,i*2:i*2+2] = self.softmax(x[:,i*2:i*2+2])
        return y

class VoxRes(nn.Module):
    def __init__(self, in_channel):
        super(VoxRes, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm3d(in_channel), 
            nn.ReLU(),
            nn.Conv3d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.BatchNorm3d(in_channel), 
            nn.ReLU(),
            nn.Conv3d(in_channel, in_channel, kernel_size=3, padding=1)
            )

    def forward(self, x):
        return self.block(x)+x
    
    
class VoxResNet(nn.Module):
    def __init__(self, in_channels, num_class):
        super(VoxResNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32), 
            nn.ReLU(),
            nn.Conv3d(32, 32, kernel_size=3, padding=1)
            )
        
        self.conv2 = nn.Sequential(
            nn.BatchNorm3d(32), 
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=(1, 2, 2), padding=1),
            VoxRes(64),
            VoxRes(64)
            )

        self.conv3 = nn.Sequential(
            nn.BatchNorm3d(64), 
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=(1, 2, 2), padding=1),
            VoxRes(64),
            VoxRes(64)
            )
        
        self.conv4 = nn.Sequential(
            nn.BatchNorm3d(64), 
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=(1, 2, 2), padding=1),
            VoxRes(64),
            VoxRes(64)
            )
        
        self.deconv_c1 = nn.Sequential(
            nn.ConvTranspose3d(32, 32, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Conv3d(32, num_class*2, kernel_size=1))
        
        self.deconv_c2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(64, num_class*2, kernel_size=1))
        
        self.deconv_c3 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=(1, 4, 4), stride=(1, 4, 4)),
            nn.Conv3d(64, num_class*2, kernel_size=1))
        
        self.deconv_c4 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=(1, 8, 8), stride=(1, 8, 8)),
            nn.Conv3d(64, num_class*2, kernel_size=1))
        
#         self.outconv = nn.Conv3d(num_class, num_class, 1)
        self.softmax = softmax(num_class)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)     
        out4 = self.conv4(out3)
        
        c1 = self.deconv_c1(out1)
        c2 = self.deconv_c2(out2)
        c3 = self.deconv_c3(out3)
        c4 = self.deconv_c4(out4)
        
        return self.softmax(c1+c2+c3+c4)


    
    
#################################### VNet  #####################################
class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=True, nll=False):
        super(VNet, self).__init__()
        self.in_tr = InputTransition(16, elu)
        self.down_tr32 = DownTransition(16, 1, elu)
        self.down_tr64 = DownTransition(32, 2, elu)
        self.down_tr128 = DownTransition(64, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(128, 2, elu, dropout=True)
        self.up_tr256 = UpTransition(256, 256, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(256, 128, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(128, 64, 1, elu)
        self.up_tr32 = UpTransition(64, 32, 1, elu)
        self.out_tr = OutputTransition(32, elu)

    def forward(self, x):
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out


    
    
############ ResUNet  #############
class ResUNet(nn.Module):
    """
    共9498260个可训练的参数, 接近九百五十万
    """
    def __init__(self, cls_num):
        super().__init__()

        self.encoder_stage1 = nn.Sequential(
            nn.Conv3d(1, 16, 3, 1, padding=1),
            nn.PReLU(16),
            nn.Conv3d(16, 16, 3, 1, padding=1),
            nn.PReLU(16),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
            nn.Conv3d(64, 64, 3, 1, padding=2, dilation=2),
            nn.PReLU(64),
            nn.Conv3d(64, 64, 3, 1, padding=4, dilation=4),
            nn.PReLU(64),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, padding=3, dilation=3),
            nn.PReLU(128),
            nn.Conv3d(128, 128, 3, 1, padding=4, dilation=4),
            nn.PReLU(128),
            nn.Conv3d(128, 128, 3, 1, padding=5, dilation=5),
            nn.PReLU(128),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256),
            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),
            nn.Conv3d(256, 256, 3, 1, padding=1),
            nn.PReLU(256),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, 3, 1, padding=1),
            nn.PReLU(128),
            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
            nn.Conv3d(128, 128, 3, 1, padding=1),
            nn.PReLU(128),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv3d(64 + 32, 64, 3, 1, padding=1),
            nn.PReLU(64),
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
            nn.Conv3d(64, 64, 3, 1, padding=1),
            nn.PReLU(64),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv3d(32 + 16, 32, 3, 1, padding=1),
            nn.PReLU(32),
            nn.Conv3d(32, 32, 3, 1, padding=1),
            nn.PReLU(32),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv3d(16, 32, 2, 2),
            nn.PReLU(32)
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv3d(32, 64, 2, 2),
            nn.PReLU(64)
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv3d(64, 128, 2, 2),
            nn.PReLU(128)
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 1, padding=1),
            nn.PReLU(256)
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, 2),
            nn.PReLU(128)
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, 2),
            nn.PReLU(64)
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, 2),
            nn.PReLU(32)
        )

        # 最后大尺度下的映射（256*256），下面的尺度依次递减
        self.map4 = nn.Sequential(
            nn.Conv3d(32, 2, 1, 1),
#             nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear'),
#             nn.Sigmoid()
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(64, 2, 1, 1),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear'),
#             nn.Sigmoid()
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(128, 2, 1, 1),
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear'),
#             nn.Sigmoid()
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, 2, 1, 1),
            nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear'),
#             nn.Sigmoid()
        )
        self.softmax = softmax(cls_num)

    def forward(self, inputs, training=True):
        long_range1 = self.encoder_stage1(inputs) + inputs

        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2(short_range1) + short_range1
        long_range2 = F.dropout(long_range2, 0.3, training)

        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3(short_range2) + short_range2
        long_range3 = F.dropout(long_range3, 0.3, training)

        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.encoder_stage4(short_range3) + short_range3
        long_range4 = F.dropout(long_range4, 0.3, training)

        short_range4 = self.down_conv4(long_range4)

        outputs = self.decoder_stage1(long_range4) + short_range4
        outputs = F.dropout(outputs, 0.3, training)

        output1 = self.softmax(self.map1(outputs))

        short_range6 = self.up_conv2(outputs)

        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
        outputs = F.dropout(outputs, 0.3, training)

        output2 = self.softmax(self.map2(outputs))

        short_range7 = self.up_conv3(outputs)

        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        outputs = F.dropout(outputs, 0.3, training)

        output3 = self.softmax(self.map3(outputs))

        short_range8 = self.up_conv4(outputs)

        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8

        output4 = self.softmax(self.map4(outputs))

        if training is True:
            return output1, output2, output3, output4
        else:
            return output4




############ DesnseResUNet  #############
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)
        
class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))

class up_in(nn.Sequential):
    def __init__(self, num_input_features1, num_input_features2, num_output_features):
        super(up_in, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.add_module('conv1_1', nn.Conv3d(num_input_features1, num_input_features2,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('conv3_3', nn.Conv3d(num_input_features2, num_output_features,
                        kernel_size=3, stride=1, padding=1, bias=False))
        self.add_module('norm', nn.BatchNorm3d(num_output_features))
        self.add_module('relu', nn.ReLU(inplace=True))
    def forward(self, x,y):
        y = self.up(y)
        x = self.conv1_1(x)
        z = self.conv3_3(x+y)
        z = self.norm(z)
        z = self.relu(z)
        return z

class upblock(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(upblock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.add_module('conv3_3', nn.Conv3d(num_input_features, num_output_features,
                        kernel_size=3, stride=1, padding=1, bias=False))
        self.add_module('norm', nn.BatchNorm3d(num_output_features))
        self.add_module('relu', nn.ReLU(inplace=True))
    def forward(self, x,y):
        y = self.up(y)
        z = self.conv3_3(x+y)
        z = self.norm(z)
        z = self.relu(z)
        return z

class up_out(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(up_out, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.add_module('conv3_3', nn.Conv3d(num_input_features, num_output_features,
                        kernel_size=3, stride=1, padding=1, bias=False))
        self.dropout = nn.Dropout3d(p=0.3)
        self.add_module('norm', nn.BatchNorm3d(num_output_features))
        self.add_module('relu', nn.ReLU(inplace=True))
    def forward(self, y):
        y = self.up(y)
        y = self.conv3_3(y)
        y = self.dropout(y)
        y = self.norm(y)
        y = self.relu(y)
        return y


class DenseUNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=48, block_config=(6, 12, 36, 24),
                 num_init_features=96, bn_size=6, drop_rate=0, num_channels=1, num_classes=2):

        super(DenseUNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(num_channels, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.up1 = up_in(48*44, 48*46, 48*16)
        self.up2 = upblock(48*16, 48*8)
        self.up3 = upblock(48*8, 96)
        self.up4 = upblock(96,96)
        self.up5 = up_out(96,64)
        self.outconv = outconv(64,num_classes)
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features.conv0(x)
        x0 = self.features.norm0(features)
        x0 = self.features.relu0(x0)
        x1 = self.features.pool0(x0)
        x1 = self.features.denseblock1(x1)
        x2 = self.features.transition1(x1)
        x2 = self.features.denseblock2(x2)
        x3 = self.features.transition2(x2)
        x3 = self.features.denseblock3(x3)
        x4 = self.features.transition3(x3)
        x4 = self.features.denseblock4(x4)
        
        y4 = self.up1(x3, x4)
        y3 = self.up2(x2, y4)
        y2 = self.up3(x1, y3)
        
        y1 = self.up4(x0, y2)
        y0 = self.up5(y1)
        out = self.outconv(y0)
        # out = F.softmax(out, dim=1)
        return out