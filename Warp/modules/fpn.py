import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from Conv import Conv
from block import *

######################################## FPN start ########################################
class FPN(nn.Module):
    def __init__(self, fpn_sizes):
        super(FPN, self).__init__()
        
        P3_channels, P4_channels, P5_channels = fpn_sizes
        self.out_channels = 256
        
        self.p5_td_conv = nn.Conv2d(P5_channels, self.out_channels, kernel_size=1, stride=1)
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.p4_td_conv = nn.Conv2d(P4_channels, self.out_channels, kernel_size=1, stride=1)
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.p3_td_conv = nn.Conv2d(P3_channels, self.out_channels, kernel_size=1, stride=1)
        
        self.p3_out_conv = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.p4_out_conv = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.p5_out_conv = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, inputs):
        P3, P4, P5 = inputs
        
        # Top-down path
        P5_td = self.p5_td_conv(P5)
        P4_td = self.p4_td_conv(P4) + self.p5_upsample(P5_td)
        P3_td = self.p3_td_conv(P3) + self.p4_upsample(P4_td)
        
        # Output path
        P3_out = self.p3_out_conv(P3_td)
        P4_out = self.p4_out_conv(P4_td)
        
        return [P3_out, P4_out]
    
######################################## FPN end ########################################

######################################## C3k2-EIEM-Faster-FPN start ########################################
class FPN_C3k2_EIEM_Faster(nn.Module):
    def __init__(self, fpn_sizes):
        super(FPN_C3k2_EIEM_Faster, self).__init__()
        
        [P3_channels, P4_channels, P5_channels] = fpn_sizes
        self.out_channels = 256
        self.depth = [2, 2]

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
    
        self.p4_out_conv = C3k2_EIEM_Faster(P4_channels + P5_channels, self.out_channels, self.depth[0], True)
        self.p3_out_conv = C3k2_EIEM_Faster(P3_channels + self.out_channels, self.out_channels, self.depth[1], True)
        
        
    def forward(self, inputs):
        [P3, P4, P5] = inputs

        P4_td = torch.concat([P4, self.upsample(P5)], dim=1)
        P4_out = self.p4_out_conv(P4_td)

        P3_td = torch.concat([P3, self.upsample(P4_out)], dim=1)
        P3_out = self.p3_out_conv(P3_td)

        return [P3_out, P4_out]
    
######################################## C3k2-EIEM-Faster-FPN end ########################################

######################################## BiFPN start ########################################
class BiFPN(nn.Module):
    def __init__(self, fpn_sizes):
        super(BiFPN, self).__init__()
        
        P3_channels, P4_channels, P5_channels = fpn_sizes
        self.out_channels = 256
        
        # P5 to P4
        self.p5_td_conv = nn.Conv2d(P5_channels, self.out_channels, kernel_size=3, stride=1, bias=True, padding=1)
        self.p5_td_conv_2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, groups=self.out_channels, bias=True, padding=1)
        self.p5_td_act = nn.ReLU()
        self.p5_td_conv_bn = nn.BatchNorm2d(self.out_channels)
        self.p5_td_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p5_td_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        
        # P4 to P3
        self.p4_td_conv = nn.Conv2d(P4_channels, self.out_channels, kernel_size=3, stride=1, bias=True, padding=1)
        self.p4_td_conv_2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, groups=self.out_channels, bias=True, padding=1)
        self.p4_td_act = nn.ReLU()
        self.p4_td_conv_bn = nn.BatchNorm2d(self.out_channels)
        self.p4_td_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_td_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        
        # Upsample P5 to P4
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # P3 output
        self.p3_out_conv = nn.Conv2d(P3_channels, self.out_channels, kernel_size=3, stride=1, bias=True, padding=1)
        self.p3_out_conv_2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, groups=self.out_channels, bias=True, padding=1)
        self.p3_out_act = nn.ReLU()
        self.p3_out_conv_bn = nn.BatchNorm2d(self.out_channels)
        self.p3_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p3_out_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        
        # Upsample P4 to P3
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # P4 output
        self.p4_out_conv = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, groups=self.out_channels, bias=True, padding=1)
        self.p4_out_act = nn.ReLU()
        self.p4_out_conv_bn = nn.BatchNorm2d(self.out_channels)
        self.p4_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_out_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)

    def forward(self, inputs):
        P3, P4, P5 = inputs
        
        # Top-down path
        P5_td = self.p5_td_conv(P5)
        P5_td = self.p5_td_act(P5_td)
        P5_td = self.p5_td_conv_bn(P5_td)
        
        P4_td = self.p4_td_conv(P4) + self.p5_upsample(P5_td)
        P4_td = self.p4_td_act(P4_td)
        P4_td = self.p4_td_conv_bn(P4_td)
        
        # Bottom-up path
        P3_out = self.p3_out_conv(P3) + self.p4_upsample(P4_td)
        P3_out = self.p3_out_act(P3_out)
        P3_out = self.p3_out_conv_bn(P3_out)
        
        P4_out = self.p4_out_conv(P4_td)
        P4_out = self.p4_out_act(P4_out)
        P4_out = self.p4_out_conv_bn(P4_out)
        
        return P3_out, P4_out
######################################## BiFPN start ########################################


if __name__ == '__main__':
    fpn = FPN_C3k2_EIEM_Faster([40, 112, 192]).cuda()

    c3 = torch.randn([1, 40, 64, 64]).cuda()
    c4 = torch.randn([1, 112, 32, 32]).cuda()
    c5 = torch.randn([1, 192, 16, 16]).cuda()

    feats = [c3, c4, c5]
    output = fpn.forward(feats)
    for feat in output:
        print(feat.shape) 