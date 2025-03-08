import torch.nn as nn
import torch
import math

import torch
import torch.nn as nn

class BiFPN(nn.Module):
    def __init__(self, fpn_sizes):
        super(BiFPN, self).__init__()
        
        P3_channels, P4_channels, P5_channels = fpn_sizes
        self.W_bifpn = 64
        
        # P5 to P4
        self.p5_td_conv = nn.Conv2d(P5_channels, self.W_bifpn, kernel_size=3, stride=1, bias=True, padding=1)
        self.p5_td_conv_2 = nn.Conv2d(self.W_bifpn, self.W_bifpn, kernel_size=3, stride=1, groups=self.W_bifpn, bias=True, padding=1)
        self.p5_td_act = nn.ReLU()
        self.p5_td_conv_bn = nn.BatchNorm2d(self.W_bifpn)
        self.p5_td_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p5_td_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        
        # P4 to P3
        self.p4_td_conv = nn.Conv2d(P4_channels, self.W_bifpn, kernel_size=3, stride=1, bias=True, padding=1)
        self.p4_td_conv_2 = nn.Conv2d(self.W_bifpn, self.W_bifpn, kernel_size=3, stride=1, groups=self.W_bifpn, bias=True, padding=1)
        self.p4_td_act = nn.ReLU()
        self.p4_td_conv_bn = nn.BatchNorm2d(self.W_bifpn)
        self.p4_td_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p4_td_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        
        # Upsample P5 to P4
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # P3 output
        self.p3_out_conv = nn.Conv2d(P3_channels, self.W_bifpn, kernel_size=3, stride=1, bias=True, padding=1)
        self.p3_out_conv_2 = nn.Conv2d(self.W_bifpn, self.W_bifpn, kernel_size=3, stride=1, groups=self.W_bifpn, bias=True, padding=1)
        self.p3_out_act = nn.ReLU()
        self.p3_out_conv_bn = nn.BatchNorm2d(self.W_bifpn)
        self.p3_out_w1 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        self.p3_out_w2 = torch.tensor(1, dtype=torch.float, requires_grad=True)
        
        # Upsample P4 to P3
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # P4 output
        self.p4_out_conv = nn.Conv2d(self.W_bifpn, self.W_bifpn, kernel_size=3, stride=1, groups=self.W_bifpn, bias=True, padding=1)
        self.p4_out_act = nn.ReLU()
        self.p4_out_conv_bn = nn.BatchNorm2d(self.W_bifpn)
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


if __name__ == '__main__':
    fpn = BiFPN([40, 112, 192])

    c3 = torch.randn([1, 40, 64, 64])
    c4 = torch.randn([1, 112, 32, 32])
    c5 = torch.randn([1, 192, 16, 16])

    feats = [c3, c4, c5]
    output = fpn.forward(feats)
    for feat in output:
        print(feat.shape) 