import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from Conv import *
from Attention import *

class SPDConv(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = self.conv(x)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut for matching dimensions if needed
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()  # When dimensions match, identity shortcut

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # Conv -> BN -> ReLU
        out = self.bn2(self.conv2(out))

        out += self.shortcut(x)  # Skip connection
        out = F.relu(out)  # Apply ReLU after adding the shortcut
        return out
       
class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))
    
class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )

class SPP(nn.Module):
    def __init__(self, pool_sizes=[1, 2, 4]):
        super(SPP, self).__init__()
        self.pool_sizes = pool_sizes
        
    def forward(self, x):
        n, c, h, w = x.size()
        spp_outs = []
        for pool_size in self.pool_sizes:
            spp = nn.AdaptiveMaxPool2d((pool_size, pool_size))(x)
            spp_flat = spp.view(n, -1)  # 展平为 [n, c*pool_size*pool_size]
            spp_outs.append(spp_flat)
        # 拼接所有尺度的池化结果
        return torch.cat(spp_outs, dim=1)

class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))
    
class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x
    
class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x
    
class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))
    
class CCL(nn.Module):
    def __init__(self):
        super(CCL, self).__init__()

    @staticmethod
    def extract_patches(x, kernel=3, stride=1):
        if kernel != 1:
            x = nn.ZeroPad2d(1)(x)
        x = x.permute(0, 2, 3, 1)
        all_patches = x.unfold(1, kernel, stride).unfold(2, kernel, stride)
        return all_patches
    
    def forward(self, feature_1, feature_2):
        bs, c, h, w = feature_1.size()

        norm_feature_1 = F.normalize(feature_1, p=2, dim=1)
        norm_feature_2 = F.normalize(feature_2, p=2, dim=1)

        # stage1 : 计算Correlation Volume
        patches = self.extract_patches(norm_feature_2) # [N ,H, W, C, K, K]
            
        # [N, H*W, C, K, K]
        matching_filters = patches.reshape((patches.size()[0], -1, patches.size()[3], patches.size()[4], patches.size()[5]))

        match_vol = []
        for i in range(bs):
            # input[1, C, H, W], wieght[H*W, C, K, K] --> output[1, H*W, H, W]
            single_match = F.conv2d(norm_feature_1[i].unsqueeze(0), matching_filters[i], padding=1, dilation=1)
            match_vol.append(single_match)

        # correlation volume: 代表ref上每个path与target上每个patch的匹配程度
        # 其中的每个像素可以看成一个长度为[H*W]的向量，代表ref上此位置的patch与target上各个patch的匹配程度
        match_vol = torch.cat(match_vol, 0) # [N, H*W, H, W]

        # stage2 : scale softmax
        softmax_scale = 10 # 论文里的scale factor,作用是抑制小的匹配值,增强大的匹配值

        # 使用softmax将patch间的匹配程度归一化到[0,1], 从而将匹配看成一个分类问题
        match_vol = F.softmax(match_vol * softmax_scale, 1)

        # stage3 : 计算Feature Flow:一个[N,2,H,W]的Tensor, 代表了ref上每个path移动到target上匹配的patch的位移
        channel = match_vol.size()[1]

        h_one = torch.linspace(0, h - 1, h)
        one1w = torch.ones(1, w)
        if torch.cuda.is_available():
            h_one = h_one.cuda()
            one1w = one1w.cuda()
        h_one = torch.matmul(h_one.unsqueeze(1), one1w)
        h_one = h_one.unsqueeze(0).unsqueeze(0).expand(bs, channel, -1, -1)

        w_one = torch.linspace(0, w - 1, w)
        oneh1 = torch.ones(h, 1)
        if torch.cuda.is_available():
            w_one = w_one.cuda()
            oneh1 = oneh1.cuda()
        w_one = torch.matmul(oneh1, w_one.unsqueeze(0))
        w_one = w_one.unsqueeze(0).unsqueeze(0).expand(bs, channel, -1, -1)

        c_one = torch.linspace(0, channel - 1, channel)
        if torch.cuda.is_available():
            c_one = c_one.cuda()
        c_one = c_one.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(bs, -1, h, w)

        flow_h = match_vol * (c_one // w - h_one)
        flow_h = torch.sum(flow_h, dim=1, keepdim=True)
        flow_w = match_vol * (c_one % w - w_one)
        flow_w = torch.sum(flow_w, dim=1, keepdim=True)

        # [N, 2, H, W]
        feature_flow = torch.cat([flow_w, flow_h], 1)

        return feature_flow
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
class CCL_ChannelAttention(CCL):
    def __init__(self, in_chs):
        super().__init__()
        self.channel_atten1 = EfficientChannelAttention(in_chs)
        self.channel_atten2 = EfficientChannelAttention(in_chs)
        self._initialize_weights()

    def forward(self, feature_1, feature_2):
        feature_1 = self.channel_atten1(feature_1)
        feature_2 = self.channel_atten2(feature_2)
        return super().forward(feature_1, feature_2)
    
class CCL_LocalWindowAttention(CCL):
    def __init__(self, in_chs):
        super().__init__()
        self.atten1 = LocalWindowAttention(in_chs)
        self.atten2 = LocalWindowAttention(in_chs)
        self._initialize_weights()

    def forward(self, feature_1, feature_2):
        feature_1 = self.atten1(feature_1)
        feature_2 = self.atten2(feature_2)
        return super().forward(feature_1, feature_2)
    
class CCL_SimAM(CCL):
    def __init__(self):
        super().__init__()
        self.atten = SimAM()
        self._initialize_weights()

    def forward(self, feature_1, feature_2):
        feature_1 = self.atten(feature_1)
        feature_2 = self.atten(feature_2)
        return super().forward(feature_1, feature_2)

def cost_volume(x1, x2, search_range, normBoth=False, fast=True):
    """Build cost volume for associating a pixel from Image1 with its corresponding pixels in Image2.
      Args:
          c1: Level of the feature pyramid of Image1
          warp: Warped level of the feature pyramid of image22
          search_range: Search range (maximum displacement)
      """
    if normBoth:
        x1 = F.normalize(x1, p=2, dim=1)
        x2 = F.normalize(x2, p=2, dim=1)
    else:
        x1 = F.normalize(x1, p=2, dim=1)
    bs, c, h, w = x1.shape
    padded_x2 = F.pad(x2, [search_range] * 4)  # [b,c,h,w] -> [b,c,h+sr*2,w+sr*2]
    max_offset = search_range * 2 + 1

    if fast:
        # faster(*2) but cost higher(*n) GPU memory
        # [N, C, max_offset^2, H, W]
        patches = F.unfold(padded_x2, (max_offset, max_offset)).reshape(bs, c, max_offset ** 2, h, w)
        # [N, max_offset^2, H, W]
        cost_vol = (x1.unsqueeze(2) * patches).mean(dim=1, keepdim=False)
    else:
        # slower but save memory
        cost_vol = []
        for j in range(0, max_offset):
            for i in range(0, max_offset):
                x2_slice = padded_x2[:, :, j:j + h, i:i + w]
                cost = torch.mean(x1 * x2_slice, dim=1, keepdim=True)
                # [N, 1, H, W]
                cost_vol.append(cost)
        # [N, max_offset^2, H, W]
        cost_vol = torch.cat(cost_vol, dim=1)

    cost_vol = F.leaky_relu(cost_vol, 0.1)
    return cost_vol

class AttentionCostVolume(nn.Module):
    def __init__(self, search_range=8, out_planes=49):
        super(AttentionCostVolume, self).__init__()
        max_offset = search_range * 2 + 1
        in_planes = max_offset * max_offset
        self.att = nn.Conv2d(in_planes, in_planes, kernel_size=7, padding=3, groups=in_planes)
        self.agg = nn.Sequential(
            Conv(in_planes,in_planes//2,3,1),
            Conv(in_planes//2,out_planes,3,1)
        )
        self.sr = search_range

    def forward(self, f1, f2, normBoth=False):
        # [N, max_offset^2, H, W]
        match_vol = cost_volume(f1,f2, self.sr, normBoth, fast=False) 
        att_vol = match_vol * self.att(match_vol)
        agg_vol = self.agg(att_vol)
        # [N, out_planes, H, W]
        return agg_vol
    
class AttentionCorrelationVolume(CCL):
    def __init__(self, in_chs, out_chs):
        super().__init__()

        self.agg = nn.Sequential(
            DSConv(in_chs, in_chs//2, k=3),
            DSConv(in_chs//2, out_chs, k=3),
        )

    def forward(self, feature_1, feature_2):
        bs, c, h, w = feature_1.size()

        norm_feature_1 = F.normalize(feature_1, p=2, dim=1)
        norm_feature_2 = F.normalize(feature_2, p=2, dim=1)

        # stage1 : 计算Correlation Volume
        patches = self.extract_patches(norm_feature_2) # [N, H, W, C, K, K]

        # [N, H*W, C, K, K]
        matching_filters = patches.reshape((patches.size()[0], -1, patches.size()[3], patches.size()[4], patches.size()[5]))

        match_vol = []
        for i in range(bs):
            # input[1, C, H, W], wieght[H*W, C, K, K] --> output[1, H*W, H, W]
            single_match = F.conv2d(norm_feature_1[i].unsqueeze(0), matching_filters[i], padding=1)
            match_vol.append(single_match)

        # Correlation Volume: 代表ref上每个path与target上每个patch的匹配程度
        # 其中的每个像素可以看成一个长度为[H*W]的向量，代表ref上此位置的patch与target上各个patch的匹配程度
        match_vol = torch.cat(match_vol, 0) # [N, H*W, H, W]
        agg_vol = self.agg(match_vol)
        # [N, out_planes, H, W]
        return agg_vol

######################################## C3k2-Faster begin ########################################
from timm.layers import DropPath

class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x

class Faster_Block(nn.Module):
    def __init__(self,
                 inc,
                 dim,
                 n_div=4,
                 mlp_ratio=2,
                 drop_path=0.1,
                 layer_scale_init_value=0.0,
                 pconv_fw_type='split_cat'
                 ):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer = [
            Conv(dim, mlp_hidden_dim, 1),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )
        
        self.adjust_channel = None
        if inc != dim:
            self.adjust_channel = Conv(inc, dim, 1)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        if self.adjust_channel is not None:
            x = self.adjust_channel(x)
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x

class C3k_Faster(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(Faster_Block(c_, c_) for _ in range(n)))

class C3k2_Faster(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_Faster(self.c, self.c, 2, shortcut, g) if c3k else Faster_Block(self.c, self.c) for _ in range(n))
######################################## C3k2-Faster end ########################################

######################################## Edge information enhancement module start ########################################
class SobelConv(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()
        
        sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sobel_kernel_y = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)
        sobel_kernel_x = torch.tensor(sobel.T, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)
        
        self.sobel_kernel_x_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.sobel_kernel_y_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        
        self.sobel_kernel_x_conv3d.weight.data = sobel_kernel_x.clone()
        self.sobel_kernel_y_conv3d.weight.data = sobel_kernel_y.clone()
        
        self.sobel_kernel_x_conv3d.requires_grad = False
        self.sobel_kernel_y_conv3d.requires_grad = False

    def forward(self, x):
        return (self.sobel_kernel_x_conv3d(x[:, :, None, :, :]) + self.sobel_kernel_y_conv3d(x[:, :, None, :, :]))[:, :, 0]

class EIEM_Faster(nn.Module):
    def __init__(self, inc, ouc) -> None:
        super().__init__()
        
        self.sobel_branch = SobelConv(inc)
        self.conv_branch = Partial_conv3(inc)
        self.conv1 = Conv(inc * 2, inc, 1)
        self.conv2 = Partial_conv3(inc)
    
    def forward(self, x):
        x_sobel = self.sobel_branch(x)
        x_conv = self.conv_branch(x)
        x_concat = torch.cat([x_sobel, x_conv], dim=1)
        x_feature = self.conv1(x_concat)
        x = self.conv2(x_feature + x)
        return x

class C3k_EIEM_Faster(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(EIEM_Faster(c_, c_) for _ in range(n)))

class C3k2_EIEM_Faster(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(C3k_EIEM_Faster(self.c, self.c, 2, shortcut, g) if c3k else EIEM_Faster(self.c, self.c) for _ in range(n))

if __name__ == '__main__':
    input1 = torch.randn(1, 512, 32, 32).cuda()
    input2 = torch.randn(1, 512, 32, 32).cuda()
    # test_block = AttentionCostVolume(8, 512)
    test_block = CCL_SimAM().cuda()
    # test_block = AttentionCorrelationVolume(32*32, out_chs=64).cuda()
    output = test_block(input1, input2)
    print(output.shape)