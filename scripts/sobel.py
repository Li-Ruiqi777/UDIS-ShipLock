import torch
import torch.nn as nn
import numpy as np
import cv2

class SobelConv2D(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()
        
        # 定义 Sobel 核（X/Y 方向）
        sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
        sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
        
        # 构造 Conv2d 的权重（形状: [C, 1, 3, 3]）
        kernel_x = torch.tensor(sobel_x).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        kernel_y = torch.tensor(sobel_y).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        
        # 创建 X/Y 方向的卷积层
        self.conv_x = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.conv_y = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        
        # 加载固定权重（不更新）
        self.conv_x.weight.data = kernel_x
        self.conv_y.weight.data = kernel_y
        self.conv_x.weight.requires_grad_(False)
        self.conv_y.weight.requires_grad_(False)

    def forward(self, x):
        edge_x = self.conv_x(x)
        edge_y = self.conv_y(x)
        sum = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        return [edge_x, edge_y, sum]

class LearnableSobelConv2D(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        
        self.conv_x = nn.Conv2d(channels, channels, kernel_size=kernel_size, 
                              padding=padding, groups=channels, bias=bias)
        self.conv_y = nn.Conv2d(channels, channels, kernel_size=kernel_size, 
                              padding=padding, groups=channels, bias=bias)
        
        self._init_sobel_weights()
        
    def _init_sobel_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
    def symmetrize(self, weight, direction):
        """ 对称化权重"""
        if direction == 'x':
            partA = weight - weight.flip(2)
            partB = partA.flip(2).flip(3)
            weight.data = (partA - partB) / 2

        elif direction == 'y':
            partA = weight - weight.flip(3)
            partB = partA.flip(2).flip(3) 
            weight.data = (partA - partB) / 2
        
            
    def forward(self, x):
        # 对称化处理
        self.symmetrize(self.conv_x.weight, direction='x')
        self.symmetrize(self.conv_y.weight, direction='y')
        
        # 计算边缘响应
        edge_x = self.conv_x(x)
        edge_y = self.conv_y(x)
        sum = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        return [edge_x, edge_y, sum]

# 测试
if __name__ == '__main__':
    img = cv2.imread('E:/DeepLearning/0_DataSets/008-UDIS-D_test/test/input1/000005.jpg')
    img = torch.tensor(img.transpose(2, 0, 1).astype(np.float32) / 255.0).unsqueeze(0).cuda()
    model = SobelConv2D(3).cuda()
    result = model(img)

    cv2.imshow('edge_x', result[0][0].cpu().permute(1, 2, 0).detach().numpy())  
    cv2.imshow('edge_y', result[1][0].cpu().permute(1, 2, 0).detach().numpy())
    cv2.imshow('sum', result[2][0].cpu().permute(1, 2, 0).detach().numpy())
    cv2.waitKey(0)

    cv2.imwrite('edge_x.jpg', 255*result[0][0].cpu().permute(1, 2, 0).detach().numpy().astype(np.uint8))
    cv2.imwrite('edge_y.jpg', 255*result[1][0].cpu().permute(1, 2, 0).detach().numpy().astype(np.uint8))
    cv2.imwrite('sum.jpg', 255*result[2][0].cpu().permute(1, 2, 0).detach().numpy().astype(np.uint8))