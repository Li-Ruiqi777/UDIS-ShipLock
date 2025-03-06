import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import timm
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from Conv import Conv
from block import C3k2, SPPF, C2PSA
from afpn import AFPN_P345

class FPN(nn.Module):
    def __init__(self, width=0.25, depth=0.5):
        super().__init__()       
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # 输入1/16
        self.fusion2 = C3k2(int((1024 + 512) * width), int(1024 * width), int(2 * depth), False)

        # 输入1/8
        self.fusion1 = C3k2(int((1024 + 512) * width), int(512 * width), int(2 * depth), False)
    
    def forward(self, x1, x2, x3):
        # 1/16
        p2 = torch.cat([x2, self.upsample(x3)], dim=1)
        p2 = self.fusion2(p2)

        # 1/8
        p1 = torch.cat([x1, self.upsample(p2)], dim=1)
        p1 = self.fusion1(p1)
        
        return [p1, p2]

class FeatureExtractor_YOLOFPN(nn.Module):
    def __init__(self):
        super().__init__()
        width = 0.25
        depth = 0.5

        self.stage1 = nn.Sequential(
            Conv(3, int(64 * width), k=3, s=2), # 1/2
            Conv(int(64 * width), int(128 * width), k=3, s=2), # 1/4
            C3k2(int(128 * width), int(256 * width), int(2 * depth), False, 0.25),
            Conv(int(256 * width), int(256 * width), k=3, s=2), # 1/8
            C3k2(int(256 * width), int(512 * width), int(2 * depth), False, 0.25),
        )

        self.stage2 = nn.Sequential(
            Conv(int(512 * width), int(512 * width), k=3, s=2), # 1/16
            C3k2(int(512 * width), int(512 * width), int(2 * depth), True),
        )

        self.stage3 = nn.Sequential(
            Conv(int(512 * width), int(1024 * width), k=3, s=2), # 1/32
            C3k2(int(1024 * width), int(1024 * width), int(2 * depth), True),
            SPPF(int(1024 * width), int(1024 * width), 5),
            C2PSA(int(1024 * width), int(1024 * width), int(2 * depth), 0.25)
        )

        self.fpn = FPN(width, depth)

        self._initialize_weights()

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
        
    def forward(self, x):
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)

        return self.fpn(x1, x2, x3)

class FeatureExtractor_resnet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet50_model = models.resnet.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.feature_extractor_stage1 = nn.Sequential(
            resnet50_model.conv1,
            resnet50_model.bn1,
            resnet50_model.relu,
            resnet50_model.maxpool,
            resnet50_model.layer1,
            resnet50_model.layer2,
        )

        self.feature_extractor_stage2 = nn.Sequential(
            resnet50_model.layer3,
        )

    def forward(self, x):
        features = []
        x = self.feature_extractor_stage1(x) # [N, 512, H/8, W/8]
        features.append(x)
        x = self.feature_extractor_stage2(x) # [N, 1024, H/16, W/16]
        features.append(x)
        
        return features

class FeatureExtractor_resnet_fpn(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50_fpn = resnet_fpn_backbone('resnet50', weights=models.ResNet50_Weights.DEFAULT, trainable_layers=5, returned_layers=[2, 3])

    def forward(self, x):
        features = []
        temp = self.resnet50_fpn(x)
        features.append(temp['0'])
        features.append(temp['1'])

        return features

class FeatureExtractor_ConvNextTiny(nn.Module):
    def __init__(self):
        super().__init__()
        conextTiny = timm.create_model("convnext_tiny", 
                                       pretrained=True,
                                    #    features_only=True
        )
        
        self.feature_extractor_stage1 = nn.Sequential(
            conextTiny.stem,
            conextTiny.stages[0],  # 对应原ResNet的layer1输出
            conextTiny.stages[1]   # 对应原ResNet的layer2输出
        )
       
        self.feature_extractor_stage2 = conextTiny.stages[2]


    def forward(self, x):
        features = []
        x = self.feature_extractor_stage1(x) # [N, 192, H/8, W/8]
        features.append(x) 
        x = self.feature_extractor_stage2(x) # [N, 384, H/16, W/16]
        features.append(x)
        
        return features
    
class FeatureExtractor_MobileNetV4(nn.Module):
    def __init__(self):
        super().__init__()
        model = timm.create_model("mobilenetv4_hybrid_medium.ix_e550_r384_in1k", 
                                    pretrained=True,
                                    features_only=True,
                                    # pretrained_cfg_overlay=dict(file='C:/Users/lrq/Desktop/mobilenetv4/snapshots/11/model.safetensors'),
                                    out_indices=[2, 3]
        )
        
        self.feature_extractor = model
       
    def forward(self, x):
        features = self.feature_extractor(x)        
        return features
    
class FeatureExtractor_MobileNetV4_FPN(nn.Module):
    def __init__(self):
        super().__init__()
        model = timm.create_model("mobilenetv4_hybrid_medium.ix_e550_r384_in1k", 
                                    pretrained=True,
                                    features_only=True,
                                    # pretrained_cfg_overlay=dict(file='C:/Users/lrq/Desktop/mobilenetv4/snapshots/11/model.safetensors'),
                                    out_indices=[2, 3, 4]
        )
        
        self.feature_extractor = model
        self.fpn = AFPN_P345([80, 160, 960])
       
    def forward(self, x):
        [c3, c4, c5] = self.feature_extractor(x)
        [p3, p4, _] = self.fpn([c3, c4, c5])
        return [p3, p4]


if __name__ == '__main__':
    # model = FeatureExtractor_ConvNextTiny()
    # model = FeatureExtractor_resnet()
    # model = FeatureExtractor_resnet_fpn()
    model = FeatureExtractor_MobileNetV4_FPN()
    input = torch.randn(1, 3, 512, 512)
    features = model(input)  # 得到4个尺度的特征图

    for feat in features:
        print(feat.shape) 