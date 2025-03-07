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
                                    pretrained=False,
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
                                    pretrained=False,
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

class FeatureExtractor_C3k2(nn.Module):
    def __init__(self):
        super().__init__()
        width = 0.75
        depth = 1

        channels = [64, 128, 256, 512, 1024]
        channels = [int(c * width) for c in channels]

        depth_list = [2, 2, 2, 2, 2]
        depth_list = [int(d * depth) for d in depth_list]
        
        self.stage1 = nn.Sequential(
            Conv(3, channels[0], k=3, s=2),  # 1/2
            Conv(channels[0], channels[1], k=3, s=2),  # 1/4
            C3k2(channels[1], channels[2], depth_list[0], False, 0.25),
        )
        self.stage2 = nn.Sequential(
            Conv(channels[2], channels[2], k=3, s=2),  # 1/8
            C3k2(channels[2], channels[3], depth_list[1], False, 0.25),
        )
        self.stage3 = nn.Sequential(
            Conv(channels[3], channels[3], k=3, s=2),  # 1/16
            C3k2(channels[3], channels[3], depth_list[2], True),
        )
        # self.stage4 = nn.Sequential(
        #     Conv(channels[3], channels[4], k=3, s=2),  # 1/32
        #     C3k2(channels[4], channels[4], depth_list[3], True),
        #     SPPF(channels[4], channels[4], 5),
        #     C2PSA(channels[4], channels[4], depth_list[4], 0.25)
        # )

        self._initialize_weights()

    def forward(self, x):
        c2 = self.stage1(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        # c5 = self.stage4(c4)

        return [c3, c4]
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    # model = FeatureExtractor_ConvNextTiny()
    # model = FeatureExtractor_resnet()
    # model = FeatureExtractor_resnet_fpn()
    # model = FeatureExtractor_MobileNetV4_FPN()
    model = FeatureExtractor_C3k2()
    input = torch.randn(1, 3, 512, 512)
    features = model(input)  # 得到4个尺度的特征图

    for feat in features:
        print(feat.shape) 