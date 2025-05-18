import torch
import torch.nn as nn
import torchvision.models as models
import ssl

from modules import *
import utils.torch_DLT as torch_DLT
import utils.torch_homo_transform as torch_homo_transform
from utils.misc import *
import utils.constant as constant

grid_h = constant.GRID_H
grid_w = constant.GRID_W
device = constant.device

class UDIS2(nn.Module):
    def __init__(self):
        super(UDIS2, self).__init__()

        # H的回归网络
        self.regressNet1_part1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.regressNet1_part2 = nn.Sequential(
            # nn.Linear(in_features=256, out_features=4096, bias=True),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=4096, out_features=1024, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=8, bias=True)
        )

        # TPS的回归网络
        self.regressNet2_part1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.regressNet2_part2 = nn.Sequential(
            # nn.Linear(in_features=512, out_features=4096, bias=True),
            nn.Linear(in_features=8192, out_features=4096, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=4096, out_features=2048, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=2048, out_features=(grid_w+1)*(grid_h+1)*2, bias=True)
        )

        self.initailize_weights()

        ssl._create_default_https_context = ssl._create_unverified_context
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

        self.CCL = CCL()
    
    def initailize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
    # 返回H中4点的偏移以及TPS中各控制点的偏移
    def forward(self, input1_tesnor, input2_tesnor):
        batch_size, _, img_h, img_w = input1_tesnor.size()

        # [N, 512, 64, 64]
        feature_1_64 = self.feature_extractor_stage1(input1_tesnor)  # refrence的1/8
        # [N, 1024, 32, 32]
        feature_1_32 = self.feature_extractor_stage2(feature_1_64)  # refrence的1/16

        # [N, 512, 64, 64]
        feature_2_64 = self.feature_extractor_stage1(input2_tesnor)  # target的1/8
        # [N, 1024, 32, 32]
        feature_2_32 = self.feature_extractor_stage2(feature_2_64)  # target的1/16

        # starge1: 计算H_motion
        # [N, 2, 32, 32]
        correlation_32 = self.CCL(feature_1_32, feature_2_32)
        
        temp_1 = self.regressNet1_part1(correlation_32)
        temp_1 = temp_1.view(temp_1.size()[0], -1)
        # [N, 8]
        offset_1 = self.regressNet1_part2(temp_1)
        # [N, 4, 2]
        H_motion_1 = offset_1.reshape(-1, 4, 2)

        src_p = torch.FloatTensor([[0.0, 0.0], [img_w, 0.0], [0.0, img_h], [img_w, img_h]]).to(device)
        src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
        dst_p = src_p + H_motion_1
        H = torch_DLT.tensor_DLT(src_p / 8, dst_p / 8)
        
        # 这里之所以/8是因为输出feature map的尺寸是原图的1/8
        M_tensor = torch.tensor(
            [
                [img_w / 8 / 2.0, 0.0, img_w / 8 / 2.0],
                [0.0, img_h / 8 / 2.0, img_h / 8 / 2.0],
                [0.0, 0.0, 1.0],
            ]
        ).to(device)


        M_tile = M_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        M_tensor_inv = torch.inverse(M_tensor)
        M_tile_inv = M_tensor_inv.unsqueeze(0).expand(batch_size, -1, -1)
        H_mat = torch.matmul(torch.matmul(M_tile_inv, H), M_tile)

        # 将全局的单应性变换H施加到target上
        warp_feature_2_64 = torch_homo_transform.transformer(feature_2_64, H_mat, (int(img_h / 8), int(img_w / 8)))

        # starge2: 计算Mesh motion
        # [N, 2, 64, 64]
        correlation_64 = self.CCL(feature_1_64, warp_feature_2_64)

        temp_2 = self.regressNet2_part1(correlation_64)
        temp_2 = temp_2.view(temp_2.size()[0], -1)
        # [N, (grid_w+1)*(grid_h+1)*2]
        offset_2 = self.regressNet2_part2(temp_2)  # 计算TPS中各控制点的偏移

        return offset_1, offset_2