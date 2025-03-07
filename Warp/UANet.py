import torch
import torch.nn as nn

from modules import *
import utils.torch_DLT as torch_DLT
import utils.torch_homo_transform as torch_homo_transform
from utils.misc import *
import utils.constant
gird_w = utils.constant.GRID_W
gird_h = utils.constant.GRID_H
device = utils.constant.device

# Unsupervised Alignment Network
class UANet(nn.Module):
    def __init__(self):
        super(UANet, self).__init__()
        self.feature_extractor = FeatureExtractor_C3k2()
        self.homo_regress_net = HomoRegressNet()
        self.mesh_regress_net = MeshRegressNet([gird_h, gird_w])
        self.CCL1 = CCL()
        self.CCL2 = CCL()

    def forward(self, ref_tensor, target_tensor):
        batch_size, _, img_h, img_w = ref_tensor.size()

        ref_feat_list = self.feature_extractor(ref_tensor) #[[N,C,64,64], [N,C,32,32]]
        target_feat_list = self.feature_extractor(target_tensor)

        # regress stage 1
        correlation_1 = self.CCL1(ref_feat_list[-1], target_feat_list[-1])
        H_motion = self.homo_regress_net(correlation_1) # [N, 4, 2]

        src_p = torch.FloatTensor([[0.0, 0.0], [img_w, 0.0], [0.0, img_h], [img_w, img_h]]).to(device)
        src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
        dst_p = src_p + H_motion

        H = torch_DLT.tensor_DLT(src_p / 8, dst_p / 8)
        
        # regress stage 2

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
        warp_target_feat_64 = torch_homo_transform.transformer(target_feat_list[-2], H_mat, (int(img_h / 8), int(img_w / 8)))

        correlation_2 = self.CCL2(ref_feat_list[-2], warp_target_feat_64)
        Mesh_motion = self.mesh_regress_net(correlation_2)

        return [H_motion, Mesh_motion]
    