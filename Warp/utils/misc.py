import torchvision.transforms as T
import cv2
import numpy as np
import torch

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
import constant
grid_h = constant.GRID_H
grid_w = constant.GRID_W

resize_512 = T.Resize((512, 512), antialias=True)

# 数据增强, 对结果好像没啥影响...
def data_aug(img1, img2):
    # Randomly shift brightness
    random_brightness = torch.randn(1).uniform_(0.7, 1.3).cuda()
    img1_aug = img1 * random_brightness
    random_brightness = torch.randn(1).uniform_(0.7, 1.3).cuda()
    img2_aug = img2 * random_brightness

    # Randomly shift color
    white = torch.ones([img1.size()[0], img1.size()[2], img1.size()[3]]).cuda()
    random_colors = torch.randn(3).uniform_(0.7, 1.3).cuda()
    color_image = torch.stack([white * random_colors[i] for i in range(3)], axis=1)
    img1_aug *= color_image

    random_colors = torch.randn(3).uniform_(0.7, 1.3).cuda()
    color_image = torch.stack([white * random_colors[i] for i in range(3)], axis=1)
    img2_aug *= color_image

    # clip
    img1_aug = torch.clamp(img1_aug, -1, 1)
    img2_aug = torch.clamp(img2_aug, -1, 1)

    return img1_aug, img2_aug

# draw mesh on image
# warp: h*w*3
# f_local: grid_h*grid_w*2
def draw_mesh_on_warp(warp, f_local):

    warp = np.ascontiguousarray(warp)

    point_color = (0, 255, 0)  # BGR
    thickness = 2
    lineType = 8

    num = 1
    for i in range(grid_h + 1):
        for j in range(grid_w + 1):

            num = num + 1
            if j == grid_w and i == grid_h:
                continue
            elif j == grid_w:
                cv2.line(
                    warp,
                    (int(f_local[i, j, 0]), int(f_local[i, j, 1])),
                    (int(f_local[i + 1, j, 0]), int(f_local[i + 1, j, 1])),
                    point_color,
                    thickness,
                    lineType,
                )
            elif i == grid_h:
                cv2.line(
                    warp,
                    (int(f_local[i, j, 0]), int(f_local[i, j, 1])),
                    (int(f_local[i, j + 1, 0]), int(f_local[i, j + 1, 1])),
                    point_color,
                    thickness,
                    lineType,
                )
            else:
                cv2.line(
                    warp,
                    (int(f_local[i, j, 0]), int(f_local[i, j, 1])),
                    (int(f_local[i + 1, j, 0]), int(f_local[i + 1, j, 1])),
                    point_color,
                    thickness,
                    lineType,
                )
                cv2.line(
                    warp,
                    (int(f_local[i, j, 0]), int(f_local[i, j, 1])),
                    (int(f_local[i, j + 1, 0]), int(f_local[i, j + 1, 1])),
                    point_color,
                    thickness,
                    lineType,
                )

    return warp

# Covert global homo into mesh
def convert_H_to_mesh(H, rigid_mesh):
    '''
    将前一步单应性矩阵带来的初始偏移施加到控制点上
    '''

    H_inv = torch.inverse(H)
    ori_pt = rigid_mesh.reshape(rigid_mesh.size()[0], -1, 2)
    ones = torch.ones(rigid_mesh.size()[0], (grid_h + 1) * (grid_w + 1), 1)
    if torch.cuda.is_available():
        ori_pt = ori_pt.cuda()
        ones = ones.cuda()

    ori_pt = torch.cat((ori_pt, ones), 2)  # bs*(grid_h+1)*(grid_w+1)*3
    tar_pt = torch.matmul(H_inv, ori_pt.permute(0, 2, 1))  # bs*3*(grid_h+1)*(grid_w+1)

    mesh_x = torch.unsqueeze(tar_pt[:, 0, :] / tar_pt[:, 2, :], 2)
    mesh_y = torch.unsqueeze(tar_pt[:, 1, :] / tar_pt[:, 2, :], 2)
    mesh = torch.cat((mesh_x, mesh_y), 2).reshape(
        [rigid_mesh.size()[0], grid_h + 1, grid_w + 1, 2]
    )

    return mesh

# 生成网格的各控制点坐标,其形状为 : [batch_size, grid_h + 1, grid_w + 1, 2]
def get_rigid_mesh(batch_size, height, width):
    ww = torch.matmul(
        torch.ones([grid_h + 1, 1]),
        torch.unsqueeze(torch.linspace(0.0, float(width), grid_w + 1), 0),
    )
    hh = torch.matmul(
        torch.unsqueeze(torch.linspace(0.0, float(height), grid_h + 1), 1),
        torch.ones([1, grid_w + 1]),
    )
    if torch.cuda.is_available():
        ww = ww.cuda()
        hh = hh.cuda()

    ori_pt = torch.cat((ww.unsqueeze(2), hh.unsqueeze(2)), 2)  # (grid_h+1)*(grid_w+1)*2
    ori_pt = ori_pt.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return ori_pt


# normalize mesh from -1 ~ 1
def get_norm_mesh(mesh, height, width):
    batch_size = mesh.size()[0]
    mesh_w = mesh[..., 0] * 2.0 / float(width) - 1.0
    mesh_h = mesh[..., 1] * 2.0 / float(height) - 1.0
    norm_mesh = torch.stack([mesh_w, mesh_h], 3)  # bs*(grid_h+1)*(grid_w+1)*2

    return norm_mesh.reshape([batch_size, -1, 2])  # bs*-1*2