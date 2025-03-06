import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.constant as constant
grid_h = constant.GRID_H
grid_w = constant.GRID_W


def get_norm_loss(img1, img2, l_num=1):
    return torch.mean(torch.abs((img1 - img2) ** l_num))


def get_overlap_loss(input1, input2, batch_outputs_for_train):
    """
    计算重叠区的像素差值, 对应论文中的alignment loss
    """
    batch_size, _, img_h, img_w = input1.size()
    epsilon = 1e-6

    warped_target_and_mask = batch_outputs_for_train['warped_target_and_mask']
    warped_reference_and_mask = batch_outputs_for_train['warped_reference_and_mask']
    tps_warped_target = batch_outputs_for_train['tps_warped_target']
    tps_warped_mask = batch_outputs_for_train['tps_warped_mask']

    # part one: sym homo loss with color balance
    delta1 = (torch.sum(warped_target_and_mask[:, 0:3, :, :], [2, 3]) - torch.sum(input1 * warped_target_and_mask[:, 3:6, :, :], [2, 3])) / (torch.sum(warped_target_and_mask[:, 3:6, :, :], [2, 3]) + epsilon)

    input1_balance = input1 + delta1.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    delta2 = (torch.sum(warped_reference_and_mask[:, 0:3, :, :], [2, 3]) - torch.sum(input2 * warped_reference_and_mask[:, 3:6, :, :], [2, 3])) / (torch.sum(warped_reference_and_mask[:, 3:6, :, :], [2, 3]) + epsilon)

    input2_balance = input2 + delta2.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    # abs(Ir·φ(1,H) - φ(It,H)) + abs(It·φ(1,H-1) - φ(Ir,H-1))
    lp_loss_1 = get_norm_loss(input1_balance * warped_target_and_mask[:, 3:6, :, :], warped_target_and_mask[:, 0:3, :, :], 1) 
    + get_norm_loss(input2_balance * warped_reference_and_mask[:, 3:6, :, :], warped_reference_and_mask[:, 0:3, :, :], 1)
    
    # part two: tps loss with color balance
    delta3 = (torch.sum(tps_warped_target, [2, 3]) - torch.sum(input1 * tps_warped_mask, [2, 3])) / (torch.sum(tps_warped_mask, [2, 3]) + epsilon)
    input1_newbalance = input1 + delta3.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    lp_loss_2 = get_norm_loss(input1_newbalance * tps_warped_mask, tps_warped_target, 1)

    lp_loss = 3.0 * lp_loss_1 + 1.0 * lp_loss_2

    return lp_loss


def get_overlap_loss_ft(input1, warp_mesh, warp_mesh_mask):
    batch_size, _, img_h, img_w = input1.size()
    epsilon = 1e-6

    delta3 = (torch.sum(warp_mesh, [2, 3]) - torch.sum(input1 * warp_mesh_mask, [2, 3])) / (torch.sum(warp_mesh_mask, [2, 3]) + epsilon)
    input1_newbalance = input1 + delta3.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_h, img_w)

    lp_loss_2 = get_norm_loss(input1_newbalance * warp_mesh_mask, warp_mesh, 1)
    lp_loss = 1.0 * lp_loss_2

    return lp_loss


def get_inter_grid_loss(overlap, mesh):

    ##############################
    # compute horizontal edges
    w_edges = mesh[:, :, 0:grid_w, :] - mesh[:, :, 1 : grid_w + 1, :]
    # 计算2条连续水平边的夹角
    cos_w = torch.sum(w_edges[:, :, 0 : grid_w - 1, :] * w_edges[:, :, 1:grid_w, :], 3) / (torch.sqrt(torch.sum(w_edges[:, :, 0 : grid_w - 1, :] * w_edges[:, :, 0 : grid_w - 1, :], 3)) * torch.sqrt(torch.sum(w_edges[:, :, 1:grid_w, :] * w_edges[:, :, 1:grid_w, :], 3)))
    # horizontal angle-preserving error for two successive horizontal edges
    delta_w_angle = 1 - cos_w
    # horizontal angle-preserving error for two successive horizontal grids
    delta_w_angle = delta_w_angle[:, 0:grid_h, :] + delta_w_angle[:, 1 : grid_h + 1, :]
    ##############################

    ##############################
    # compute vertical edges
    h_edges = mesh[:, 0:grid_h, :, :] - mesh[:, 1 : grid_h + 1, :, :]
    # compute angles of two successive vertical edges
    cos_h = torch.sum(h_edges[:, 0 : grid_h - 1, :, :] * h_edges[:, 1:grid_h, :, :], 3) / (torch.sqrt( torch.sum(h_edges[:, 0 : grid_h - 1, :, :] * h_edges[:, 0 : grid_h - 1, :, :], 3))* torch.sqrt( torch.sum(h_edges[:, 1:grid_h, :, :] * h_edges[:, 1:grid_h, :, :], 3)))
    # vertical angle-preserving error for two successive vertical edges
    delta_h_angle = 1 - cos_h
    # vertical angle-preserving error for two successive vertical grids
    delta_h_angle = delta_h_angle[:, :, 0:grid_w] + delta_h_angle[:, :, 1 : grid_w + 1]
    ##############################

    # on overlapping regions
    depth_diff_w = ( 1 - torch.abs(overlap[:, :, 0 : grid_w - 1] - overlap[:, :, 1:grid_w])) * overlap[:, :, 0 : grid_w - 1]
    error_w = depth_diff_w * delta_w_angle
    
    # on overlapping regions
    depth_diff_h = (1 - torch.abs(overlap[:, 0 : grid_h - 1, :] - overlap[:, 1:grid_h, :])) * overlap[:, 0 : grid_h - 1, :]
    error_h = depth_diff_h * delta_h_angle

    return torch.mean(error_w) + torch.mean(error_h)


# intra-grid constraint
def get_intra_grid_loss(pts):

    max_w = 512 / grid_w * 2
    max_h = 512 / grid_h * 2

    delta_x = pts[:, :, 1 : grid_w + 1, 0] - pts[:, :, 0:grid_w, 0]
    delta_y = pts[:, 1 : grid_h + 1, :, 1] - pts[:, 0:grid_h, :, 1]

    loss_x = F.relu(delta_x - max_w)
    loss_y = F.relu(delta_y - max_h)
    loss = torch.mean(loss_x) + torch.mean(loss_y)

    return loss
