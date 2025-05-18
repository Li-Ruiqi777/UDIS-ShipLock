import argparse
import numpy as np
import cv2
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
import torchvision.transforms as T

from UDIS2 import UDIS2
from UANet import UANet
from loss import get_overlap_loss_ft

from utils.ImageSaver import ImageSaver
from utils.get_output import get_stitched_result, get_batch_outputs_for_ft
from utils.logger_config import *
from utils import constant

device = constant.device
logger = logging.getLogger(__name__)

transform = T.Compose([
    T.ToTensor(),                     
    T.Lambda(lambda x: x * 2 - 1),  
    T.Resize((512, 512),  
             antialias=True)
])

def load_image_pair(img_path, img_name, device='cuda'):
    """优化后的图像加载函数"""
    # 读取图像
    input1 = cv2.imread(os.path.join(img_path, 'input1', img_name))
    input2 = cv2.imread(os.path.join(img_path, 'input2', img_name))
    
    input1_tensor = transform(input1).unsqueeze(0).to(device)
    input2_tensor = transform(input2).unsqueeze(0).to(device)
    
    return input1_tensor, input2_tensor

def train(args):
    model = UANet().to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=3e-5, betas=(0.9, 0.999), eps=1e-08)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    # 加载权重
    check_point = torch.load(args.ckpt_path)
    logger.info(f"load model from {args.ckpt_path}!")
    model.load_state_dict(check_point["model"], strict=True)
    # optimizer.load_state_dict(check_point['optimizer'])

    input1_tensor, input2_tensor = load_image_pair(args.img_path, args.img_name)

    loss_list = []

    image_saver = ImageSaver(args.save_path)

    for current_iter in range(1, args.max_iter):

        optimizer.zero_grad()

        batch_out = get_batch_outputs_for_ft(model, input1_tensor, input2_tensor)
        warp_mesh = batch_out['tps_warped_target']
        warp_mesh_mask = batch_out['tps_warped_mask']
        rigid_mesh = batch_out['rigid_mesh']
        mesh = batch_out['mesh']

        total_loss = get_overlap_loss_ft(input1_tensor, warp_mesh, warp_mesh_mask)
        total_loss.backward()
        loss_list.append(total_loss)

        print("Training: Iteration[{:0>3}/{:0>3}] Total Loss: {:.4f} lr={:.8f}".format(current_iter, args.max_iter, total_loss, optimizer.state_dict()['param_groups'][0]['lr']))
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3, norm_type=2)
        optimizer.step()
       
        # 初始拼接结果
        if current_iter == 1:
            with torch.no_grad():
                output = get_stitched_result(input1_tensor, input2_tensor, rigid_mesh, mesh)

            image_saver.add_image('before_optimization', output['stitched'][0].cpu().detach().numpy().transpose(1,2,0))
            image_saver.add_image('before_optimization_mesh', output['stitched_mesh'])

        if current_iter >= 4:
            # 结束迭代条件: 连续四次loss变化小于1e-4
            if (torch.abs(loss_list[current_iter-4] - loss_list[current_iter-3]) <= 1e-4 \
            and torch.abs(loss_list[current_iter-3] - loss_list[current_iter-2]) <= 1e-4 \
            and torch.abs(loss_list[current_iter-2] - loss_list[current_iter-1]) <= 1e-4) \
            or current_iter == args.max_iter - 1:
                
                with torch.no_grad():
                    output = get_stitched_result(input1_tensor, input2_tensor, rigid_mesh, mesh)

                image_saver.add_image(f"iter-{str(current_iter).zfill(3)}", output['stitched'][0].cpu().detach().numpy().transpose(1,2,0))
                image_saver.add_image(f"iter-{str(current_iter).zfill(3)}_mesh", output['stitched_mesh'])
                image_saver.add_image('warp1', output['warp1'][0].cpu().detach().numpy().transpose(1,2,0))
                image_saver.add_image('warp2', output['warp2'][0].cpu().detach().numpy().transpose(1,2,0))
                image_saver.add_image('mask1', output['mask1'][0].cpu().detach().numpy().transpose(1,2,0))
                image_saver.add_image('mask2', output['mask2'][0].cpu().detach().numpy().transpose(1,2,0))
                break

        scheduler.step()

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--max_iter', type=int, default=50)
    parser.add_argument('--ckpt_path', type=str, default="model/UDIS-ship/Warp/DSEL-FPN/epoch170_model.pth")
    parser.add_argument('--save_path', type=str, default='F:/MasterGraduate/03-Code/UDIS-ShipLock/results/Warp/Fine_tuning/')
    # parser.add_argument('--ckpt_path', type=str, default="model/UDIS-ship/Warp/UDIS-D-fine_tuning/epoch200_model.pth")
    parser.add_argument('--img_path', type=str, default='F:/imgs-purge/same_camera/resize512/')
    parser.add_argument('--img_name', type=str, default='000075.jpg')

    args = parser.parse_args()

    train(args)