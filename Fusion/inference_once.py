import argparse
import torch
import os
import numpy as np
import cv2

from UDIS2 import build_model, Fusion
from utils.ImageSaver import ImageSaver
from utils.logger_config import *
from utils import constant

device = constant.device
logger = logging.getLogger(__name__)

@torch.no_grad()
def inference_once(args):

    os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    model = Fusion().to(device)

    # 加载权重
    check_point = torch.load(args.ckpt_path)
    logger.info(f"load model from {args.ckpt_path}!")
    model.load_state_dict(check_point["model"], strict=True)

    image_saver = ImageSaver(args.save_path)

    warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor = loadSingleData(args.img_path)

    # 计算融合结果
    batch_out = build_model(model, warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor)
    learned_mask1 = batch_out["learned_mask1"]
    learned_mask2 = batch_out["learned_mask2"]
    fusion_img = batch_out["stitched_image"]

    # 将不同部分绘制不同颜色
    s1 = ((warp1_tensor[0] + 1) * 127.5 * learned_mask1[0]).cpu().numpy().transpose(1, 2, 0)
    s2 = ((warp2_tensor[0] + 1) * 127.5 * learned_mask2[0]).cpu().detach().numpy().transpose(1, 2, 0)
    fusion_color = np.zeros((warp1_tensor.shape[2], warp1_tensor.shape[3], 3), np.uint8)
    fusion_color[..., 0] = s2[..., 0]
    fusion_color[..., 1] = s1[..., 1] * 0.5 + s2[..., 1] * 0.5
    fusion_color[..., 2] = s1[..., 2]

    fusion_img = ((fusion_img[0] + 1) * 127.5).cpu().numpy().transpose(1, 2, 0)
    learned_mask1 = (learned_mask1[0] * 255).cpu().numpy().transpose(1, 2, 0)
    learned_mask2 = (learned_mask2[0] * 255).cpu().numpy().transpose(1, 2, 0)

    # 保存结果
    image_saver.add_image('learned_mask1', learned_mask1)
    image_saver.add_image('learned_mask2', learned_mask2)
    image_saver.add_image('fusion', fusion_img)
    image_saver.add_image('fusion_color', fusion_color)

    image_saver.flush()
    logger.info(f"Inference Done!")

def loadSingleData(data_path):
    warp1 = cv2.imread(data_path + "warp1.jpg").astype(dtype=np.float32)
    warp1 = (warp1 / 127.5) - 1.0
    warp1 = np.transpose(warp1, [2, 0, 1])

    warp2 = cv2.imread(data_path + "warp2.jpg").astype(dtype=np.float32)
    warp2 = (warp2 / 127.5) - 1.0
    warp2 = np.transpose(warp2, [2, 0, 1])

    mask1 = cv2.imread(data_path + "mask1.jpg").astype(dtype=np.float32)
    mask1 = mask1 / 255
    mask1 = np.transpose(mask1, [2, 0, 1])

    mask2 = cv2.imread(data_path + "mask2.jpg").astype(dtype=np.float32)
    mask2 = mask2 / 255
    mask2 = np.transpose(mask2, [2, 0, 1])

    warp1_tensor = torch.from_numpy(warp1).unsqueeze(0).to(device)
    warp2_tensor = torch.from_numpy(warp2).unsqueeze(0).to(device)
    mask1_tensor = torch.from_numpy(mask1).unsqueeze(0).to(device)
    mask2_tensor = torch.from_numpy(mask2).unsqueeze(0).to(device)

    return warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--img_path", type=str, default="E:/DeepLearning/7_Stitch/UDIS2/Carpark-DHW/")
    parser.add_argument('--ckpt_path', type=str, default='F:/MasterGraduate/03-Code/UDIS-ShipLock/model/Fusion/UDIS2/epoch020_model.pth')
    parser.add_argument('--save_path', type=str, default='F:/MasterGraduate/03-Code/UDIS-ShipLock/results/Fusion/inference_once/')
    args = parser.parse_args()

    inference_once(args)
