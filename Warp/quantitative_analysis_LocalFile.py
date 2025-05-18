"""
计算PSNR和SSIM
"""
import torch
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np
import skimage
from tqdm import tqdm

from dataset import *
from utils.logger_config import *
from utils import constant
from utils.warp_tradition import *

device = constant.device
logger = logging.getLogger(__name__)

@torch.no_grad()
def quantitative_analysis(args):
    test_dataset = TestDataset(data_path=args.test_dataset_path, resize=False, width=512, height=512)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        drop_last=False,
    )

    psnr_list = []
    ssim_list = []
    
    for i, batch_value in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Processing batches"):

        inpu1_tesnor = batch_value[0].float()
        inpu2_tesnor = batch_value[1].float()
       
        inpu1_np = ((inpu1_tesnor[0] + 1) * 127.5).numpy().transpose(1, 2, 0)
        inpu2_np = ((inpu2_tesnor[0] + 1) * 127.5).numpy().transpose(1, 2, 0)

        # 计算PSNR/SSIM
        psnr = skimage.metrics.peak_signal_noise_ratio(
            inpu1_np,
            inpu2_np,
            data_range=255,
        )

        ssim = skimage.metrics.structural_similarity(
            inpu1_np,
            inpu2_np,
            channel_axis=2,
            data_range=255,
        )

        # logger.info(f"i = {i+1}, psnr = {psnr:.6f}")

        psnr_list.append(psnr)
        ssim_list.append(ssim)

    total_image_nums = len(test_dataset)
    imgs_0_30 = int(total_image_nums * 0.3)
    imgs_30_60 = int(total_image_nums * 0.6)
    logger.info(f"totoal image nums: {total_image_nums}")

    logger.info("--------------------- PSNR ---------------------")
    psnr_list.sort(reverse=True)
    psnr_list_30 = psnr_list[0:imgs_0_30]
    psnr_list_60 = psnr_list[imgs_0_30:imgs_30_60]
    psnr_list_100 = psnr_list[imgs_30_60:-1]
    
    logger.info(f"top 30%: {np.mean(psnr_list_30):.6f}")
    logger.info(f"top 30~60%: {np.mean(psnr_list_60):.6f}")
    logger.info(f"top 60~100%: {np.mean(psnr_list_100):.6f}")
    logger.info(f"average psnr: {np.mean(psnr_list):.6f}")

    logger.info("--------------------- SSIM ---------------------")
    ssim_list.sort(reverse=True)
    ssim_list_30 = ssim_list[0:imgs_0_30]
    ssim_list_60 = ssim_list[imgs_0_30:imgs_30_60]
    ssim_list_100 = ssim_list[imgs_30_60:-1]

    logger.info(f"top 30%: {np.mean(ssim_list_30):.6f}")
    logger.info(f"top 30~60%: {np.mean(ssim_list_60):.6f}")
    logger.info(f"top 60~100%: {np.mean(ssim_list_100):.6f}")
    logger.info(f"average ssim: {np.mean(ssim_list):.6f}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=1)
    # parser.add_argument("--test_dataset_path", type=str, default="E:/DeepLearning/0_DataSets/008-UDIS-Ship/test/ELA_output")
    parser.add_argument("--test_dataset_path", type=str, default="E:/DeepLearning/0_DataSets/008-UDIS-Ship/test/SPW_output")
    parser.add_argument('--ckpt_folder', type=str, default='')
    parser.add_argument('--ckpt_path', type=str, default='')
    args = parser.parse_args()
    
    quantitative_analysis(args)
