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
from natsort import natsorted

from UDIS2 import UDIS2
from UANet import UANet
from dataset import *
from utils.get_output import get_batch_outputs_for_train
from utils.logger_config import *
from utils import constant

device = constant.device
logger = logging.getLogger(__name__)

@torch.no_grad()
def quantitative_analysis(args):

    os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    test_dataset = TestDataset(data_path=args.test_dataset_path, width=512, height=512)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        drop_last=False,
    )

    model = UANet().to(device)
    model.eval()

    # 加载权重
    check_point = torch.load(args.ckpt_path)
    logger.info(f"load model from {args.ckpt_path}!")
    model.load_state_dict(check_point["model"], strict=False)

    # for key in model.state_dict().keys():
    #     print(key)

    psnr_list = []
    ssim_list = []
    
    for i, batch_value in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Processing batches"):

        inpu1_tesnor = batch_value[0].float().to(device)
        inpu2_tesnor = batch_value[1].float().to(device)
    
        batch_outputs = get_batch_outputs_for_train(model, inpu1_tesnor, inpu2_tesnor, is_training=False)

        tps_warped_mask = batch_outputs["tps_warped_mask"]
        tps_warped_target = batch_outputs["tps_warped_target"]

        tps_warped_target_np = ((tps_warped_target[0] + 1) * 127.5).cpu().numpy().transpose(1, 2, 0)
        tps_warped_mask_np = tps_warped_mask[0].cpu().numpy().transpose(1, 2, 0)
        inpu1_np = ((inpu1_tesnor[0] + 1) * 127.5).cpu().numpy().transpose(1, 2, 0)
        
        # calculate psnr/ssim
        psnr = skimage.metrics.peak_signal_noise_ratio(
            inpu1_np * tps_warped_mask_np,
            tps_warped_target_np * tps_warped_mask_np,
            data_range=255,
        )
        ssim = skimage.metrics.structural_similarity(
            inpu1_np * tps_warped_mask_np,
            tps_warped_target_np * tps_warped_mask_np,
            data_range=255,
            multichannel=True,
            win_size=3
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
    parser.add_argument("--test_dataset_path", type=str, default="/root/data/lrq/008-UDIS-Ship/test",)
    parser.add_argument('--ckpt_folder', type=str, default='/root/data/lrq/UDIS-ShipLock/model/UDIS-ship/Warp/DESLBlock')
    parser.add_argument('--ckpt_path', type=str, default='')
    args = parser.parse_args()
    
    sorted_files = natsorted(os.listdir(args.ckpt_folder))
    sorted_files = sorted_files[-5:]
    for file in sorted_files:
        if file.endswith('.pth'):
            args.ckpt_path = os.path.join(args.ckpt_folder, file)
            quantitative_analysis(args)
