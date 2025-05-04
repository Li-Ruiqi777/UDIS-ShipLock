"""
计算soft-coded seam quality
"""
import argparse
import os
import numpy as np
import skimage
from tqdm import tqdm

from UDIS2 import Fusion
from dataset import *
from utils.logger_config import *
from utils import constant

device = constant.device
logger = logging.getLogger(__name__)

def quantitative_analysis(args):

    img_name_list = list(sorted(os.listdir(os.path.join(args.warp_result_path, 'warp1'))))
    
    data_range = 255
    k_size = 5
    l_num = 2
    metirc_model = SeamQuality(k_size, data_range, l_num)
    sq_list = []

    for i in tqdm(range(len(img_name_list))):
        mask1_path = os.path.join(args.fusion_result_path, 'learned_mask1/' + img_name_list[i])
        mask2_path = os.path.join(args.fusion_result_path, 'learned_mask2/' + img_name_list[i])
        fusion_img_path = os.path.join(args.fusion_result_path, 'fusion/' + img_name_list[i])

        img1_path = os.path.join(args.warp_result_path, 'warp1/' + img_name_list[i])
        img2_path = os.path.join(args.warp_result_path, 'warp2/' + img_name_list[i])

        fusion_img = cv2.imread(fusion_img_path)
        mask1 = cv2.imread(mask1_path)
        mask2 = cv2.imread(mask2_path)
        
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        # resize到128x128
        # img1 = cv2.resize(img1, (128, 128))
        # img2 = cv2.resize(img2, (128, 128))
        # fusion_img = cv2.resize(fusion_img, (128, 128))
        # mask1 = cv2.resize(mask1, (128, 128))
        # mask2 = cv2.resize(mask2, (128, 128))

        sq, dst1 = metirc_model.forward(img1, img2, fusion_img, mask1, mask2)
        sq_list.append(sq)

        # logger.info("i = {}, image name = {}, seam quality = {:.4f}".format(i + 1, img_name_list[i], sq))

    total_image_nums = len(img_name_list)
    imgs_0_30 = int(total_image_nums * 0.3)
    imgs_30_60 = int(total_image_nums * 0.6)
    logger.info(f"totoal image nums: {total_image_nums}")

    sq_list.sort(reverse=False)
    sq_list_30 = sq_list[0: imgs_0_30]
    sq_list_60 = sq_list[imgs_0_30: imgs_30_60]
    sq_list_100 = sq_list[imgs_30_60: -1]

    logger.info("--------------------- Seam Quality ---------------------")
    logger.info(f"top 30%: {np.mean(sq_list_30):.6f}")
    logger.info(f"top 30~60%: {np.mean(sq_list_60):.6f}")
    logger.info(f"top 60~100%: {np.mean(sq_list_100):.6f}")
    logger.info(f'average seam quality: {np.mean(sq_list):.6f}')

class SeamQuality():
    def __init__(self, k_size_lp=5, k_size_ssim=5, data_range=255., l_num=2):
        super().__init__()
        self.k_size_ssim = k_size_ssim
        self.k_size_lp = k_size_lp
        self.data_range = data_range
        self.l_num = l_num

    def intensity_loss(self, gen_frames, gt_frames, l_num=1):
        return np.mean(np.abs((gen_frames - gt_frames) ** l_num))

    def get_mask(self, mask1, mask2, k_size=5):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        dst1 = cv2.dilate(mask1, kernel) # 膨胀操作
        dst2 = cv2.dilate(mask2, kernel)
        # 计算拼接缝的轮廓
        dst = cv2.bitwise_and(dst1, dst2)
        thresh, dst = cv2.threshold(dst, 127.5, 255, cv2.THRESH_BINARY)
        # cv2.imshow('dst', dst)
        # cv2.waitKey(0)

        dst = dst / 255.
        return dst

    def local_consistent_ssim(self, input1, input2, stitchedImg, dst):
        # 取出各图中与拼接缝的重叠区域
        img1_m = input1 * dst
        img2_m = input2 * dst
        stitchedImg_m = stitchedImg * dst

        ssim1 = skimage.metrics.structural_similarity(img1_m, stitchedImg_m, data_range=self.data_range, channel_axis=2)
        ssim2 = skimage.metrics.structural_similarity(img2_m, stitchedImg_m, data_range=self.data_range, channel_axis=2)
        metirc = (2 - ssim1 - ssim2) / 2.0
        return metirc

    def local_consistent(self, input1, input2, stitchedImg, dst):
        # 取出各图中与拼接缝的重叠区域
        img1_m = input1 * dst
        img2_m = input2 * dst
        stitchedImg = stitchedImg * dst

        # cv2.imshow('dst', dst)
        # cv2.imshow('input1', input1)
        # cv2.imshow('input2', input2)
        # cv2.imshow('img1_m', img1_m)
        # cv2.imshow('img2_m', img2_m)
        # cv2.imshow('stitchedImg', stitchedImg)
        # cv2.waitKey(0)

        lp_1 = self.intensity_loss(img1_m, stitchedImg, l_num=self.l_num)
        lp_2 = self.intensity_loss(img2_m, stitchedImg, l_num=self.l_num)
        metirc = (lp_1 + lp_2) / 2.0
        return metirc
    
    def forward(self, input1, input2, stitchedImg, mask1, mask2):
        # step 1: get ssim mask
        dst = self.get_mask(mask1, mask2, self.k_size_ssim)

        # step 2: global sturcture consistent
        metirc1 = self.local_consistent_ssim(input1, input2, stitchedImg, dst)

        # step 3: local seam field consistent
        metirc2 = self.local_consistent(input1, input2, stitchedImg, dst)

        # step 4: combine global and local
        metirc = metirc1 * 1000 + metirc2

        # logger.info(f"metirc1: {metirc1:.6f}, metirc2: {metirc2:.6f}, metirc: {metirc:.6f}")

        return metirc, dst


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--warp_result_path', type=str, default='E:/DeepLearning/0_DataSets/007-UDIS-D-subset/test')
    parser.add_argument('--fusion_result_path', type=str, default='F:/MasterGraduate/03-Code/UDIS-ShipLock/results/Fusion/UDIS2-20')
    args = parser.parse_args()
    
    quantitative_analysis(args)
