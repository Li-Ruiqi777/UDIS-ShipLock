import cv2
import numpy as np
import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from UDIS2 import UDIS2
from utils.get_output import get_batch_outputs_for_stitch
from utils.ImageSaver import ImageSaver

ckpt_path = "E:/DeepLearning/7_Stitch/UDIS2/Warp/model/epoch100_model.pth"
save_path = 'E:/DeepLearning/7_Stitch/UDIS2/Warp/results/test_stitch'

@torch.no_grad()
def test_get_batch_outputs_for_stitch():
    model = UDIS2()
    model = model.cuda()

    # 加载权重
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # 加载输入
    ref_img = cv2.imread("E:/DeepLearning/0_DataSets/007-UDIS-D/testing/testing/input1/000001.jpg")
    target_img = cv2.imread("E:/DeepLearning/0_DataSets/007-UDIS-D/testing/testing/input2/000001.jpg")
    # 归一化
    ref_img = ref_img.astype(np.float32) / 127.5 - 1.0
    target_img = target_img.astype(np.float32) / 127.5 - 1.0
    
    ref_img = np.transpose(ref_img, (2, 0, 1))
    target_img = np.transpose(target_img, (2, 0, 1))
    
    target_tensor = torch.from_numpy(target_img).unsqueeze(0).cuda()
    ref_tensor = torch.from_numpy(ref_img).unsqueeze(0).cuda()

    # 推理
    batch_output = get_batch_outputs_for_stitch(model, ref_tensor, target_tensor)
        
    # 保存结果
    image_saver = ImageSaver(save_path)

    translated_reference = batch_output['translated_reference'].cpu().numpy().transpose(0, 2, 3, 1)
    translated_mask = batch_output['translated_mask'].cpu().numpy().transpose(0, 2, 3, 1)

    tps_warped_target = batch_output['tps_warped_target'].cpu().numpy().transpose(0, 2, 3, 1)
    tps_warped_mask = batch_output['tps_warped_mask'].cpu().numpy().transpose(0, 2, 3, 1)

    image_saver.add_image('translated_reference', de_normalize(translated_reference[0]))
    image_saver.add_image('translated_mask', de_normalize(translated_mask[0]))
    image_saver.add_image('tps_warped_target', de_normalize(tps_warped_target[0]))
    image_saver.add_image('tps_warped_mask', de_normalize(tps_warped_mask[0]))


    image_saver.flush()

def de_normalize(img):
    '''
    将图像数据从[-1, 1]归一化到[0, 255]
    还原归一化
    '''
    img = (img + 1.0) * 127.5
    img = img.astype(np.uint8)
    return img

    
if __name__ == '__main__':
    test_get_batch_outputs_for_stitch()
    