"""
在UDIS-D数据集上进行测试Fusion(输入是warp后的图像和mask)
"""
import torch
from torch.utils.data import DataLoader
import argparse

from UDIS2 import build_model, Fusion
from dataset import *
from utils.ImageSaver import ImageSaver
from utils.logger_config import *
from utils import constant

device = constant.device
logger = logging.getLogger(__name__)

@torch.no_grad()
def test_fusion(args):

    os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    test_data = TestDataset(data_path=args.test_dataset_path)
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
    )

    model = Fusion().to(device)
    model.eval()

    # 加载权重
    check_point = torch.load(args.ckpt_path)
    logger.info(f"load model from {args.ckpt_path}!")
    model.load_state_dict(check_point["model"], strict=True)

    image_saver = ImageSaver(args.save_path)

    for idx, batch_value in enumerate(test_loader):

        warp1_tensor = batch_value[0].float().to(device)
        warp2_tensor = batch_value[1].float().to(device)
        mask1_tensor = batch_value[2].float().to(device)
        mask2_tensor = batch_value[3].float().to(device)

        batch_out = build_model(model, warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor)

        learned_mask1 = batch_out["learned_mask1"].cpu().numpy().transpose(0, 2, 3, 1)
        learned_mask2 = batch_out["learned_mask2"].cpu().numpy().transpose(0, 2, 3, 1)
        fusion_img = batch_out["stitched_image"].cpu().numpy().transpose(0, 2, 3, 1)

        learned_mask1 = de_normalize1(learned_mask1)
        learned_mask2 = de_normalize1(learned_mask2)
        fusion_img = de_normalize(fusion_img)

        # 保存结果
        image_saver.add_image(f'learned_mask1/{str(idx + 1).zfill(6)}', learned_mask1[0])
        image_saver.add_image(f'learned_mask2/{str(idx + 1).zfill(6)}', learned_mask2[0])
        image_saver.add_image(f'fusion/{str(idx + 1).zfill(6)}', fusion_img[0])
    
        image_saver.flush()

        logger.info("idx = {}".format(idx + 1))

def de_normalize(img):
    '''
    将图像数据从[-1, 1]归一化到[0, 255]
    还原归一化
    '''
    img = (img + 1.0) * 127.5
    img = img.astype(np.uint8)
    return img

def de_normalize1(img):
    '''
    用于还原mask的归一化
    '''
    img = (img) * 255
    img = img.astype(np.uint8)
    return img

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--test_dataset_path",type=str, default="F:/imgs-purge/same_camera/test")
    parser.add_argument('--ckpt_path', type=str, default='F:/MasterGraduate/03-Code/UDIS-ShipLock/model/Fusion/UDIS2/epoch050_model.pth')
    parser.add_argument('--save_path', type=str, default='F:/MasterGraduate/03-Code/UDIS-ShipLock/results/Fusion')
    args = parser.parse_args()
    
    test_fusion(args)
