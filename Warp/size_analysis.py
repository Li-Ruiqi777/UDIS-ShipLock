import torch
from torchsummary import summary
from thop import profile


from UDIS2 import UDIS2
from UANet import UANet

from utils.logger_config import *
from utils import constant


device = constant.device
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    model = UANet().to(device)
    # 模型结构
    summary(model, [(3, 512, 512),(3, 512, 512)])
    
    # 参数量和计算量
    inpu1_tesnor = torch.randn(1, 3, 512, 512).to(device)
    inpu2_tesnor = torch.randn(1, 3, 512, 512).to(device)

    macs, params = profile(model, inputs=[inpu1_tesnor, inpu2_tesnor])
    logger.info("Number of Params: %.2f M" % (params / 1e6))
    logger.info("Number of FLOPs: %.2f G" % (macs / 1e9))