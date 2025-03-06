import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import numpy as np
import random

from loss import get_overlap_loss, get_inter_grid_loss, get_intra_grid_loss
from UDIS2 import UDIS2
from utils.get_output import get_batch_outputs_for_train
from dataset import TrainDataset
from utils.logger_config import *
from utils import constant
from UANet import UANet

device = constant.device
logger = logging.getLogger(__name__)

def train(args):
    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # 定义数据集
    train_dataset = TrainDataset(data_path=args.train_dataset_path)
    train_dataloader = DataLoader(dataset=train_dataset, 
                                  batch_size=args.batch_size, 
                                  num_workers=args.num_workers, 
                                  shuffle=True, 
                                  pin_memory=True)

    # 定义网络模型
    model = UANet().to(device)
    model.train()

    # 定义优化器和学习率
    target_lr = 1e-4
    initial_lr = 5e-5
    optimizer = optim.AdamW(model.parameters(), lr=target_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    # 加载已有模型权重
    if(args.resume):
        check_point = torch.load(args.ckpt_path)
        model.load_state_dict(check_point["model"])
        optimizer.load_state_dict(check_point["optimizer"])
        start_epoch = check_point["epoch"]
        current_iter = check_point["glob_iter"]
        scheduler.last_epoch = start_epoch
    
        logger.info(f"load model from {args.ckpt_path}!")
        logger.info(f"start epoch {start_epoch}")

    else:
        start_epoch = 0
        current_iter = 0
        logger.info('training from stratch!')
    
    # 定义tensorboard
    tensorboard_writer = SummaryWriter(log_dir=args.tensorboard_save_folder)

    average_total_loss = 0
    average_overlap_loss = 0
    average_nonoverlap_loss = 0

    # 开始训练
    for epoch in range(start_epoch, args.max_epoch):
        if epoch < args.warmup_epoch:
            lr = initial_lr + (target_lr - initial_lr) * (epoch / args.warmup_epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        for idx, batch_value in enumerate(train_dataloader):

            inpu1_tesnor = batch_value[0].float().to(device)
            inpu2_tesnor = batch_value[1].float().to(device)
            
            optimizer.zero_grad()

            batch_outputs = get_batch_outputs_for_train(model, inpu1_tesnor, inpu2_tesnor)

            mesh = batch_outputs['mesh']
            overlap = batch_outputs['overlap']

            # 计算重叠区域的损失
            overlap_loss = get_overlap_loss(inpu1_tesnor, inpu2_tesnor, batch_outputs)
            
            # 计算非重叠区域的损失
            nonoverlap_loss = 10 * get_inter_grid_loss(overlap, mesh) + 10 * get_intra_grid_loss(mesh)

            total_loss = overlap_loss + nonoverlap_loss
            total_loss.backward()

            # 裁剪梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3, norm_type=2)
            # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1)
            optimizer.step()

            average_overlap_loss += overlap_loss.item()
            average_nonoverlap_loss += nonoverlap_loss.item()
            average_total_loss += total_loss.item()

            if current_iter % args.print_log_interval == 0 and current_iter != 0:
                average_loss = average_total_loss / args.print_log_interval
                average_overlap_loss = average_overlap_loss / args.print_log_interval
                average_nonoverlap_loss = average_nonoverlap_loss / args.print_log_interval

                logger.info(f"Epoch[{epoch + 1}/{args.max_epoch}] "
                            f"Iter[{current_iter % len(train_dataloader)}/{len(train_dataloader)}] - "
                            f"Loss: {average_loss:.4f}  "
                            f"Overlap Loss: {average_overlap_loss:.4f}  "
                            f"Nonoverlap Loss: {average_nonoverlap_loss:.4f}  "
                            f"LR: {optimizer.state_dict()['param_groups'][0]['lr']:.8f}  "
                        )
                
                if current_iter % args.tensorboard_log_interval == 0:
                    # tensorboard_writer.add_image("inpu1", (inpu1_tesnor[0]+1.)/2., glob_iter)
                    # tensorboard_writer.add_image("inpu2", (inpu2_tesnor[0]+1.)/2., glob_iter)
                    # tensorboard_writer.add_image("warp_H", (output_H[0,0:3,:,:]+1.)/2., glob_iter)
                    # tensorboard_writer.add_image("warp_mesh", (warp_mesh[0]+1.)/2., glob_iter)
                    tensorboard_writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], current_iter)
                    tensorboard_writer.add_scalar('total loss', average_loss, current_iter)
                    tensorboard_writer.add_scalar('overlap loss', average_overlap_loss, current_iter)
                    tensorboard_writer.add_scalar('nonoverlap loss', average_nonoverlap_loss, current_iter)
                
                average_total_loss = 0
                average_overlap_loss = 0
                average_nonoverlap_loss = 0

            current_iter += 1

        if epoch >= args.warmup_epoch:
            scheduler.step()

        # 保存模型
        if ((epoch + 1) % args.save_epoch_interval == 0 or (epoch + 1) == args.max_epoch):
            filename ='epoch' + str(epoch+1).zfill(3) + '_model.pth'
            model_save_path = os.path.join(args.model_save_folder, filename)
            state = {
                'model': model.state_dict(), 
                'optimizer': optimizer.state_dict(), 
                'epoch': epoch + 1, 
                "glob_iter": current_iter
            }
            torch.save(state, model_save_path)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

if __name__=="__main__":
    setup_seed(200147)
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--save_epoch_interval', type=int, default=10)
    parser.add_argument('--print_log_interval', type=int, default=20)
    parser.add_argument('--tensorboard_log_interval', type=int, default=100)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--warmup_epoch', type=int, default=10)
    parser.add_argument('--train_dataset_path', type=str, default='E:/DeepLearning/0_DataSets/007-UDIS-D-subset/train')
    parser.add_argument('--ckpt_path', type=str, default='E:/DeepLearning/7_Stitch/UDIS2/Warp/model/epoch100_model.pth')
    parser.add_argument('--model_save_folder', type=str, default='F:/MasterGraduate/03-Code/UDIS-ShipLock/model/Warp/')
    parser.add_argument('--tensorboard_save_folder', type=str, default='F:/MasterGraduate/03-Code/UDIS-ShipLock/summary/Warp/')
    args = parser.parse_args()

    if(not os.path.exists(args.tensorboard_save_folder)):
        os.makedirs(args.tensorboard_save_folder)
        
    if(not os.path.exists(args.model_save_folder)):
        os.makedirs(args.model_save_folder)

    train(args)

