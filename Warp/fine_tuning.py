"""
在任意图片上进行warp,会进行迭代以优化最终结果,可以看成对单对图片的Finetune
这个迭代优化主要是优化重叠区没完全对齐产生的鬼影,但不能完全消除,如果鬼影实在是太大,应该用seamcut
"""

import argparse
import torch

import numpy as np
import os
import torch.nn as nn
import torch.optim as optim

import cv2

from UDIS2 import UDIS2
from utils.get_output import get_stitched_result, get_batch_outputs_for_ft

import glob
from loss import get_overlap_loss_ft
import torchvision.transforms as T

# import PIL
resize_512 = T.Resize((512, 512), antialias=True)


def loadSingleData(data_path, img1_name, img2_name):
    """
    将图片中的像素归一化到[-1,1],并转成tensor
    """
    # load image1
    input1 = cv2.imread(data_path + img1_name)
    # 缩小input1的尺寸
    input1 = cv2.resize(input1, (800, 600))
    # input1 = cv2.resize(input1, (512, 512))
    input1 = input1.astype(dtype=np.float32)
    input1 = (input1 / 127.5) - 1.0
    input1 = np.transpose(input1, [2, 0, 1])

    # load image2
    input2 = cv2.imread(data_path + img2_name)
    input2 = cv2.resize(input2, (800, 600))
    # input2 = cv2.resize(input2, (512, 512))
    input2 = input2.astype(dtype=np.float32)
    input2 = (input2 / 127.5) - 1.0
    input2 = np.transpose(input2, [2, 0, 1])

    # convert to tensor
    input1_tensor = torch.tensor(input1).unsqueeze(0)
    input2_tensor = torch.tensor(input2).unsqueeze(0)
    return (input1_tensor, input2_tensor)


# path of project
# nl: os.path.dirname("__file__") ----- the current absolute path
# nl: os.path.pardir ---- the last path
last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))


# nl: path to save the model files
MODEL_DIR = "E:/DeepLearning/7_Stitch/UDIS2/Warp/model"

# nl: create folders if it dose not exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


def train(args):

    os.environ["CUDA_DEVICES_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # define the network
    net = UDIS2()
    if torch.cuda.is_available():
        net = net.cuda()

    # define the optimizer and learning rate
    optimizer = optim.Adam(
        net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08
    )  # default as 0.0001
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    # load the existing models if it exists
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()
    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)

        net.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        scheduler.last_epoch = start_epoch
        print("load model from {}!".format(model_path))
    else:
        start_epoch = 0
        print("training from stratch!")

    # load dataset(only one pair of images)
    input1_tensor, input2_tensor = loadSingleData(
        data_path=args.path, img1_name=args.img1_name, img2_name=args.img2_name
    )
    if torch.cuda.is_available():
        input1_tensor = input1_tensor.cuda()
        input2_tensor = input2_tensor.cuda()

    input1_tensor_512 = resize_512(input1_tensor)
    input2_tensor_512 = resize_512(input2_tensor)

    loss_list = []

    print("##################start iteration#######################")
    for epoch in range(start_epoch, start_epoch + args.max_iter):
        net.train()

        optimizer.zero_grad()

        batch_out = get_batch_outputs_for_ft(net, input1_tensor_512, input2_tensor_512)
        warp_mesh = batch_out["warp_mesh"]
        warp_mesh_mask = batch_out["warp_mesh_mask"]
        rigid_mesh = batch_out["rigid_mesh"]
        mesh = batch_out["mesh"]

        total_loss = get_overlap_loss_ft(input1_tensor_512, warp_mesh, warp_mesh_mask)
        total_loss.backward()
        # clip the gradient
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
        optimizer.step()

        current_iter = epoch - start_epoch + 1
        print(
            "Training: Iteration[{:0>3}/{:0>3}] Total Loss: {:.4f} lr={:.8f}".format(
                current_iter,
                args.max_iter,
                total_loss,
                optimizer.state_dict()["param_groups"][0]["lr"],
            )
        )

        loss_list.append(total_loss)

        if current_iter == 1:
            with torch.no_grad():
                output = get_stitched_result(
                    input1_tensor, input2_tensor, rigid_mesh, mesh
                )
            cv2.imwrite(
                args.path + "before_optimization.jpg",
                output["stitched"][0].cpu().detach().numpy().transpose(1, 2, 0),
            )
            cv2.imwrite(
                args.path + "before_optimization_mesh.jpg", output["stitched_mesh"]
            )

        if current_iter >= 4:
            if (
                torch.abs(loss_list[current_iter - 4] - loss_list[current_iter - 3])
                <= 1e-4
                and torch.abs(loss_list[current_iter - 3] - loss_list[current_iter - 2])
                <= 1e-4
                and torch.abs(loss_list[current_iter - 2] - loss_list[current_iter - 1])
                <= 1e-4
            ):
                with torch.no_grad():
                    output = get_stitched_result(
                        input1_tensor, input2_tensor, rigid_mesh, mesh
                    )

                path = (
                    args.path + "iter-" + str(epoch - start_epoch + 1).zfill(3) + ".jpg"
                )
                cv2.imwrite(
                    path,
                    output["stitched"][0].cpu().detach().numpy().transpose(1, 2, 0),
                )
                cv2.imwrite(
                    args.path
                    + "iter-"
                    + str(epoch - start_epoch + 1).zfill(3)
                    + "_mesh.jpg",
                    output["stitched_mesh"],
                )
                cv2.imwrite(
                    args.path + "warp1.jpg",
                    output["warp1"][0].cpu().detach().numpy().transpose(1, 2, 0),
                )
                cv2.imwrite(
                    args.path + "warp2.jpg",
                    output["warp2"][0].cpu().detach().numpy().transpose(1, 2, 0),
                )
                cv2.imwrite(
                    args.path + "mask1.jpg",
                    output["mask1"][0].cpu().detach().numpy().transpose(1, 2, 0),
                )
                cv2.imwrite(
                    args.path + "mask2.jpg",
                    output["mask2"][0].cpu().detach().numpy().transpose(1, 2, 0),
                )
                break

        if current_iter == args.max_iter:
            with torch.no_grad():
                output = get_stitched_result(
                    input1_tensor, input2_tensor, rigid_mesh, mesh
                )

            path = args.path + "iter-" + str(epoch - start_epoch + 1).zfill(3) + ".jpg"
            cv2.imwrite(
                path, output["stitched"][0].cpu().detach().numpy().transpose(1, 2, 0)
            )
            cv2.imwrite(
                args.path
                + "iter-"
                + str(epoch - start_epoch + 1).zfill(3)
                + "_mesh.jpg",
                output["stitched_mesh"],
            )
            cv2.imwrite(
                args.path + "warp1.jpg",
                output["warp1"][0].cpu().detach().numpy().transpose(1, 2, 0),
            )
            cv2.imwrite(
                args.path + "warp2.jpg",
                output["warp2"][0].cpu().detach().numpy().transpose(1, 2, 0),
            )
            cv2.imwrite(
                args.path + "mask1.jpg",
                output["mask1"][0].cpu().detach().numpy().transpose(1, 2, 0),
            )
            cv2.imwrite(
                args.path + "mask2.jpg",
                output["mask2"][0].cpu().detach().numpy().transpose(1, 2, 0),
            )

        scheduler.step()

    print("##################end iteration#######################")


if __name__ == "__main__":

    print("<==================== setting arguments ===================>\n")

    # nl: create the argument parser
    parser = argparse.ArgumentParser()

    # nl: add arguments
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument(
        "--path", type=str, default="F:/MasterGraduate/03-Code/PanoramicTracking/datasets/images/data5/"
    )
    parser.add_argument("--img1_name", type=str, default="3.jpg")
    parser.add_argument("--img2_name", type=str, default="1.jpg")

    # nl: parse the arguments
    args = parser.parse_args()
    print(args)

    # nl: rain
    train(args)
