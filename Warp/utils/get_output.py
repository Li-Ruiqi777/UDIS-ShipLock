import utils.torch_tps_transform as torch_tps_transform
import utils.torch_homo_transform as torch_homo_transform
import utils.torch_DLT as torch_DLT
from utils.misc import *
import utils.constant as constant

grid_h = constant.GRID_H
grid_w = constant.GRID_W
device = constant.device

def get_batch_outputs_for_train(net, ref_tensor, target_tensor, is_training=True):
    """
    用于训练和指标计算, 计算一个batch的输出, 输出图像的大小和输入图像一样

    1.用网络预测H的4pt和TPS的控制点的偏移
    2.计算mesh(的控制点)在网络预测的H和TPS motion后的坐标
    3.将target及其mask通过单应性矩阵变到reference坐标系下(同名像素点的在2图中一样)
    4.将reference及其mask通过单应性矩阵变到target坐标系下
    5.将target及其mask通过TPS变到reference坐标系下
    6.使用计算重叠区域,并用一个二值图表示重叠区和非重叠区----?有吗
    """
    batch_size, _, img_h, img_w = ref_tensor.size()

    if is_training == True:
        aug_input1_tensor, aug_input2_tensor = data_aug(ref_tensor, target_tensor)
        H_motion, mesh_motion = net(aug_input1_tensor, aug_input2_tensor)
    else:
        H_motion, mesh_motion = net(ref_tensor, target_tensor)

    H_motion = H_motion.reshape(-1, 4, 2)
    mesh_motion = mesh_motion.reshape(-1, grid_h + 1, grid_w + 1, 2)

    src_p = torch.FloatTensor([[0.0, 0.0], [img_w, 0.0], [0.0, img_h], [img_w, img_h]]).to(device)
    src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
    dst_p = src_p + H_motion

    H_mat = torch_DLT.tensor_DLT(src_p, dst_p)
    H_inv_mat = torch.inverse(H_mat)
    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    ini_mesh = convert_H_to_mesh(H_mat, rigid_mesh)
    mesh = ini_mesh + mesh_motion

    normalize_mat = torch.tensor(
        [
            [2.0/img_w, 0.0, -1.0],
            [0.0, 2.0/img_h, -1.0],
            [0.0, 0.0, 1.0],
        ]
    ).to(device)

    denormalize_mat = torch.tensor(
        [
            [img_w/2.0, 0.0, img_w/2.0],
            [0.0, img_h/2.0, img_h/2.0],
            [0.0, 0.0, 1.0],
        ]
    ).to(device)

    mask = torch.ones_like(target_tensor).to(device)

    denormalize_mat = denormalize_mat.unsqueeze(0).expand(batch_size, -1, -1)
    normalize_mat = normalize_mat.unsqueeze(0).expand(batch_size, -1, -1)

    H_mat = torch.matmul(torch.matmul(normalize_mat, H_mat), denormalize_mat)
    H_inv_mat = torch.matmul(torch.matmul(normalize_mat, H_inv_mat), denormalize_mat)   

    temp = torch.cat((target_tensor, mask), 1)
    warped_target_and_mask = torch_homo_transform.transformer(temp, H_mat, (img_h, img_w))

    temp = torch.cat((ref_tensor, mask), 1)
    warped_reference_and_mask = torch_homo_transform.transformer(temp, H_inv_mat, (img_h, img_w))

    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh = get_norm_mesh(mesh, img_h, img_w)

    temp = torch.cat((target_tensor, mask), 1)
    tps_warped_target_and_mask = torch_tps_transform.transformer(temp, norm_mesh, norm_rigid_mesh, (img_h, img_w))
    tps_warped_target = tps_warped_target_and_mask[:, 0:3, ...]
    tps_warped_mask = tps_warped_target_and_mask[:, 3:6, ...]

    # 计算重叠区域以施加shape-preserving约束
    overlap = torch_tps_transform.transformer(tps_warped_mask, norm_rigid_mesh, norm_mesh, (img_h, img_w))
    overlap = (
        overlap.permute(0, 2, 3, 1)
        .unfold(1, int(img_h / grid_h), int(img_h / grid_h))
        .unfold(2, int(img_w / grid_w), int(img_w / grid_w))
    )
    overlap = torch.mean(overlap.reshape(batch_size, grid_h, grid_w, -1), 3)
    overlap_one = torch.ones_like(overlap)
    overlap_zero = torch.zeros_like(overlap)
    overlap = torch.where(overlap < 0.9, overlap_one, overlap_zero)

    batch_outputs = {
        #[N, 6, 512, 512]
        'warped_target_and_mask':warped_target_and_mask,
        #[N, 6, 512, 512]
        'warped_reference_and_mask':warped_reference_and_mask,
        #[N, 3, 512, 512]
        'tps_warped_target':tps_warped_target,
        #[N, 3, 512, 512]
        'tps_warped_mask':tps_warped_mask,
        #[N, 13, 13, 2]
        'rigid_mesh':rigid_mesh,
        #[N, 13, 13, 2]
        'mesh':mesh,
        #[N, 12, 12]
        'overlap':overlap, # 在target系下的重叠区,0:重叠,1:非重叠
    }

    return batch_outputs

def get_batch_outputs_for_stitch(net, ref_tensor, target_tensor):
    """
    用于计算拼接时一个batch的输出, 输出图像的大小和最终拼接结果图像的大小一致
    由于是测试,所以输入图片的大小不一定得是512,但是推理之前会resize到512,结果再resize回原尺寸
    
    1.用网络预测H的4pt和TPS的控制点的偏移
    2.根据变换后控制点的坐标最值,确定stitched img的h,w大小
    3.对reference进行平移变换,将其变到stitched img的坐标系下
    4.对target进行TPS变换,将其变到stitched img的坐标系下
    """
    batch_size, _, img_h, img_w = ref_tensor.size()

    resized_ref_tensor = resize_512(ref_tensor)
    resized_target_tensor = resize_512(target_tensor)
    H_motion, mesh_motion = net(resized_ref_tensor, resized_target_tensor)

    H_motion = H_motion.reshape(-1, 4, 2)
    # 由于H是在(512,512)分辨率下训练的,这里先除以512再乘以当前的h,w是为了适应不同的分辨率
    H_motion = torch.stack([H_motion[..., 0] * img_w / 512, H_motion[..., 1] * img_h / 512], 2)
    mesh_motion = mesh_motion.reshape(-1, grid_h + 1, grid_w + 1, 2)
    mesh_motion = torch.stack([mesh_motion[..., 0] * img_w / 512, mesh_motion[..., 1] * img_h / 512], 3)

    src_p = torch.FloatTensor([[0.0, 0.0], [img_w, 0.0], [0.0, img_h], [img_w, img_h]]).to(device)
    src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
    dst_p = src_p + H_motion

    H_mat = torch_DLT.tensor_DLT(src_p, dst_p)

    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    ini_mesh = convert_H_to_mesh(H_mat, rigid_mesh)  # 使用H对网格上各控制点施加一个初始的偏移
    mesh = ini_mesh + mesh_motion

    # 根据控制点的坐标最值,确定stitched img的h,w大小
    width_max = torch.max(mesh[..., 0])
    width_max = torch.maximum(torch.tensor(img_w).cuda(), width_max)
    width_min = torch.min(mesh[..., 0])
    width_min = torch.minimum(torch.tensor(0).cuda(), width_min)
    height_max = torch.max(mesh[..., 1])
    height_max = torch.maximum(torch.tensor(img_h).cuda(), height_max)
    height_min = torch.min(mesh[..., 1])
    height_min = torch.minimum(torch.tensor(0).cuda(), height_min)

    # stitched img的h,w大小
    out_width = width_max - width_min
    out_height = height_max - height_min

    # print(f"x_max: {width_max}, x_min: {width_min}, y_max: {height_max}, y_min: {height_min}")
    # print(f"output_width: {out_width}, output_height: {out_height}")

    normalize_mat = torch.tensor(
        [
            [2.0/img_w, 0.0, -1.0],
            [0.0, 2.0/img_h, -1.0],
            [0.0, 0.0, 1.0],
        ]
    ).to(device)

    denormalize_mat = torch.tensor(
        [
            [out_width / 2.0, 0.0, out_width / 2.0],
            [0.0, out_height / 2.0, out_height / 2.0],
            [0.0, 0.0, 1.0],
        ]
    ).to(device)

    reference_translation_mat = torch.tensor(
        [
            [1.0, 0.0, width_min],
            [0.0, 1.0, height_min],
            [0.0, 0.0, 1.0]
         ]
    ).to(device)

    mask = torch.ones_like(target_tensor).to(device)     

    reference_translation_mat = torch.matmul(torch.matmul(normalize_mat, reference_translation_mat), denormalize_mat).unsqueeze(0)
    
    translated_reference_and_mask = torch_homo_transform.transformer(
        torch.cat((ref_tensor + 1, mask), 1),
        reference_translation_mat,
        (out_height.int(), out_width.int()),
    )

    torch.cuda.empty_cache()

    mesh_trans = torch.stack([mesh[..., 0] - width_min, mesh[..., 1] - height_min], 3)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh = get_norm_mesh(mesh_trans, out_height, out_width)
    
    tps_warped_target_and_mask = torch_tps_transform.transformer(
        torch.cat([target_tensor + 1, mask], 1),
        norm_mesh,
        norm_rigid_mesh,
        (out_height.int(), out_width.int()),
    )

    # out_dict = {}
    # out_dict.update(
    #     #[N, 3, output_height, output_width]
    #     final_warp1=translated_reference_and_mask[:, 0:3, ...] - 1, #将reference平移变换到stitched image的坐标系下
    #     #[N, 3, output_height, output_width]
    #     final_warp1_mask=translated_reference_and_mask[:, 3:6, ...],#将reference平移变换到stitched image的坐标系下的mask
    #     #[N, 3, output_height, output_width]
    #     final_warp2=tps_warped_target_and_mask[:, 0:3, ...] - 1,  #将target进行TPS变换到stitched image的坐标系下
    #     #[N, 3, output_height, output_width]
    #     final_warp2_mask=tps_warped_target_and_mask[:, 3:6, ...], #将target进行TPS变换到stitched image的坐标系下的mask
    #     mesh1=rigid_mesh,
    #     mesh2=mesh_trans,
    # )

    batch_outputs = {
        #[N, 3, output_height, output_width]
        'translated_reference': translated_reference_and_mask[:, 0:3, ...] - 1,
        #[N, 3, output_height, output_width]
        'translated_mask': translated_reference_and_mask[:, 3:6, ...] - 1,
        #[N, 3, output_height, output_width]
        'tps_warped_target': tps_warped_target_and_mask[:, 0:3, ...] - 1,
        #[N, 3, output_height, output_width]
        'tps_warped_mask': tps_warped_target_and_mask[:, 3:6, ...] - 1,
        'rigid_mesh': rigid_mesh,
        'mesh_trans': mesh_trans,
    }

    return batch_outputs

def get_batch_outputs_for_ft(net, ref_tensor, target_tensor):
    """
    计算在线Fine tuning中的初步配准结果
    与`get_batch_outputs_for_stitch`的区别:此函数没有改变将warp后的图片的(h,w).导致warp后一部分图像的缺失.
    并且也没进行normalize, denormalize等操作, 所以输入图像必须为512*512的.

    1.用网络预测H的4pt和TPS的控制点的偏移
    2.计算mesh(的控制点)在网络预测的H和TPS motion后的坐标
    3.带入(2)的数据对target进行TPS变换
    """
    batch_size, _, img_h, img_w = ref_tensor.size()

    H_motion, mesh_motion = net(ref_tensor, target_tensor)

    H_motion = H_motion.reshape(-1, 4, 2)
    # H_motion = torch.stack([H_motion[...,0]*img_w/512, H_motion[...,1]*img_h/512], 2)

    mesh_motion = mesh_motion.reshape(-1, grid_h + 1, grid_w + 1, 2)
    # mesh_motion = torch.stack([mesh_motion[...,0]*img_w/512, mesh_motion[...,1]*img_h/512], 3)

    # initialize the source points bs x 4 x 2
    src_p = torch.FloatTensor([[0.0, 0.0], [img_w, 0.0], [0.0, img_h], [img_w, img_h]]).to(device)
    src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
    # target points
    dst_p = src_p + H_motion
    # solve homo using DLT
    H = torch_DLT.tensor_DLT(src_p, dst_p)

    # 下面这一堆mesh的计算是用于TPS的
    # 首先生成原始均匀分布的控制点(rigid_mesh),然后根据H变换后得到的mesh_motion进行偏移得到mesh
    # 然后对rigid_mesh和mesh进行归一化,并进行TPS进行变换
    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    ini_mesh = convert_H_to_mesh(H, rigid_mesh)
    mesh = ini_mesh + mesh_motion

    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh = get_norm_mesh(mesh, img_h, img_w)

    mask = torch.ones_like(target_tensor).to(device)

    tps_warped_target_and_mask = torch_tps_transform.transformer(
        torch.cat((target_tensor, mask), 1), 
        norm_mesh,
        norm_rigid_mesh, 
        (img_h, img_w)
    )
    tps_warped_target = tps_warped_target_and_mask[:, 0:3, ...]
    tps_warped_mask = tps_warped_target_and_mask[:, 3:6, ...]

    batch_outputs = {}
    batch_outputs.update(
        tps_warped_target=tps_warped_target,
        tps_warped_mask=tps_warped_mask,
        rigid_mesh=rigid_mesh,
        mesh=mesh,
    )

    return batch_outputs

# 使用Fine tuning获得的优化后的mesh进行stitch操作
def get_stitched_result(ref_tensor, tar_tensor, rigid_mesh, mesh):
    """
    1.根据变换后控制点的坐标最值,确定stitched img的h,w大小
    2.对reference进行单应性变换(平移),将其变到stitched img的坐标系下
    3.对target进行TPS变换,将其变到stitched img的坐标系下
    4.对2张配准后的图片进行average fusion
    """
    batch_size, _, img_h, img_w = ref_tensor.size()

    # 根据实际的img size通过缩放调整控制点的坐标
    rigid_mesh = torch.stack([rigid_mesh[..., 0] * img_w / 512, rigid_mesh[..., 1] * img_h / 512], 3)
    mesh = torch.stack([mesh[..., 0] * img_w / 512, mesh[..., 1] * img_h / 512], 3)

    # 根据控制点的坐标最值,确定stitched img的h,w大小
    width_max = torch.max(mesh[..., 0])
    width_max = torch.maximum(torch.tensor(img_w).cuda(), width_max)
    width_min = torch.min(mesh[..., 0])
    width_min = torch.minimum(torch.tensor(0).cuda(), width_min)
    height_max = torch.max(mesh[..., 1])
    height_max = torch.maximum(torch.tensor(img_h).cuda(), height_max)
    height_min = torch.min(mesh[..., 1])
    height_min = torch.minimum(torch.tensor(0).cuda(), height_min)

    out_width = width_max - width_min
    out_height = height_max - height_min
    print(out_width)
    print(out_height)

    # 将reference变到stitched image的坐标系下(可以看成平移操作)
    translated_ref_img = torch.zeros([batch_size, 3, out_height.int(), out_width.int()]).cuda()
    translated_ref_img[
        :,
        :,
        int(torch.abs(height_min)) : int(torch.abs(height_min)) + img_h, #warp1的在新坐标系下的范围[abs(height_min), abs(height_min) + img_h]
        int(torch.abs(width_min)) : int(torch.abs(width_min)) + img_w,
    ] = (ref_tensor + 1) * 127.5

    translated_ref_mask = torch.zeros([batch_size, 3, out_height.int(), out_width.int()]).cuda()
    translated_ref_mask[
        :,
        :,
        int(torch.abs(height_min)) : int(torch.abs(height_min)) + img_h,
        int(torch.abs(width_min)) : int(torch.abs(width_min)) + img_w,
    ] = 255

    # get warped img2
    mesh_trans = torch.stack([mesh[..., 0] - width_min, mesh[..., 1] - height_min], 3)
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh = get_norm_mesh(mesh_trans, out_height, out_width)

    # 用TPS变换将target变到stitched image的坐标系下
    mask = torch.ones_like(tar_tensor).to(device)
    stitch_tps_out = torch_tps_transform.transformer(
        torch.cat([tar_tensor + 1, mask], 1),
        norm_mesh,
        norm_rigid_mesh,
        (out_height.int(), out_width.int()),
    )
    warped_tar_img = stitch_tps_out[:, 0:3, :, :] * 127.5
    warped_tar_mask = stitch_tps_out[:, 3:6, :, :] * 255

    # average fustion
    stitched = translated_ref_img * (translated_ref_img / (translated_ref_img + warped_tar_img + 1e-6)) \
               + warped_tar_img * (warped_tar_img / (translated_ref_img + warped_tar_img + 1e-6))

    stitched_mesh = draw_mesh_on_warp(stitched[0].cpu().detach().numpy().transpose(1, 2, 0), mesh_trans[0].cpu().detach().numpy(),)

    batch_outputs = {}
    batch_outputs.update(
        warp1=translated_ref_img,
        mask1=translated_ref_mask,
        warp2=warped_tar_img,
        mask2=warped_tar_mask,
        stitched=stitched,
        stitched_mesh=stitched_mesh,
    )

    return batch_outputs
