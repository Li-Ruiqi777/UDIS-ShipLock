import os
import random
import shutil

def generate_image_pair(dataset_dir, output_dir, start_idx):
    # 获取文件夹A中的所有图片文件
    image_files = [f for f in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, f))]
    
    # 确保有足够的图片可供选择
    num_images = len(image_files)
    if num_images < 2:
        print("文件夹A中的图片数量不足，无法生成数据集")
        return
    
    # 计算最大组合数 C(n, 2)
    max_pairs = num_images * (num_images - 1) // 2
    print(f"文件夹A中有 {num_images} 张图片，最多可以生成 {max_pairs} 对图片。")
    
    # 目标文件夹B和子文件夹如果不存在则创建
    input1_dir = os.path.join(output_dir, 'input1')
    input2_dir = os.path.join(output_dir, 'input2')
    if not os.path.exists(input1_dir):
        os.makedirs(input1_dir)
    if not os.path.exists(input2_dir):
        os.makedirs(input2_dir)
    
    # 生成所有可能的图片对
    all_pairs = []
    for i in range(num_images):
        for j in range(i + 1, num_images):
            all_pairs.append((image_files[i], image_files[j]))
    
    # 随机选择所有可能的图片对
    selected_pairs = random.sample(all_pairs, max_pairs)
    
    # 复制并重命名图片
    for pair in selected_pairs:
        input1, input2 = pair
        
        # 生成统一的文件名
        filename = f"{start_idx:06d}.jpg"  # 文件名格式：000001.jpg, 000002.jpg, ...

        # 定义复制到文件夹B的路径
        input1_dst = os.path.join(input1_dir, filename)
        input2_dst = os.path.join(input2_dir, filename)
        
        # 构建源文件路径
        input1_src = os.path.join(dataset_dir, input1)
        input2_src = os.path.join(dataset_dir, input2)

        # 复制文件到目标文件夹
        shutil.copy(input1_src, input1_dst)
        shutil.copy(input2_src, input2_dst)
        
        print(f"已从{dataset_dir}中选择图片：{input1} 和 {input2}")
        print(f"图片已复制到{input1_dir}和{input2_dir}，命名为{filename}")

        # 更新编号
        start_idx += 1

    return start_idx


if __name__ == '__main__':
    dataset_dir = 'F:/MasterGraduate/03-Code/PanoramicTracking/datasets/images/data5/'
    output_dir = 'E:/DeepLearning/0_DataSets/007-UDIS-D/training-ship/training/'
    start_idx = 1

    # 生成数据集
    start_idx = generate_image_pair(dataset_dir, output_dir, start_idx)
