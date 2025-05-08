import cv2
import numpy as np

def compute_homography_SIFT(ref_img, tar_img, visualize=False):
    """
    计算从目标图像到参考图像的单应性矩阵H (使用SIFT特征)
    参数：
        ref_img: 参考图像(BGR格式)
        tar_img: 目标图像(BGR格式)
    返回：
        warped_target_img: 变换后的目标图像
        warped mask: 变换后的mask(用于求重叠部分)
    """
    # 转换为灰度图
    img_target_gray = cv2.cvtColor(tar_img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    img_reference_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY).astype(np.uint8)

    # 初始化SIFT检测器
    sift = cv2.SIFT_create()
    
    # 检测关键点和计算描述符
    keypoints_target, descriptors_target = sift.detectAndCompute(img_target_gray, None)
    keypoints_reference, descriptors_reference = sift.detectAndCompute(img_reference_gray, None)

    # 使用FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 进行特征匹配
    matches = flann.knnMatch(descriptors_reference, descriptors_target, k=2)

    # Lowe's比率测试筛选优质匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 1.0 * n.distance: # 0.7
            good_matches.append(m)
    H = None
    if len(good_matches) < 4:
        H = np.eye(3, dtype=np.float32)
        print("Not enough matches are found - %d/%d" % (len(good_matches), 4))
    else:
        # 提取匹配点坐标（注意方向：目标图->参考图）
        points_target = np.float32([keypoints_target[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        points_reference = np.float32([keypoints_reference[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # 计算单应性矩阵（RANSAC方法）
        H, _ = cv2.findHomography(points_target, points_reference, cv2.RANSAC, 5.0)

    if H is None:
        H = np.eye(3, dtype=np.float32)
    # 应用透视变换（保持彩色）
    h, w = ref_img.shape[:2]
    warped_target_img = cv2.warpPerspective(tar_img, H, (w, h))

    # 创建全1 mask并进行warp
    mask = np.ones_like(tar_img, dtype=np.uint8)
    warped_mask = cv2.warpPerspective(mask, H, (w, h))
    
    if visualize:
        # 显示结果
        cv2.imshow("Warped Target Image", warped_target_img)
        cv2.imshow("Reference Image", ref_img)
        cv2.imshow("Warped Mask", warped_mask*255)
        cv2.imshow("overlap", warped_mask * ref_img)
        cv2.waitKey(0)

    return warped_target_img, warped_mask

def compute_homography_AKAZE(ref_img, tar_img, visualize=False):
    """
    计算从目标图像到参考图像的单应性矩阵H (使用SIFT特征)
    参数：
        ref_img: 参考图像(BGR格式)
        tar_img: 目标图像(BGR格式)
    返回：
        warped_target_img: 变换后的目标图像
        warped mask: 变换后的mask(用于求重叠部分)
    """
    # 转换为灰度图
    img_target_gray = cv2.cvtColor(tar_img, cv2.COLOR_BGR2GRAY)
    img_reference_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    # 初始化AKAZE检测器
    akaze = cv2.AKAZE_create()
    
    # 检测关键点和计算描述符
    keypoints_target, descriptors_target = akaze.detectAndCompute(img_target_gray, None)
    keypoints_reference, descriptors_reference = akaze.detectAndCompute(img_reference_gray, None)

    # 暴力匹配器（汉明距离）
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.match(descriptors_target, descriptors_reference)

    # 筛选匹配点（距离小于30）
    good_matches = [m for m in matches if m.distance < 30]
    
    # 提取匹配点坐标
    points_target = np.float32([keypoints_target[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points_reference = np.float32([keypoints_reference[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 计算单应性矩阵（RANSAC方法）
    H, _ = cv2.findHomography(points_target, points_reference, cv2.RANSAC)

    h, w = ref_img.shape[:2]
    warped_target_img = cv2.warpPerspective(tar_img, H, (w, h))

    # 创建全1 mask并进行warp
    mask = np.ones_like(tar_img, dtype=np.uint8)
    warped_mask = cv2.warpPerspective(mask, H, (w, h))
    
    if visualize:
        # 显示结果
        cv2.imshow("Warped Target Image", warped_target_img)
        cv2.imshow("Reference Image", ref_img)
        cv2.imshow("Warped Mask", warped_mask*255)
        cv2.imshow("overlap", warped_mask * ref_img)
        cv2.waitKey(0)

    return warped_target_img, warped_mask

if __name__ == "__main__":
    ref_img = cv2.imread('E:/DeepLearning/0_DataSets/007-UDIS-D/testing/input1/000001.jpg')
    target_img = cv2.imread('E:/DeepLearning/0_DataSets/007-UDIS-D/testing/input2/000001.jpg')

    # 执行配准
    warped_target_img, H = compute_homography_SIFT(ref_img, target_img, True)
