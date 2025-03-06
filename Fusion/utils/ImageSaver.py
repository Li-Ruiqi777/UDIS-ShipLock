import cv2
import time
from pathlib import Path
import logging
import numpy as np

class ImageSaver:
    def __init__(self, folder_path: str):
        """
        初始化保存路径，在 folder_path 下以当前时间创建子文件夹。

        :param folder_path: 保存图片的文件夹路径
        """
        base_folder = Path(folder_path)
        base_folder.mkdir(parents=True, exist_ok=True)

        # 创建以当前时间命名的子文件夹
        time_folder_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        self.folder_path = base_folder / time_folder_name
        self.folder_path.mkdir(parents=True, exist_ok=True)
    
        self.mats = []
        self.names = []
        self.logger = logging.getLogger(__name__)

    def __del__(self):
        self.flush()

    def flush(self):
        """
        保存所有图片到文件夹。
        遍历容器中的图片名称，若包含文件夹路径则创建对应的文件夹。
        """
        if not self.mats:
            return

        for mat, name in zip(self.mats, self.names):
            file_path = self.folder_path / name

            # 如果文件路径包含文件夹，则创建文件夹
            if file_path.parent != self.folder_path:
                file_path.parent.mkdir(parents=True, exist_ok=True)

            # 保存图片
            success = cv2.imwrite(str(file_path.with_suffix(".jpg")), mat)
            if not success:
                self.logger.error(f"Error saving image: {file_path}")
            # else:
                # self.logger.info(f"Image saved: {file_path}")

        # 清空图片容器
        self.mats.clear()
        self.names.clear()

    def add_image(self, name: str, img: np.ndarray):
        """
        向容器中添加一个 cv2.Mat 和图片名称。

        :param name: 图片名称（无后缀）
        :param img: OpenCV 图像矩阵
        """
        if not isinstance(img, np.ndarray):
            raise ValueError("The image must be a numpy ndarray.")

        self.mats.append(img.copy())
        self.names.append(name)