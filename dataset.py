import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class AerialImageDataset(Dataset):
    def __init__(self, image_dir):
        """
        初始化数据集
        :param image_dir: 存放 Google Earth 原始图像的文件夹路径
        """
        self.image_dir = image_dir
        self.image_names = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((400, 400))
        ])

    def __len__(self):
        return len(self.image_names)

    def generate_random_affine(self):
        """
        生成随机的 2x3 仿射变换矩阵 (包含平移、旋转、缩放)
        返回两个矩阵：一个用于 CV2 图像变换，一个用于神经网络标签
        """
        import random
        
        angle = random.uniform(-20, 20)      
        scale = random.uniform(0.8, 1.2)     
        tx = random.uniform(-30, 30)         
        ty = random.uniform(-30, 30)         
        
        # 1. 构建 OpenCV 像素级仿射矩阵 (用于生成目标图片)
        center = (200.0, 200.0)
        matrix_cv = cv2.getRotationMatrix2D(center, angle, scale)
        matrix_cv[0, 2] += tx
        matrix_cv[1, 2] += ty
        
        # 2. 构建 PyTorch 归一化仿射矩阵 (作为神经网络的 Ground Truth)
        # 在 [-1, 1] 的空间中，围绕中心 (0,0) 的旋转缩放矩阵 A 保持不变
        # 平移量 tx, ty 需要除以图像尺寸的一半 (200.0) 进行归一化
        matrix_norm = np.zeros((2, 3), dtype=np.float32)
        matrix_norm[0, 0] = matrix_cv[0, 0]
        matrix_norm[0, 1] = matrix_cv[0, 1]
        matrix_norm[1, 0] = matrix_cv[1, 0]
        matrix_norm[1, 1] = matrix_cv[1, 1]
        matrix_norm[0, 2] = tx / 200.0
        matrix_norm[1, 2] = ty / 200.0
        
        return matrix_cv.astype(np.float32), matrix_norm.astype(np.float32)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        source_img = cv2.imread(img_path)
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB) 
        
        # 获取两种尺度的变换矩阵
        matrix_cv, matrix_norm = self.generate_random_affine()
        
        # 对源图像进行几何变换，生成目标图像 (使用像素级矩阵)
        h, w = source_img.shape[:2]
        target_img = cv2.warpAffine(source_img, matrix_cv, (w, h))
        
        source_tensor = self.transform(source_img)
        target_tensor = self.transform(target_img)
        
        # 将归一化矩阵转为 Tensor 喂给网络 (使用归一化矩阵)
        affine_tensor = torch.from_numpy(matrix_norm).flatten() 

        return source_tensor, target_tensor, affine_tensor

if __name__ == "__main__":
    print("开始测试 Dataset 框架...")
    test_dataset = AerialImageDataset(image_dir="./data")
    if len(test_dataset) > 0:
        source, target, affine = test_dataset[0]
        print(f"成功读取并处理图像！")
        print(f"真实的仿射变换参数 (归一化 Ground Truth): \n{affine}")
    else:
        print("警告：在 data 文件夹中没有找到图片！")