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
        # 获取文件夹下所有的图片文件名
        self.image_names = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # 定义图像预处理操作：转为 Tensor 并缩放至 400x400
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((400, 400))
        ])

    def __len__(self):
        return len(self.image_names)

    def generate_random_affine(self):
        """
        生成随机的 2x3 仿射变换矩阵 (包含平移、旋转、缩放)
        输出为 6 个参数的组合矩阵
        """
        import random
        
        # 1. 设定随机变换的范围
        angle = random.uniform(-20, 20)      # 随机旋转：-20度 到 20度
        scale = random.uniform(0.8, 1.2)     # 随机缩放：0.8倍 到 1.2倍
        tx = random.uniform(-30, 30)         # X轴随机平移：-30 到 30 像素
        ty = random.uniform(-30, 30)         # Y轴随机平移：-30 到 30 像素
        
        # 2. 假设原图中心点为 (200, 200) 进行旋转和缩放
        center = (200.0, 200.0)
        
        # 3. 使用 OpenCV 获取旋转和缩放的 2x3 矩阵
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        
        # 4. 在矩阵的平移分量上叠加随机平移量
        matrix[0, 2] += tx
        matrix[1, 2] += ty
        
        return matrix.astype(np.float32)

    def __getitem__(self, idx):
        # 1. 读取源图像
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        source_img = cv2.imread(img_path)
        source_img = cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB) # 转为 RGB
        
        # 2. 生成随机仿射变换矩阵
        affine_matrix = self.generate_random_affine()
        
        # 3. 对源图像进行几何变换，生成目标图像 (合成变换)
        h, w = source_img.shape[:2]
        target_img = cv2.warpAffine(source_img, affine_matrix, (w, h))
        
        # 4. 转换为 PyTorch 的 Tensor，并统一尺寸到 400x400
        source_tensor = self.transform(source_img)
        target_tensor = self.transform(target_img)
        
        # 将矩阵转为 Tensor
        affine_tensor = torch.from_numpy(affine_matrix).flatten() # 展平为 6 个参数

        return source_tensor, target_tensor, affine_tensor

# 测试代码，直接运行本文件时会执行
if __name__ == "__main__":
    print("开始测试 Dataset 框架...")
    
    # 实例化数据集，指明图片存放的路径
    test_dataset = AerialImageDataset(image_dir="./data")
    
    if len(test_dataset) > 0:
        # 获取数据集中的第一个样本（也就是我们刚下载的 test.jpg）
        source, target, affine = test_dataset[0]
        
        print(f"成功读取并处理图像！")
        print(f"源图像 Tensor 形状: {source.shape}") 
        print(f"目标图像 Tensor 形状: {target.shape}") 
        print(f"仿射变换参数 Tensor 形状: {affine.shape}") 
        print(f"真实的仿射变换参数 (Ground Truth): \n{affine}")
    else:
        print("警告：在 data 文件夹中没有找到图片！")