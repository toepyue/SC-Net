import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class AerialImageDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_names = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # 移除了 Resize，确保原生读取后立刻强行对齐尺寸
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_names)

    def generate_random_affine(self):
        """
        生成随机的 2x3 仿射变换矩阵
        🌟 课程学习阶段一：Medium-Hard (中高难度大尺度形变) 🌟
        """
        import random
        
        # 旋转：扩大到 [-60°, 60°]
        angle = random.uniform(-60, 60)      
        # 缩放：扩大到 [0.6, 1.5]
        scale = random.uniform(0.6, 1.5)     
        # 平移：扩大到 [-40, 40]
        tx = random.uniform(-40, 40)         
        ty = random.uniform(-40, 40)         
        
        center = (200.0, 200.0)
        matrix_cv = cv2.getRotationMatrix2D(center, angle, scale)
        matrix_cv[0, 2] += tx
        matrix_cv[1, 2] += ty
        
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
        
        # 🌟 修复底层坐标系错位 Bug：先强行锁定 400x400 尺寸
        source_img = cv2.resize(source_img, (400, 400))
        
        matrix_cv, matrix_norm = self.generate_random_affine()
        target_img = cv2.warpAffine(source_img, matrix_cv, (400, 400))
        
        source_tensor = self.transform(source_img)
        target_tensor = self.transform(target_img)
        affine_tensor = torch.from_numpy(matrix_norm).flatten() 

        return source_tensor, target_tensor, affine_tensor