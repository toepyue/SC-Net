import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from models.sc_net import SCNet

def load_image(image_path, size=(400, 400)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像，请检查路径: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img

def main():
    print("--- 开始进行 SC-Net (中高难度抗形变版) 真实图像推理 ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前计算设备: {device}")

    # 1. 核心配置：指定刚刚跑出来的第31轮巅峰权重
    checkpoint_path = "./checkpoints_hard_mode/hard_mode_epoch_31.pth" 
    
    source_img_path = "./real_source.jpg" 
    target_img_path = "./real_target.jpg" 

    if not os.path.exists(checkpoint_path):
        print(f"找不到权重文件 {checkpoint_path}，请确认路径或文件名是否正确！")
        return

    # 2. 读取并预处理图像
    try:
        source_img_np = load_image(source_img_path)
        target_img_np = load_image(target_img_path)
    except ValueError as e:
        print(e)
        return

    transform = transforms.Compose([transforms.ToTensor()])
    source_tensor = transform(source_img_np).unsqueeze(0).to(device)
    target_tensor = transform(target_img_np).unsqueeze(0).to(device)

    # 3. 加载模型
    model = SCNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 4. 前向推理获取仿射参数
    print("⏳ 正在计算对齐参数...")
    with torch.no_grad():
        pred_theta = model(source_tensor, target_tensor)
        # 将 [1, 6] 的输出 reshape 为 [1, 2, 3] 的仿射矩阵
        pred_matrix = pred_theta.view(1, 2, 3).cpu().numpy()[0]
        print(f"✅ 模型预测的归一化仿射矩阵:\n{pred_matrix}")

    # 5. 坐标系转换 (极其关键的一步)
    # 这里的 pred_matrix 是在归一化坐标系 [-1, 1] 下训练出来的
    # 我们需要将其转换回 OpenCV 的真实像素级仿射矩阵，才能正确对图像进行扭曲
    h, w = source_img_np.shape[:2]
    cv_matrix = np.zeros((2, 3), dtype=np.float32)
    cv_matrix[0, 0] = pred_matrix[0, 0]
    cv_matrix[0, 1] = pred_matrix[0, 1]
    cv_matrix[1, 0] = pred_matrix[1, 0]
    cv_matrix[1, 1] = pred_matrix[1, 1]
    cv_matrix[0, 2] = pred_matrix[0, 2] * (w / 2.0) + (w / 2.0) - (pred_matrix[0, 0] * (w / 2.0) + pred_matrix[0, 1] * (h / 2.0))
    cv_matrix[1, 2] = pred_matrix[1, 2] * (h / 2.0) + (h / 2.0) - (pred_matrix[1, 0] * (w / 2.0) + pred_matrix[1, 1] * (h / 2.0))
    
    print(f"🔄 转换后的 OpenCV 像素矩阵:\n{cv_matrix}")

    # 使用预测的矩阵对源图像进行几何扭曲 (Warp)
    warped_source_img = cv2.warpAffine(source_img_np, cv_matrix, (w, h))

    # 6. 使用 Matplotlib 绘制 1x3 对比图
    print("🎨 正在生成结果对比图...")
    plt.figure(figsize=(15, 5))

    # 解决中文乱码问题
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] 
    plt.rcParams['axes.unicode_minus'] = False

    plt.subplot(1, 3, 1)
    plt.imshow(source_img_np)
    plt.title("Real Source")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(target_img_np)
    plt.title("Real Target")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(warped_source_img)
    plt.title("SC-NetWarped Source")
    plt.axis("off")

    # 保存图片
    save_path = "real_inference_result.png"
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"见证奇迹时刻！结果图已成功保存至当前目录: {save_path}")

if __name__ == "__main__":
    main()