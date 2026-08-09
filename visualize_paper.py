import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dataset import AerialImageDataset
from models.sc_net import SCNet

def unnormalize_image(tensor):
    """将 PyTorch Tensor 转换回可供 matplotlib 显示的 NumPy 图像"""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    return img

def main():
    print("--- 启动 SC-Net 论文同款排版可视化模块 (已修复矩阵逆向映射) ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型与巅峰权重
    model = SCNet(pretrained=False).to(device)
    checkpoint_path = "./checkpoints_finetune/finetune_epoch_7.pth" 
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"成功加载巅峰权重: {checkpoint_path}")
    else:
        print(f"未找到权重文件 {checkpoint_path}，请检查路径")
        return
        
    model.eval()

    # 2. 随机抽取【两张】测试图像，对应排版中的两列
    dataset = AerialImageDataset(image_dir="./data/train") 
    idx1, idx2 = np.random.choice(len(dataset), 2, replace=False)
    
    data1 = dataset[idx1]
    data2 = dataset[idx2]
    
    sources = torch.stack([data1[0], data2[0]]).to(device)
    targets = torch.stack([data1[1], data2[1]]).to(device)

    # 3. 运行 SC-Net 前向推理
    with torch.no_grad():
        pred_thetas = model(sources, targets)
    
    # 4. 利用预测参数生成“变换图像”
    pred_thetas_matrix = pred_thetas.view(-1, 2, 3)
    B = pred_thetas_matrix.shape[0]
    
    # 将正向 2x3 矩阵转化为 3x3 并求逆，以适配 PyTorch 渲染机制
    # 4.1 构建 3x3 齐次变换矩阵
    M = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
    M[:, :2, :] = pred_thetas_matrix
    
    # 4.2 计算逆矩阵
    inv_M = torch.linalg.inv(M)
    
    # 4.3 提取前两行，恢复为 grid_sample 所需的 2x3 逆矩阵
    inv_thetas_matrix = inv_M[:, :2, :]
    
    # 4.4 使用【逆矩阵】生成采样网格并扭曲图像
    warp_grids = F.affine_grid(inv_thetas_matrix, sources.size(), align_corners=False)
    warped_tensors = F.grid_sample(sources, warp_grids, align_corners=False)
    
    # 5. 渲染 3行 x 2列 的图表
    fig, axes = plt.subplots(3, 2, figsize=(8, 12))
    
    for col in range(2):
        # 第一行：源图像
        axes[0, col].imshow(unnormalize_image(sources[col]))
        axes[0, col].axis('off')
        
        # 第二行：目标图像
        axes[1, col].imshow(unnormalize_image(targets[col]))
        axes[1, col].axis('off')
        
        # 第三行：变换图像 
        axes[2, col].imshow(unnormalize_image(warped_tensors[col]))
        axes[2, col].axis('off')

    # 调整间距
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    
    # 6. 保存为高清图片
    save_file = "registration_paper_style.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"🎨 完美排版渲染完毕！图片已保存至: {os.path.abspath(save_file)}")

if __name__ == "__main__":
    main()