import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dataset import AerialImageDataset
from models.sc_net import SCNet

def generate_grid_numpy(step=0.2):
    """生成用于可视化的归一化网格点"""
    x = np.arange(-0.8, 0.8 + step, step)
    y = np.arange(-0.8, 0.8 + step, step)
    grid_x, grid_y = np.meshgrid(x, y)
    points = np.stack([grid_x.flatten(), grid_y.flatten(), np.ones_like(grid_x.flatten())])
    return points

def unnormalize_image(tensor):
    """将 PyTorch Tensor 转换回 NumPy 图像"""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    return img

def main():
    print("--- 启动 SC-Net 综合可视化模块 (已修复矩阵逆向渲染) ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载模型与巅峰权重
    model = SCNet(pretrained=False).to(device)
    checkpoint_path = "./checkpoints_finetune/finetune_epoch_7.pth" 
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✅ 成功加载巅峰权重: {checkpoint_path}")
    else:
        print(f"❌ 未找到权重文件 {checkpoint_path}，请检查路径！")
        return
        
    model.eval()

    # 2. 随机抽取测试图像
    dataset = AerialImageDataset(image_dir="./data/train") 
    idx = np.random.randint(0, len(dataset))
    source_img, target_img, affine_gt = dataset[idx]
    
    source_tensor = source_img.unsqueeze(0).to(device)
    target_tensor = target_img.unsqueeze(0).to(device)
    affine_gt = affine_gt.unsqueeze(0).to(device).float()

    # 3. 运行前向推理
    with torch.no_grad():
        pred_theta = model(source_tensor, target_tensor)
    
    # ---------------------------------------------------------
    # 矩阵求逆，专供 F.grid_sample 渲染图像使用
    # ---------------------------------------------------------
    pred_theta_matrix = pred_theta.view(-1, 2, 3)
    B = pred_theta_matrix.shape[0]
    
    # 构建 3x3 齐次矩阵并求逆
    M = torch.eye(3, device=device).unsqueeze(0).repeat(B, 1, 1)
    M[:, :2, :] = pred_theta_matrix
    inv_M = torch.linalg.inv(M)
    inv_thetas_matrix = inv_M[:, :2, :]
    
    # 使用【逆矩阵】生成采样网格并扭曲图像
    warp_grid = F.affine_grid(inv_thetas_matrix, source_tensor.size(), align_corners=False)
    warped_tensor = F.grid_sample(source_tensor, warp_grid, align_corners=False)
    warped_img_np = unnormalize_image(warped_tensor[0])

    # ---------------------------------------------------------
    # 4. PCK 计算与误差连线：继续使用【正向矩阵】进行坐标系映射
    # ---------------------------------------------------------
    pred_matrix = pred_theta[0].cpu().numpy().reshape(2, 3)
    gt_matrix = affine_gt[0].cpu().numpy().reshape(2, 3)
    
    grid_points = generate_grid_numpy(step=0.2)
    pred_points = pred_matrix @ grid_points
    gt_points = gt_matrix @ grid_points
    
    H, W = source_img.shape[1], source_img.shape[2]
    def to_pixel(pts, w, h):
        pts_pixel = pts.copy()
        pts_pixel[0, :] = (pts[0, :] + 1) * w / 2
        pts_pixel[1, :] = (pts[1, :] + 1) * h / 2
        return pts_pixel

    src_pts_pixel = to_pixel(grid_points, W, H)
    gt_pts_pixel = to_pixel(gt_points, W, H)
    pred_pts_pixel = to_pixel(pred_points, W, H)
    
    # 5. 渲染 1x3 的并排高级图表
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 左图：基准源图像
    axes[0].imshow(unnormalize_image(source_img))
    axes[0].set_title("1. Source Image", fontsize=14)
    axes[0].axis('off')
    
    # 中图：目标图及配准误差彩线 (验证正向数学准确度)
    axes[1].imshow(unnormalize_image(target_img))
    axes[1].scatter(gt_pts_pixel[0], gt_pts_pixel[1], c='lime', s=40, marker='o', edgecolors='black', label='Ground Truth')
    axes[1].scatter(pred_pts_pixel[0], pred_pts_pixel[1], c='red', s=50, marker='X', edgecolors='black', label='Predicted')
    for i in range(pred_pts_pixel.shape[1]):
        axes[1].plot([gt_pts_pixel[0, i], pred_pts_pixel[0, i]], 
                     [gt_pts_pixel[1, i], pred_pts_pixel[1, i]], 
                     color='yellow', linestyle='-', linewidth=1.5, alpha=0.8)
    axes[1].set_title("2. Registration PCK Error", fontsize=14)
    axes[1].legend(loc='lower right')
    axes[1].axis('off')

    # 右图：变换后的源图像 (论文同款视觉验证，使用逆矩阵渲染)
    axes[2].imshow(warped_img_np)
    axes[2].set_title("3. Transformed (Warped) Source", fontsize=14)
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # 6. 保存图片
    save_file = "registration_comprehensive.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"🎨 综合可视化图像渲染完毕！请查看: {os.path.abspath(save_file)}")

if __name__ == "__main__":
    main()