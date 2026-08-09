import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 导入数据集和完整的 SC-Net 模型
from dataset import AerialImageDataset
from models.sc_net import SCNet

def generate_grid(batch_size, device, step=0.1):
    """生成均匀分布在 -1 到 1 之间的网格点"""
    x = torch.arange(-1.0, 1.0 + step, step, device=device)
    y = torch.arange(-1.0, 1.0 + step, step, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
    num_points = grid_x.numel()
    ones = torch.ones(num_points, device=device)
    grid = torch.stack([grid_x.flatten(), grid_y.flatten(), ones], dim=0)
    grid = grid.unsqueeze(0).repeat(batch_size, 1, 1)
    return grid

def grid_distance_loss(pred_theta, gt_theta, grid):
    """计算变换后的网格距离损失"""
    B = pred_theta.shape[0]
    pred_matrix = pred_theta.view(B, 2, 3)
    gt_matrix = gt_theta.view(B, 2, 3)
    
    pred_points = torch.bmm(pred_matrix, grid)
    gt_points = torch.bmm(gt_matrix, grid)
    
    loss = nn.functional.mse_loss(pred_points, gt_points)
    return loss

def main():
    print("--- 启动 SC-Net 高精度微调 (Fine-tuning) 训练 ---")
    
    save_dir = "./checkpoints_finetune"
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前计算设备: {device}")

    # 1. 实例化数据集与 DataLoader
    train_dataset = AerialImageDataset(image_dir="./data/train")
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    print(f"成功加载训练集，共计 {len(train_dataset)} 张图像。")
    
    # 2. 实例化完整的 SC-Net 模型
    model = SCNet(pretrained=True).to(device)
    
    checkpoint_path = "./checkpoints/scnet_epoch_25.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f" 成功加载预训练权重: {checkpoint_path}\n在此基础上开启高精度微调！")
    else:
        print(f" 未找到权重文件 {checkpoint_path}，请检查路径或文件名！")
        return

    model.train()
    
    # 3. 优化器配置
    # 将初始学习率降低 10 倍，从 0.0005 降为 0.00005
    optimizer = optim.AdamW(model.parameters(), lr=0.00005)
    
    # 引入学习率衰减调度器
    # step_size=5 表示每训练 5 轮触发一次衰减
    # gamma=0.5 表示每次触发时，将当前学习率乘以 0.5（即减半）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # 4. 正式微调训练循环
    # 微调阶段设定为 20 轮
    num_epochs = 20
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        # 获取当前 Epoch 的学习率以便在终端观察调度器的工作状态
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n开始 Epoch [{epoch+1}/{num_epochs}] | 当前学习率: {current_lr:.7f}")
        
        for batch_idx, (source_img, target_img, affine_gt) in enumerate(train_dataloader):
            source_img = source_img.to(device)
            target_img = target_img.to(device)
            affine_gt = affine_gt.to(device).float()
            
            optimizer.zero_grad()
            pred_theta = model(source_img, target_img)
            
            current_batch_size = source_img.shape[0]
            grid = generate_grid(current_batch_size, device)
            loss = grid_distance_loss(pred_theta, affine_gt, grid)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch [{batch_idx+1}/{len(train_dataloader)}] | Loss: {loss.item():.6f}")
        
        # 每个 Epoch 结束后，调用 step() 让调度器更新学习率
        scheduler.step()
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"-> Epoch [{epoch+1}/{num_epochs}] 平均 Loss: {avg_epoch_loss:.6f}")
        
        # 5. 保存模型权重
        # 保存文件加上 finetune_ 前缀
        save_path = os.path.join(save_dir, f"finetune_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"已保存微调权重至: {save_path}\n" + "-"*40)

if __name__ == "__main__":
    main()