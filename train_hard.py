import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import AerialImageDataset
from models.sc_net import SCNet

def generate_grid(batch_size, device, step=0.1):
    x = torch.arange(-1.0, 1.0 + step, step, device=device)
    y = torch.arange(-1.0, 1.0 + step, step, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    num_points = grid_x.numel()
    ones = torch.ones(num_points, device=device)
    grid = torch.stack([grid_x.flatten(), grid_y.flatten(), ones], dim=0)
    return grid.unsqueeze(0).repeat(batch_size, 1, 1)

def grid_distance_loss(pred_theta, gt_theta, grid):
    B = pred_theta.shape[0]
    pred_matrix = pred_theta.view(B, 2, 3)
    gt_matrix = gt_theta.view(B, 2, 3)
    pred_points = torch.bmm(pred_matrix, grid)
    gt_points = torch.bmm(gt_matrix, grid)
    return nn.functional.mse_loss(pred_points, gt_points)

def main():
    print("🔥 启动 SC-Net 课程学习: Medium-Hard 阶段 🔥")
    
    save_dir = "./checkpoints_hard_mode"
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = AerialImageDataset(image_dir="./data/train")
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # 🌟 核心策略：继承 88.87% 的巅峰权重作为强大的特征提取基座
    model = SCNet(pretrained=False).to(device)
    checkpoint_path = "./checkpoints_finetune/finetune_epoch_7.pth"
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✅ 成功加载巅峰航拍特征基座: {checkpoint_path}")
        print("💡 模型将基于此前的特征提取能力，快速适应中高难度的几何形变！")
    else:
        print(f"❌ 未找到权重文件 {checkpoint_path}，请检查路径！")
        return

    model.train()
    
    # 因为有强大的预训练基座，采用相对温和的 5e-5 初始学习率，避免破坏骨干网络
    optimizer = optim.AdamW(model.parameters(), lr=0.00005)
    num_epochs = 50
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n开始 Epoch [{epoch+1}/{num_epochs}] | 当前学习率: {current_lr:.7f}")
        
        for batch_idx, (source_img, target_img, affine_gt) in enumerate(train_dataloader):
            source_img = source_img.to(device)
            target_img = target_img.to(device)
            affine_gt = affine_gt.to(device).float()
            
            optimizer.zero_grad()
            pred_theta = model(source_img, target_img)
            
            grid = generate_grid(source_img.shape[0], device)
            loss = grid_distance_loss(pred_theta, affine_gt, grid)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch [{batch_idx+1}/{len(train_dataloader)}] | Loss: {loss.item():.6f}")
        
        scheduler.step()
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"-> Epoch [{epoch+1}/{num_epochs}] 平均 Loss: {avg_epoch_loss:.6f}")
        
        save_path = os.path.join(save_dir, f"hard_mode_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"已保存训练权重至: {save_path}\n" + "-"*40)

if __name__ == "__main__":
    main()