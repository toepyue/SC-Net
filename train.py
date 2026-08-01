import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 导入写好的 Dataset 和 Model
from dataset import AerialImageDataset
from models.sc_net import SCNetFeatureExtractor

def main():
    print("--- 初始化训练大流程测试 ---")
    
    # 1. 硬件设备配置：自动检测 RTX 3090 (CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device}")

    # 2. 数据加载器 (DataLoader) 配置
    # 论文中指定的批大小为 8[cite: 1]

    dataset = AerialImageDataset(image_dir="./data")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 3. 模型实例化并移动到 GPU
    model = SCNetFeatureExtractor(pretrained=True).to(device)
    model.train() # 设置为训练模式
    
    # 4. 优化器配置
    # 严格按照论文参数：AdamW 优化器，学习率 0.0005[cite: 1]
    optimizer = optim.AdamW(model.parameters(), lr=0.0005)
    
    # 5. 训练循环
    # 论文设定训练 90 个 Epoch[cite: 1]，这里为了测试大流程，只跑 2 个 Epoch
    num_epochs = 2
    for epoch in range(num_epochs):
        print(f"\n开始第 {epoch+1}/{num_epochs} 轮训练...")
        
        for batch_idx, (source_img, target_img, affine_gt) in enumerate(dataloader):
            # 将数据推送到 GPU
            source_img = source_img.to(device)
            target_img = target_img.to(device)
            affine_gt = affine_gt.to(device)
            
            # 步骤 A: 梯度清零
            optimizer.zero_grad()
            
            # 步骤 B: 前向传播 (Forward)
            F_s, F_t = model(source_img, target_img)
            
            # 【临时测试逻辑】
            # 因为目前网络还没有加上最后的回归层，无法直接和 affine_gt (真实 6 参数) 计算损失。
            # 为了测试整个管道的反向传播能否成功，我们临时做一个假损失 (Fake Loss)。
            # 即强行计算两个特征图的均方差，迫使模型进行权重更新。
            loss_fn = nn.MSELoss()
            fake_loss = loss_fn(F_s, F_t)
            
            # 步骤 C: 反向传播 (Backward)
            fake_loss.backward()
            
            # 步骤 D: 参数更新
            optimizer.step()
            
            print(f"  Batch {batch_idx+1} | 成功完成前向与反向传播！临时 Fake Loss: {fake_loss.item():.4f}")
            print(f"  -> 提取的源特征图 F_s 形状: {F_s.shape}")

    print("\n--- 大流程测试圆满成功！---")

if __name__ == "__main__":
    main()