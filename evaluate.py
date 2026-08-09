import os
import torch
from torch.utils.data import DataLoader

# 导入数据集和模型
from dataset import AerialImageDataset
from models.sc_net import SCNet

def generate_eval_grid(batch_size, device):
    """生成用于评估的测试网格点 (模拟遍布全图的特征点)"""
    step = 0.1
    x = torch.arange(-1.0, 1.0 + step, step, device=device)
    y = torch.arange(-1.0, 1.0 + step, step, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
    
    num_points = grid_x.numel()
    ones = torch.ones(num_points, device=device)
    grid = torch.stack([grid_x.flatten(), grid_y.flatten(), ones], dim=0)
    grid = grid.unsqueeze(0).repeat(batch_size, 1, 1)
    return grid

def calculate_pck(pred_theta, gt_theta, grid, tau_list=[0.1, 0.05, 0.03]):
    """
    计算特定阈值下的 PCK 指标[cite: 1, 2]。
    由于在 PyTorch 的 grid_sample 机制中，坐标轴范围被归一化为 [-1, 1]，跨度为 2。
    因此在归一化坐标系下的阈值 sigma = tau * 2。
    """
    B = pred_theta.shape[0]
    num_points = grid.shape[2]
    
    pred_matrix = pred_theta.view(B, 2, 3)
    gt_matrix = gt_theta.view(B, 2, 3)
    
    # 获取变换后的点坐标
    # \tilde{p}_i: 模型预测点[cite: 1, 2]
    pred_points = torch.bmm(pred_matrix, grid) 
    # p_i: 真实基准点[cite: 1, 2]
    gt_points = torch.bmm(gt_matrix, grid)     
    
    # 计算预测值与真实值之间的欧氏距离 D[cite: 1, 2]
    distances = torch.norm(pred_points - gt_points, dim=1)
    print(f"当前 Batch 平均预测误差距离: {distances.mean().item():.4f}, 阈值 sigma(0.03): {0.03 * 2.0}")

    pck_results = {}
    for tau in tau_list:
        # 在归一化坐标系下计算阈值
        sigma = tau * 2.0 
        # 统计距离小于 sigma 的正确点数量[cite: 1, 2]
        correct_points = (distances < sigma).sum(dim=1).float()
        
        # 计算该 Batch 的平均 PCK
        pck_batch = correct_points / num_points
        pck_results[tau] = pck_batch.mean().item() * 100 # 转换为百分比输出
        
    return pck_results

def evaluate():
    print("--- 启动 SC-Net 模型性能评估 (PCK) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 指向想测试的权重文件，比如训练完第7轮后生成的 .pth
    checkpoint_path = "./checkpoints_finetune/finetune_epoch_7.pth" 
    
    if not os.path.exists(checkpoint_path):
        print(f"找不到权重文件 {checkpoint_path}。")
        print("请等待 train.py 跑完至少一个 Epoch，并在 checkpoints_finetune 文件夹中生成 .pth 文件后再运行此脚本！")
        return

    # 加载独立划分的测试集（严格包含 500 张图像）

    test_dataset = AerialImageDataset(image_dir="./data/test")
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # 实例化网络并加载权重
    model = SCNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval() # 开启评估模式，冻结 Dropout 和 BatchNorm
    
    # 初始化三个阈值的累加器
    total_pck = {0.1: 0.0, 0.05: 0.0, 0.03: 0.0}
    
    print(f"模型加载成功！开始在 {len(test_dataset)} 张测试图上进行黑盒测试...")
    
    # 评估阶段切忌计算梯度，节约显存加速推理
    with torch.no_grad(): 
        for batch_idx, (source_img, target_img, affine_gt) in enumerate(test_dataloader):
            source_img = source_img.to(device)
            target_img = target_img.to(device)
            affine_gt = affine_gt.to(device).float()
            
            # 前向推理获得 6 个参数
            pred_theta = model(source_img, target_img)
            
            # 生成网格并计算 PCK
            grid = generate_eval_grid(source_img.shape[0], device)
            batch_pck = calculate_pck(pred_theta, affine_gt, grid)
            
            for tau in total_pck.keys():
                total_pck[tau] += batch_pck[tau]
                
    # 计算整个测试集（500 张图）的平均 PCK[cite: 1]
    num_batches = len(test_dataloader)
    print("\n===  SC-Net 最终 PCK 评估结果 ===")
    for tau in [0.1, 0.05, 0.03]:
        final_pck = total_pck[tau] / num_batches
        print(f"PCK (τ = {tau}): {final_pck:.2f}%")
    print("====================================")

if __name__ == "__main__":
    evaluate()