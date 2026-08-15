import os
import glob
import torch
from torch.utils.data import DataLoader

# 导入数据集和模型
from dataset import AerialImageDataset
from models.sc_net import SCNet

def generate_eval_grid(batch_size, device):
    """生成用于评估的测试网格点"""
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
    计算特定阈值下的 PCK 指标。
    在 PyTorch 的 grid_sample 机制中，坐标轴范围被归一化为 [-1, 1]，跨度为 2。
    因此在归一化坐标系下的阈值 sigma = tau * 2。
    """
    B = pred_theta.shape[0]
    num_points = grid.shape[2]
    
    pred_matrix = pred_theta.view(B, 2, 3)
    gt_matrix = gt_theta.view(B, 2, 3)
    
    # 获取变换后的点坐标
    pred_points = torch.bmm(pred_matrix, grid) 
    gt_points = torch.bmm(gt_matrix, grid)     
    
    # 计算预测值与真实值之间的欧氏距离 D
    distances = torch.norm(pred_points - gt_points, dim=1)
    
    # 仅打印第一个 Batch 的误差，避免日志刷屏
    if not hasattr(calculate_pck, "has_printed"):
        print(f" 当前 Batch 平均预测误差距离: {distances.mean().item():.4f}, 极限阈值 sigma(0.03): {0.03 * 2.0}")
        calculate_pck.has_printed = True

    pck_results = {}
    for tau in tau_list:
        sigma = tau * 2.0 
        correct_points = (distances < sigma).sum(dim=1).float()
        pck_batch = correct_points / num_points
        pck_results[tau] = pck_batch.mean().item() * 100 
        
    return pck_results

def evaluate():
    print("--- 启动 SC-Net 极限边界测试 PCK 评估 ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint_dir = "./checkpoints_hard_mode"
    
    if not os.path.exists(checkpoint_dir):
        print(f"❌ 找不到权重文件夹 {checkpoint_dir}。")
        print("请等待 train_hard.py 跑完至少一个 Epoch 后再运行此脚本！")
        return

    # 寻找该目录下所有的 .pth 权重文件
    pth_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not pth_files:
        print(f"❌ 在 {checkpoint_dir} 中没有找到任何权重文件！")
        return
        
    #  交互式选择器：按文件的修改时间倒序排列（最新的在最上面）
    pth_files.sort(key=os.path.getmtime, reverse=True)
    
    print("\n 发现以下权重文件 (按生成时间由新到旧排序):")
    for idx, f in enumerate(pth_files):
        print(f"  [{idx + 1}] {os.path.basename(f)}")
        
    # 等待用户输入选择
    while True:
        try:
            choice = input(f"\n👉 请输入你想评估的权重编号 (1-{len(pth_files)}): ")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(pth_files):
                selected_checkpoint = pth_files[choice_idx]
                break
            else:
                print("⚠️ 编号超出范围，请重新输入！")
        except ValueError:
            print("⚠️ 请输入有效的数字！")

    print(f"\n✅ 已锁定权重文件: {selected_checkpoint}")

    # 加载独立划分的测试集
    test_dataset = AerialImageDataset(image_dir="./data/test")
    if len(test_dataset) == 0:
        print("❌ 警告：在 ./data/test 文件夹中没有找到测试图片！")
        return
        
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # 实例化网络并加载权重
    model = SCNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(selected_checkpoint, map_location=device))
    model.eval() 
    
    total_pck = {0.1: 0.0, 0.05: 0.0, 0.03: 0.0}
    
    print(f"模型加载成功！开始在 {len(test_dataset)} 张极端大形变测试图上进行黑盒测试...")
    
    with torch.no_grad(): 
        for batch_idx, (source_img, target_img, affine_gt) in enumerate(test_dataloader):
            source_img = source_img.to(device)
            target_img = target_img.to(device)
            affine_gt = affine_gt.to(device).float()
            
            pred_theta = model(source_img, target_img)
            
            grid = generate_eval_grid(source_img.shape[0], device)
            batch_pck = calculate_pck(pred_theta, affine_gt, grid)
            
            for tau in total_pck.keys():
                total_pck[tau] += batch_pck[tau]
                
    num_batches = len(test_dataloader)
    print("\n===  SC-Net 最终 PCK 评估结果 ===")
    for tau in [0.1, 0.05, 0.03]:
        final_pck = total_pck[tau] / num_batches
        print(f"PCK (τ = {tau}): {final_pck:.2f}%")
    print("==================================================")

if __name__ == "__main__":
    evaluate()