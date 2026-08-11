import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# 导入底层 4D 稀疏卷积扩展库
import MinkowskiEngine as ME

class SpatialContextEncoder(nn.Module):
    def __init__(self, channels, k=3):
        super(SpatialContextEncoder, self).__init__()
        self.k = k
        self.transform = nn.Sequential(
            nn.Conv2d(channels + k * k, channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, F_local):
        B, C, H, W = F_local.shape
        pad = self.k // 2
        
        # 边缘零填充，确保边缘点存在 k*k 区域[cite: 3]
        F_pad = F.pad(F_local, (pad, pad, pad, pad), mode='constant', value=0)
        
        unfolded = F.unfold(F_pad, kernel_size=self.k, padding=0)
        unfolded = unfolded.view(B, C, self.k * self.k, H, W)
        
        # 计算空间上下文描述符自相似性[cite: 3]
        F_center = F_local.unsqueeze(2)  
        S_tilde = (F_center * unfolded).sum(dim=1) 
        
        # 将空间信息与局部特征连接后，应用非线性变换[cite: 3]
        concat_feat = torch.cat([S_tilde, F_local], dim=1)
        S_spatial = self.transform(concat_feat)
        
        return S_spatial


class SelectiveFusion(nn.Module):
    def __init__(self, channels):
        super(SelectiveFusion, self).__init__()
        self.fc1 = nn.Conv2d(channels, channels // 4, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // 4, channels * 2, kernel_size=1)
        
    def forward(self, F_local, S_spatial):
        B, C, H, W = F_local.shape
        
        # 逐元素求和策略进行融合以减少冗余[cite: 3]
        U = F_local + S_spatial
        
        # 采用全局平均池化策略 (GAP) 来压缩空间获取全局感受野[cite: 3]
        G = F.adaptive_avg_pool2d(U, (1, 1)) 
        
        # 送入全连接层减少维度，再还原得到 A 和 B[cite: 3]
        Z = self.fc2(F.relu(self.fc1(G))) 
        A, B = torch.chunk(Z, 2, dim=1)
        
        # 通过 SoftMax 层实现选择，确保 Ac + Bc = 1[cite: 3]
        attention = torch.cat([A.unsqueeze(1), B.unsqueeze(1)], dim=1) 
        attention = F.softmax(attention, dim=1)
        
        A_weight = attention[:, 0, :, :, :] 
        B_weight = attention[:, 1, :, :, :] 
        
        # 计算 Vc = Ac * Fc + Bc * Sc[cite: 3]
        V = A_weight * F_local + B_weight * S_spatial
        return V


class SCNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(SCNetFeatureExtractor, self).__init__()
        
        # 选择 ResNet-101 作为骨干网络[cite: 3]
        resnet101 = models.resnet101(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet101.children())[:-2])
        
        channels = 2048
        
        self.spatial_encoder = SpatialContextEncoder(channels=channels, k=3)
        self.selective_fusion = SelectiveFusion(channels=channels)
        
    def forward_once(self, x):
        F_local = self.backbone(x)
        S_spatial = self.spatial_encoder(F_local)
        V_fused = self.selective_fusion(F_local, S_spatial)
        return V_fused

    def forward(self, source_img, target_img):
        V_s = self.forward_once(source_img)
        V_t = self.forward_once(target_img)
        return V_s, V_t


class LightweightNeighbourhoodConsensus(nn.Module):
    def __init__(self):
        super(LightweightNeighbourhoodConsensus, self).__init__()
        
        # 稀疏邻域共识滤波器：三层四维稀疏卷积层，5*5*5*5卷积核[cite: 3]
        self.sparse_conv1 = ME.MinkowskiConvolution(in_channels=1, out_channels=16, kernel_size=5, dimension=4)
        self.sparse_conv2 = ME.MinkowskiConvolution(in_channels=16, out_channels=16, kernel_size=5, dimension=4)
        self.sparse_conv3 = ME.MinkowskiConvolution(in_channels=16, out_channels=1, kernel_size=5, dimension=4)
        self.relu = ME.MinkowskiReLU(inplace=True)
        
        # 稠密邻域共识滤波器：三层二维卷积层 (11*11, 7*7, 5*5)[cite: 3]
        self.dense_filter = nn.Sequential(
            nn.Conv2d(1, 225, kernel_size=11, padding=5),
            nn.BatchNorm2d(225),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(225, 128, kernel_size=7, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def extract_asymmetric_sparse_tensor(self, V_s, V_t, m=10):
        """
        🌟 核心改进：严格复现论文中的非对称匹配矩阵 C_ST = C_S->T + C_T->S 🌟
        """
        B, C, H, W = V_s.shape
        N = H * W
        device = V_s.device
        
        vs_flat = F.normalize(V_s.view(B, C, -1), p=2, dim=1) 
        vt_flat = F.normalize(V_t.view(B, C, -1), p=2, dim=1) 

        # -----------------------------------------------------
        # 1. 计算 S -> T 的稀疏相关矩阵[cite: 3]
        # -----------------------------------------------------
        sim_st = torch.bmm(vs_flat.transpose(1, 2), vt_flat)
        val_st, idx_st = torch.topk(sim_st, m, dim=2)
        
        b_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, m).flatten()
        src_idx_st = torch.arange(N, device=device).view(1, N, 1).expand(B, N, m).flatten()
        dst_idx_st = idx_st.flatten()
        
        y_s_st, x_s_st = src_idx_st // W, src_idx_st % W
        y_t_st, x_t_st = dst_idx_st // W, dst_idx_st % W
        
        coords_st = torch.stack([b_idx, x_s_st, y_s_st, x_t_st, y_t_st], dim=1).int()
        feats_st = val_st.flatten().unsqueeze(1).float()

        # -----------------------------------------------------
        # 2. 计算 T -> S 的稀疏相关矩阵 (顺序反转)[cite: 3]
        # -----------------------------------------------------
        sim_ts = torch.bmm(vt_flat.transpose(1, 2), vs_flat)
        val_ts, idx_ts = torch.topk(sim_ts, m, dim=2)
        
        src_idx_ts = torch.arange(N, device=device).view(1, N, 1).expand(B, N, m).flatten()
        dst_idx_ts = idx_ts.flatten()
        
        # 注意：此处源是 T，目标是 S，要将其映射回相同的 [x_s, y_s, x_t, y_t] 坐标系
        y_t_ts, x_t_ts = src_idx_ts // W, src_idx_ts % W
        y_s_ts, x_s_ts = dst_idx_ts // W, dst_idx_ts % W
        
        coords_ts = torch.stack([b_idx, x_s_ts, y_s_ts, x_t_ts, y_t_ts], dim=1).int()
        feats_ts = val_ts.flatten().unsqueeze(1).float()

        # -----------------------------------------------------
        # 3. 将两个相关张量加在一起，实现不对称性[cite: 3]
        # ME.SparseTensor 默认会将相同坐标 (Coordinates) 的特征 (Features) 进行求和
        # -----------------------------------------------------
        coords_combined = torch.cat([coords_st, coords_ts], dim=0)
        feats_combined = torch.cat([feats_st, feats_ts], dim=0)
        
        sparse_tensor = ME.SparseTensor(features=feats_combined, coordinates=coords_combined)
        return sparse_tensor

    def forward(self, V_s, V_t):
        # A. 生成非对称稀疏相关张量
        sparse_input = self.extract_asymmetric_sparse_tensor(V_s, V_t, m=10)
        
        # B. 稀疏 4D 卷积滤波
        x = self.relu(self.sparse_conv1(sparse_input))
        x = self.relu(self.sparse_conv2(x))
        x = self.sparse_conv3(x)
        
        # C. 转换为稠密张量
        dense_tensor, _, _ = x.dense() 
        B = dense_tensor.shape[0]
        
        # 将空间维度合并，作为 2D 卷积的输入 (B, 1, H_s*H_t, W_s*W_t)
        dense_2d_input = dense_tensor.view(B, 1, -1, dense_tensor.shape[-1])
        
        # D. 2D 稠密卷积滤波
        out = self.dense_filter(dense_2d_input)
        
        return out


class SCNet(nn.Module):
    """
    SC-Net 顶层模块，将特征提取、一致性滤波与参数回归串联
    """
    def __init__(self, pretrained=True):
        super(SCNet, self).__init__()
        
        self.feature_extractor = SCNetFeatureExtractor(pretrained=pretrained)
        self.consensus_module = LightweightNeighbourhoodConsensus()
        
        # 参数回归：使用全连接层来进行参数回归来获取仿射变换[cite: 3]
        self.regression = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6)         
        )
        
        self.regression[-1].weight.data.zero_()
        self.regression[-1].bias.data.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def forward(self, source_img, target_img):
        V_s, V_t = self.feature_extractor(source_img, target_img)
        consensus_out = self.consensus_module(V_s, V_t)
        theta = self.regression(consensus_out)
        
        return theta


# --- 单元测试代码 ---
if __name__ == "__main__":
    print("开始测试端到端 SC-Net 完整网络 (非对称匹配增强版)...")
    model = SCNet(pretrained=False).cuda() 
    
    dummy_source = torch.randn(2, 3, 400, 400).cuda()
    dummy_target = torch.randn(2, 3, 400, 400).cuda()
    
    predicted_theta = model(dummy_source, dummy_target)
    
    print("前向传播圆满成功！")
    print(f"预测的仿射变换参数 theta 形状: {predicted_theta.shape}")