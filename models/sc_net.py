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
        # 经过 3x3 邻域点乘后，会多出 k*k = 9 个维度的空间相似性描述符
        # 将局部特征 (channels) 和空间描述符 (9) 拼接后，再通过 1x1 卷积变回 channels 维度
        self.transform = nn.Sequential(
            nn.Conv2d(channels + k * k, channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, F_local):
        """
        F_local: 局部特征图，形状为 (B, C, H, W)
        """
        B, C, H, W = F_local.shape
        pad = self.k // 2
        
        # 1. 边缘零填充，保证边界点也有完整的 3x3 邻域
        F_pad = F.pad(F_local, (pad, pad, pad, pad), mode='constant', value=0)
        
        # 2. 提取 3x3 邻域特征 (利用 unfold)
        unfolded = F.unfold(F_pad, kernel_size=self.k, padding=0)
        unfolded = unfolded.view(B, C, self.k * self.k, H, W)
        
        # 3. 计算自相似性 (Self-similarity): 核心像素与 3x3 邻域的点乘
        F_center = F_local.unsqueeze(2)  
        S_tilde = (F_center * unfolded).sum(dim=1) 
        
        # 4. 拼接并进行非线性变换
        concat_feat = torch.cat([S_tilde, F_local], dim=1)
        S_spatial = self.transform(concat_feat)
        
        return S_spatial


class SelectiveFusion(nn.Module):
    def __init__(self, channels):
        super(SelectiveFusion, self).__init__()
        # 利用 1x1 卷积代替全连接层来处理空间特征图，降低计算量
        self.fc1 = nn.Conv2d(channels, channels // 4, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // 4, channels * 2, kernel_size=1)
        
    def forward(self, F_local, S_spatial):
        B, C, H, W = F_local.shape
        
        # 1. 元素级相加融合 (Fuse)
        U = F_local + S_spatial
        
        # 2. 挤压 (Squeeze): 全局平均池化 (GAP)
        G = F.adaptive_avg_pool2d(U, (1, 1)) 
        
        # 3. 选择 (Select): 通过两层网络计算权重
        Z = self.fc2(F.relu(self.fc1(G))) 
        
        # 拆分为 A 和 B 两部分
        A, B = torch.chunk(Z, 2, dim=1)
        
        # 拼接在一起以在特征维度上做 SoftMax 保证 A+B=1
        attention = torch.cat([A.unsqueeze(1), B.unsqueeze(1)], dim=1) 
        attention = F.softmax(attention, dim=1)
        
        A_weight = attention[:, 0, :, :, :] 
        B_weight = attention[:, 1, :, :, :] 
        
        # 4. 加权输出
        V = A_weight * F_local + B_weight * S_spatial
        return V


class SCNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(SCNetFeatureExtractor, self).__init__()
        
        # 1. 主干网络: ResNet-101
        resnet101 = models.resnet101(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet101.children())[:-2])
        
        channels = 2048
        
        # 2. 空间信息编码器
        self.spatial_encoder = SpatialContextEncoder(channels=channels, k=3)
        
        # 3. 选择性特征融合
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
        
        # 1. 稀疏 4D 邻域一致性滤波器
        # 包含三层 4D 稀疏卷积，核大小均为 5
        self.sparse_conv1 = ME.MinkowskiConvolution(in_channels=1, out_channels=16, kernel_size=5, dimension=4)
        self.sparse_conv2 = ME.MinkowskiConvolution(in_channels=16, out_channels=16, kernel_size=5, dimension=4)
        self.sparse_conv3 = ME.MinkowskiConvolution(in_channels=16, out_channels=1, kernel_size=5, dimension=4)
        self.relu = ME.MinkowskiReLU(inplace=True)
        
        # 2. 稠密 2D 邻域一致性滤波器
        # 接收 4D 稀疏网络输出并 reshape 后的稠密张量
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

    def extract_sparse_tensor(self, V_s, V_t, m=10):
        """
        计算余弦相似度，提取 top m 个匹配，并构建 Minkowski 稀疏张量
        """
        B, C, H, W = V_s.shape
        
        # 将特征图归一化，以便通过点乘直接计算余弦相似度
        vs_flat = F.normalize(V_s.view(B, C, -1), p=2, dim=1) # (B, C, H*W)
        vt_flat = F.normalize(V_t.view(B, C, -1), p=2, dim=1) # (B, C, H*W)
        
        # 计算相关性矩阵 (B, H*W, H*W)
        sim = torch.bmm(vs_flat.transpose(1, 2), vt_flat)
        
        # 提取 top m=10 的匹配，稀疏化存储
        top_m_vals, top_m_indices = torch.topk(sim, m, dim=2)
        
        # 构建 Minkowski 坐标: [batch_idx, x_s, y_s, x_t, y_t]
        coordinates = []
        features = []
        
        for b in range(B):
            for i in range(H * W): 
                y_s, x_s = i // W, i % W
                for j in range(m): 
                    match_idx = top_m_indices[b, i, j].item()
                    y_t, x_t = match_idx // W, match_idx % W
                    
                    coordinates.append([b, x_s, y_s, x_t, y_t])
                    features.append([top_m_vals[b, i, j].item()])
                    
        # 转换为张量，指定设备
        device = V_s.device
        coords_tensor = torch.IntTensor(coordinates).to(device)
        feats_tensor = torch.FloatTensor(features).to(device)
        
        # 构建并返回稀疏张量 (4D 维度)
        sparse_tensor = ME.SparseTensor(features=feats_tensor, coordinates=coords_tensor)
        return sparse_tensor

    def forward(self, V_s, V_t):
        # A. 生成稀疏相关张量 (源到目标)
        sparse_input = self.extract_sparse_tensor(V_s, V_t, m=10)
        
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
        
        # 1. 特征提取层
        self.feature_extractor = SCNetFeatureExtractor(pretrained=pretrained)
        
        # 2. 邻域一致性滤波层
        self.consensus_module = LightweightNeighbourhoodConsensus()
        
        # 3. 参数回归层：基于置信度得分回归仿射变换的 6 个参数[cite: 1]
        self.regression = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # 全局池化，抹平空间维度
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6)         # 输出 a1, a2, tx, a3, a4, ty[cite: 1]
        )
        
        # 初始化技巧：为了防止初始训练时产生剧烈形变导致梯度爆炸，
        # 我们将回归层初始输出设为单位矩阵参数 [1, 0, 0, 0, 1, 0]
        self.regression[-1].weight.data.zero_()
        self.regression[-1].bias.data.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def forward(self, source_img, target_img):
        # A. 提取局部+空间选择性融合特征
        V_s, V_t = self.feature_extractor(source_img, target_img)
        
        # B. 提取一致性匹配信息
        consensus_out = self.consensus_module(V_s, V_t)
        
        # C. 回归 6 个几何变换参数
        theta = self.regression(consensus_out)
        
        return theta


# --- 单元测试代码 ---
if __name__ == "__main__":
    print("开始测试端到端 SC-Net 完整网络...")
    # 测试时设为False加快实例化速度
    model = SCNet(pretrained=False).cuda() 
    
    # 模拟输入：Batch Size 为 2，3 通道 RGB，尺寸 400x400
    dummy_source = torch.randn(2, 3, 400, 400).cuda()
    dummy_target = torch.randn(2, 3, 400, 400).cuda()
    
    # 执行前向传播
    predicted_theta = model(dummy_source, dummy_target)
    
    print("前向传播圆满成功！")
    print(f"预测的仿射变换参数 theta 形状: {predicted_theta.shape}")
    print(f"参数输出预览: \n{predicted_theta.cpu().detach().numpy()}")