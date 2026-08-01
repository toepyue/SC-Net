import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

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
        # unfolded 形状: (B, C * 9, H * W)
        unfolded = F.unfold(F_pad, kernel_size=self.k, padding=0)
        # 改变形状以便于计算: (B, C, 9, H, W)
        unfolded = unfolded.view(B, C, self.k * self.k, H, W)
        
        # 3. 计算自相似性 (Self-similarity): 核心像素与 3x3 邻域的点乘
        F_center = F_local.unsqueeze(2)  # (B, C, 1, H, W)
        # 在通道维度 (C) 上求和，得到每个像素与 9 个邻居的相似度，形状 (B, 9, H, W)
        S_tilde = (F_center * unfolded).sum(dim=1) 
        
        # 4. 拼接并进行非线性变换
        # 形状变为 (B, C + 9, H, W)
        concat_feat = torch.cat([S_tilde, F_local], dim=1)
        # 输出形状还原为 (B, C, H, W)
        S_spatial = self.transform(concat_feat)
        
        return S_spatial


class SelectiveFusion(nn.Module):
    def __init__(self, channels):
        super(SelectiveFusion, self).__init__()
        # 利用 1x1 卷积代替全连接层来处理空间特征图，降低计算量
        # 论文中提到需要 FC 层提取全局信息然后拆分为 A 和 B
        self.fc1 = nn.Conv2d(channels, channels // 4, kernel_size=1)
        self.fc2 = nn.Conv2d(channels // 4, channels * 2, kernel_size=1)
        
    def forward(self, F_local, S_spatial):
        B, C, H, W = F_local.shape
        
        # 1. 元素级相加融合 (Fuse)
        U = F_local + S_spatial
        
        # 2. 挤压 (Squeeze): 全局平均池化 (GAP)
        G = F.adaptive_avg_pool2d(U, (1, 1)) # 形状: (B, C, 1, 1)
        
        # 3. 选择 (Select): 通过两层网络计算权重
        Z = self.fc2(F.relu(self.fc1(G))) # 形状: (B, 2C, 1, 1)
        
        # 拆分为 A 和 B 两部分，每个形状为 (B, C, 1, 1)
        A, B = torch.chunk(Z, 2, dim=1)
        
        # 拼接在一起以在特征维度上做 SoftMax 保证 A+B=1
        attention = torch.cat([A.unsqueeze(1), B.unsqueeze(1)], dim=1) # (B, 2, C, 1, 1)
        attention = F.softmax(attention, dim=1)
        
        A_weight = attention[:, 0, :, :, :] # (B, C, 1, 1)
        B_weight = attention[:, 1, :, :, :] # (B, C, 1, 1)
        
        # 4. 加权输出
        V = A_weight * F_local + B_weight * S_spatial
        return V


class SCNetFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super(SCNetFeatureExtractor, self).__init__()
        
        # 1. 主干网络: ResNet-101
        resnet101 = models.resnet101(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet101.children())[:-2])
        
        # ResNet101 layer4 输出的通道数是 2048
        channels = 2048
        
        # 2. 空间信息编码器
        self.spatial_encoder = SpatialContextEncoder(channels=channels, k=3)
        
        # 3. 选择性特征融合
        self.selective_fusion = SelectiveFusion(channels=channels)
        
    def forward_once(self, x):
        # a. 提取局部特征 F
        F_local = self.backbone(x)
        
        # b. 嵌入空间信息，得到空间特征 S
        S_spatial = self.spatial_encoder(F_local)
        
        # c. 选择性融合，得到最终的融合特征图 V
        V_fused = self.selective_fusion(F_local, S_spatial)
        
        return V_fused

    def forward(self, source_img, target_img):
        V_s = self.forward_once(source_img)
        V_t = self.forward_once(target_img)
        return V_s, V_t


# --- 单元测试代码 ---
if __name__ == "__main__":
    print("开始测试包含空间信息和注意力机制的 SCNet 完整特征提取端...")
    model = SCNetFeatureExtractor(pretrained=False) # 测试时设为False加快速度
    
    dummy_source = torch.randn(2, 3, 400, 400)
    dummy_target = torch.randn(2, 3, 400, 400)
    
    V_s, V_t = model(dummy_source, dummy_target)
    
    print("前向传播成功！")
    print(f"融合后的源图像特征图 V_s 形状: {V_s.shape}")