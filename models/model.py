# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, reduce


# 深度可分离卷积模块
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


# 改进的MRCNN模块
class EnhancedMRCNN(nn.Module):
    def __init__(self, afr_reduced_cnn_size):
        super().__init__()
        drate = 0.5
        self.GELU = nn.GELU()

        # 多尺度深度可分离卷积
        self.features1 = nn.Sequential(
            DepthwiseSeparableConv(1, 64, 50, 6, 24),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(8, 2, 4),
            nn.Dropout(drate),

            DepthwiseSeparableConv(64, 128, 8, 1, 4),
            nn.BatchNorm1d(128),
            self.GELU,

            DepthwiseSeparableConv(128, 128, 8, 1, 4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(4, 4, 2)
        )

        self.features2 = nn.Sequential(
            DepthwiseSeparableConv(1, 64, 400, 50, 200),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(4, 2, 2),
            nn.Dropout(drate),

            DepthwiseSeparableConv(64, 128, 7, 1, 3),
            nn.BatchNorm1d(128),
            self.GELU,

            DepthwiseSeparableConv(128, 128, 7, 1, 3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(2, 2, 1)
        )

        # 自适应特征融合
        self.adaptive_fusion = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(256, 128, 1),
            nn.Sigmoid()
        )

        self.AFR = nn.Sequential(
            DepthwiseSeparableConv(256, afr_reduced_cnn_size, 3, 1, 1),
            nn.BatchNorm1d(afr_reduced_cnn_size),
            self.GELU
        )

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)

        # 自适应加权融合
        x_cat = torch.cat([x1, x2], dim=1)
        weights = self.adaptive_fusion(x_cat)
        x_fused = x1 * weights[:, :128] + x2 * weights[:, 128:]

        return self.AFR(x_fused)


# 混合注意力机制
class HybridAttention(nn.Module):
    def __init__(self, d_model, kernel_size=7):
        super().__init__()
        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_model, d_model // 8, 1),
            nn.GELU(),
            nn.Conv1d(d_model // 8, d_model, 1),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv1d(d_model, d_model // 8, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(d_model // 8),
            nn.GELU(),
            nn.Conv1d(d_model // 8, 1, 1),
            nn.Sigmoid()
        )

        # 时序卷积
        self.tcn = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        )

    def forward(self, x):
        # 通道注意力
        ca = self.channel_att(x)
        # 空间注意力
        sa = self.spatial_att(x)
        # 特征增强
        x = x * ca * sa
        # 时序建模
        return self.tcn(x)


# 双向TCN模块
class BiTCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Conv1d(hidden_size, hidden_size, 3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.GELU()
        )
        self.bwd_tcn = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Conv1d(hidden_size, hidden_size, 3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.GELU()
        )

    def forward(self, x):
        fwd = self.tcn(x)
        bwd = self.bwd_tcn(torch.flip(x, dims=[-1]))
        return fwd + torch.flip(bwd, dims=[-1])


# 改进后的AttnSleep模型
class EnhancedAttnSleep(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        afr_reduced_cnn_size = 30
        d_model = 80
        d_ff = 120
        self.num_classes = num_classes

        # 改进的特征提取
        self.mrcnn = EnhancedMRCNN(afr_reduced_cnn_size)

        # 双向时序卷积
        self.bitcn = BiTCN(afr_reduced_cnn_size, afr_reduced_cnn_size)

        # 混合注意力模块
        self.hybrid_att = HybridAttention(afr_reduced_cnn_size)

        # 多任务输出
        self.main_fc = nn.Linear(afr_reduced_cnn_size * d_model, num_classes)
        self.aux_fc = nn.Sequential(
            nn.Linear(afr_reduced_cnn_size * d_model, 32),
            nn.GELU(),
            nn.Linear(32, 1)  # 睡眠质量评分回归
        )

    def forward(self, x):
        # 特征提取
        x = self.mrcnn(x)  # [B, C, T]

        # 双向时序建模
        x = self.bitcn(x)

        # 混合注意力
        x = self.hybrid_att(x)

        # 特征聚合
        x = rearrange(x, 'b c t -> b (c t)')

        # 多任务输出
        main_out = self.main_fc(x)
        aux_out = self.aux_fc(x)

        return main_out, aux_out