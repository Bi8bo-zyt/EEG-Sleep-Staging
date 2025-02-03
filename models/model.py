# model.py改进代码
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from einops import rearrange


# -------------------- 改进模块1: Depthwise Separable Conv --------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


# -------------------- 改进模块2: CBAM注意力 --------------------
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        out = avg_out + max_out
        return self.sigmoid(out).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channel, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


# -------------------- 改进模块3: 相对位置编码Transformer --------------------
class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, max_len=512):
        super().__init__()
        self.num_heads = num_heads
        self.max_len = max_len
        self.relative_bias = nn.Parameter(torch.randn(num_heads, max_len, max_len))

    def forward(self, seq_len):
        bias = self.relative_bias[:, :seq_len, :seq_len]
        return bias.unsqueeze(0)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, afr_reduced_cnn_size, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.relative_pos = RelativePositionBias(h)

        # 使用深度可分离卷积改进key/value投影
        self.query_conv = DepthwiseSeparableConv(afr_reduced_cnn_size, afr_reduced_cnn_size, kernel_size=7)
        self.key_conv = DepthwiseSeparableConv(afr_reduced_cnn_size, afr_reduced_cnn_size, kernel_size=7)
        self.value_conv = DepthwiseSeparableConv(afr_reduced_cnn_size, afr_reduced_cnn_size, kernel_size=7)

        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        batch_size = query.size(0)
        seq_len = query.size(2)

        # 改进的卷积投影
        query = self.query_conv(query).transpose(1, 2)
        key = self.key_conv(key).transpose(1, 2)
        value = self.value_conv(value).transpose(1, 2)

        # 加入相对位置编码
        q = rearrange(query, 'b t (h d) -> b h t d', h=self.h)
        k = rearrange(key, 'b t (h d) -> b h t d', h=self.h)
        v = rearrange(value, 'b t (h d) -> b h t d', h=self.h)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores += self.relative_pos(seq_len)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        x = torch.matmul(attn, v)
        x = rearrange(x, 'b h t d -> b t (h d)')
        return self.linear(x)


# -------------------- 改进的MRCNN模块 --------------------
class MRCNN_Improved(nn.Module):
    def __init__(self, afr_reduced_cnn_size):
        super().__init__()
        drate = 0.5

        # 使用深度可分离卷积改进特征提取
        self.features1 = nn.Sequential(
            DepthwiseSeparableConv(1, 64, kernel_size=50, stride=6, padding=24),
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),

            CBAM(64),  # 加入CBAM注意力

            DepthwiseSeparableConv(64, 128, kernel_size=8, padding=4),
            DepthwiseSeparableConv(128, 128, kernel_size=8, padding=4),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
        )

        self.features2 = nn.Sequential(
            DepthwiseSeparableConv(1, 64, kernel_size=400, stride=50, padding=200),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(drate),

            CBAM(64),  # 加入CBAM注意力

            DepthwiseSeparableConv(64, 128, kernel_size=7, padding=3),
            DepthwiseSeparableConv(128, 128, kernel_size=7, padding=3),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        # 特征金字塔融合
        self.fusion = nn.Sequential(
            DepthwiseSeparableConv(256, afr_reduced_cnn_size, kernel_size=3, padding=1),
            CBAM(afr_reduced_cnn_size),
            nn.AdaptiveAvgPool1d(100)  # 统一特征长度
        )

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = torch.cat((x1, x2), dim=1)
        return self.fusion(x_concat)


# -------------------- 改进的AttnSleep整体架构 --------------------
class AttnSleep_Improved(nn.Module):
    def __init__(self):
        super().__init__()
        N = 3  # 增加Transformer层数
        d_model = 128  # 增大特征维度
        d_ff = 256
        h = 8  # 增加注意力头数
        dropout = 0.2
        num_classes = 5
        afr_reduced_cnn_size = 64  # 增大特征维度

        self.mrcnn = MRCNN_Improved(afr_reduced_cnn_size)

        attn = MultiHeadedAttention(h, d_model, afr_reduced_cnn_size)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.tce = TCE(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), afr_reduced_cnn_size, dropout), N)

        # 改进分类头：加入多层级感知机
        self.classifier = nn.Sequential(
            nn.Linear(d_model * afr_reduced_cnn_size, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x_feat = self.mrcnn(x)
        encoded_features = self.tce(x_feat)
        encoded_features = encoded_features.contiguous().view(encoded_features.shape[0], -1)
        return self.classifier(encoded_features)