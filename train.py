import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy


# ----------------- 新增模块 -----------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   padding=padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class CBAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channel, channel // reduction, 1),
            nn.GELU(),
            nn.Conv1d(channel // reduction, channel, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(channel, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x)
        x = ca * x
        sa = self.spatial_attention(x)
        return sa * x


# ----------------- 改进的MRCNN模块 -----------------
class EnhancedMRCNN(nn.Module):
    def __init__(self, afr_reduced_cnn_size):
        super().__init__()
        drate = 0.5
        self.features = nn.Sequential(
            DepthwiseSeparableConv(1, 64, 50, 24),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(8, 2, 4),
            CBAM(64),
            nn.Dropout(drate),

            DepthwiseSeparableConv(64, 128, 8, 4),
            nn.BatchNorm1d(128),
            nn.GELU(),
            CBAM(128),

            DepthwiseSeparableConv(128, 256, 8, 4),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.MaxPool1d(4, 4, 2),
            CBAM(256)
        )

        self.temporal_pool = nn.AdaptiveAvgPool1d(128)  # 统一时序维度
        self.AFR = nn.Sequential(
            SEBasicBlock(256, afr_reduced_cnn_size),
            CBAM(afr_reduced_cnn_size)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.temporal_pool(x)
        return self.AFR(x)


# ----------------- 改进的注意力模块 -----------------
class RelativePositionEncoder(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.position_bias = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.position_bias[:seq_len].unsqueeze(0)


class EnhancedMultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, afr_size, dropout=0.1):
        super().__init__()
        self.d_k = d_model // h
        self.h = h
        self.position_enc = RelativePositionEncoder(d_model)

        self.query = DepthwiseSeparableConv(afr_size, afr_size, 7, 3)
        self.key = DepthwiseSeparableConv(afr_size, afr_size, 7, 3)
        self.value = DepthwiseSeparableConv(afr_size, afr_size, 7, 3)

        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        b = query.size(0)

        # 加入相对位置编码
        query = self.position_enc(query.permute(0, 2, 1)).permute(0, 2, 1)
        key = self.position_enc(key.permute(0, 2, 1)).permute(0, 2, 1)

        q = self.query(query).view(b, self.h, self.d_k, -1)
        k = self.key(key).view(b, self.h, self.d_k, -1)
        v = self.value(value).view(b, self.h, self.d_k, -1)

        scores = torch.einsum('bhdi,bhdj->bhij', q, k) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('bhij,bhdj->bhdi', attn, v)
        out = out.contiguous().view(b, -1, self.h * self.d_k)
        return self.proj(out)


# ----------------- 最终模型 -----------------
class EnhancedAttnSleep(nn.Module):
    def __init__(self):
        super().__init__()
        N = 3  # 增加Transformer层数
        d_model = 128  # 扩大特征维度
        d_ff = 256
        h = 8  # 增加注意力头数
        afr_size = 64  # 调整AFR输出维度

        self.mrcnn = EnhancedMRCNN(afr_size)

        attn = EnhancedMultiHeadAttention(h, d_model, afr_size)
        ff = PositionwiseFeedForward(d_model, d_ff)
        self.tce = TCE(EncoderLayer(d_model, attn, ff, afr_size, dropout=0.1), N)

        self.bilstm = nn.LSTM(d_model, d_model // 2,
                              bidirectional=True, batch_first=True)

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * afr_size),
            nn.Linear(d_model * afr_size, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 5)
        )

    def forward(self, x):
        x = self.mrcnn(x)  # [B, afr_size, T]
        x = x.permute(0, 2, 1)  # [B, T, D]

        # Transformer编码
        x = self.tce(x)

        # 双向LSTM
        x, _ = self.bilstm(x)

        # 分类
        x = x.contiguous().view(x.size(0), -1)
        return self.classifier(x)