# model.py改进代码
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from copy import deepcopy
from einops import rearrange

# -------------------- 改进模块1: Depthwise Separable Conv --------------------
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels
        )
        self.padding = (kernel_size - 1) // 2  # 自动计算对称填充
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)
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
            nn.MaxPool1d(kernel_size=8, stride=4, padding=4),  # 修改stride
            nn.Dropout(drate),

            CBAM(64),

            DepthwiseSeparableConv(64, 128, kernel_size=8, padding=4),
            DepthwiseSeparableConv(128, 128, kernel_size=8, padding=4),
            nn.AdaptiveAvgPool1d(100)  # 新增自适应池化
        )

        self.features2 = nn.Sequential(
            DepthwiseSeparableConv(1, 64, kernel_size=400, stride=50, padding=200),
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(drate),

            CBAM(64),

            DepthwiseSeparableConv(64, 128, kernel_size=7, padding=3),
            DepthwiseSeparableConv(128, 128, kernel_size=7, padding=3),
            nn.AdaptiveAvgPool1d(100)  # 新增自适应池化
        )

        # 特征金字塔融合
        self.fusion = nn.Sequential(
            DepthwiseSeparableConv(256, afr_reduced_cnn_size, kernel_size=3, padding=1),
            CBAM(afr_reduced_cnn_size),
            nn.AdaptiveAvgPool1d(128)  # 统一输出为128点
        )

        self.dim_validator = nn.Sequential(
            nn.Conv1d(256, afr_reduced_cnn_size, kernel_size=1),
            nn.BatchNorm1d(afr_reduced_cnn_size),
            nn.GELU()
        )

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)

        # 打印中间维度用于调试
        print(f"Feature1 shape: {x1.shape}, Feature2 shape: {x2.shape}")

        x_concat = torch.cat((x1, x2), dim=1)  # 在通道维度拼接
        x_concat = self.dim_validator(x_concat)

        return x_concat



# -------------------- 改进的AttnSleep整体架构 --------------------
class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result

class PositionwiseFeedForward(nn.Module):
    "Positionwise feed-forward network."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        "Implements FFN equation."
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class SublayerOutput(nn.Module):
    '''
    A residual connection followed by a layer norm.
    '''

    def __init__(self, size, dropout):
        super(SublayerOutput, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps

        # 参数需要匹配最后一个维度的尺寸
        self.a_2 = nn.Parameter(torch.ones(*normalized_shape))
        self.b_2 = nn.Parameter(torch.zeros(*normalized_shape))

    def forward(self, x):
        # 确保输入维度匹配
        input_shape = x.shape
        assert input_shape[-len(self.normalized_shape):] == self.normalized_shape, \
            f"LayerNorm维度不匹配: 输入形状{input_shape}，期望最后{len(self.normalized_shape)}维为{self.normalized_shape}"

        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class TCE(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, afr_reduced_cnn_size, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward

        # 修改LayerNorm初始化方式
        self.sublayer_output = clones(SublayerOutput(d_model, dropout), 2)
        self.conv = CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size, kernel_size=7)

        # 添加维度适配器
        self.dim_adapter = nn.Sequential(
            nn.Conv1d(afr_reduced_cnn_size, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.GELU()
        ) if afr_reduced_cnn_size != d_model else nn.Identity()

    def forward(self, x_in):
        # 维度适配
        x_in = self.dim_adapter(x_in)
        # 调整维度顺序 [batch, channels, seq] -> [batch, seq, channels]
        x_in = x_in.permute(0, 2, 1)

        query = self.conv(x_in)
        x = self.sublayer_output[0](query, lambda x: self.self_attn(query, x_in, x_in))
        x = x.permute(0, 2, 1)  # 恢复原始维度
        return self.sublayer_output[1](x, self.feed_forward)

class AttnSleep_Improved(nn.Module):
    def __init__(self):
        super().__init__()
        N = 3  # 增加Transformer层数
        d_model = 64  # 增大特征维度
        d_ff = 128
        h = 8  # 增加注意力头数
        dropout = 0.2
        num_classes = 5
        afr_reduced_cnn_size = 64  # 增大特征维度

        self.mrcnn = MRCNN_Improved(afr_reduced_cnn_size)

        attn = MultiHeadedAttention(h, d_model, afr_reduced_cnn_size)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.tce = TCE(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff),
                                    afr_reduced_cnn_size, dropout), N)

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