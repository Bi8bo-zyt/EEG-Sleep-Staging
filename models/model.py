import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from copy import deepcopy


# ---------------------- 改进模块1：CBAM注意力机制 ----------------------
class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x)
        x = x * ca
        sa = self.spatial_attention(torch.cat([x.mean(dim=1, keepdim=True), x.max(dim=1, keepdim=True)[0]], dim=1))
        return x * sa


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.cbam = CBAM(planes, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return F.relu(out)

# ---------------------- 改进模块2：深度监督机制 ----------------------
class DeepSupervisionHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.layer(x)


# ---------------------- 核心模型结构改进 ----------------------
class MRCNN(nn.Module):
    def __init__(self, afr_reduced_cnn_size):
        super(MRCNN, self).__init__()
        drate = 0.5

        self.features1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, padding=25, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(8, 2, padding=4),
            nn.Dropout(drate),
            self._make_layer(SEBasicBlock, 64, 128, stride=1),
            nn.MaxPool1d(4, 4, padding=2)
        )

        self.features2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50, padding=200, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(4, 2, padding=2),
            nn.Dropout(drate),
            self._make_layer(SEBasicBlock, 64, 128, stride=1),
            nn.MaxPool1d(2, 2, padding=1)
        )

        # 通道调整层
        self.channel_adjust = nn.Sequential(
            nn.Conv1d(256, afr_reduced_cnn_size, kernel_size=1),
            nn.BatchNorm1d(afr_reduced_cnn_size),
            nn.GELU()
        )

        # 自适应融合模块
        self.AFR = self._make_layer(SEBasicBlock, afr_reduced_cnn_size, afr_reduced_cnn_size, stride=1)

    def _make_layer(self, block, inplanes, planes, stride):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes)
            )
        return block(inplanes, planes, stride, downsample)

    def forward(self, x):
        x1 = self.features1(x)  # [B, 128, L1]
        x2 = self.features2(x)  # [B, 128, L2]

        # 统一序列长度
        max_len = max(x1.size(2), x2.size(2))
        x1 = F.pad(x1, (0, max_len - x1.size(2)))
        x2 = F.pad(x2, (0, max_len - x2.size(2)))

        # 合并特征
        x_concat = torch.cat([x1, x2], dim=1)  # [B, 256, L]

        # 通道调整
        x_concat = self.channel_adjust(x_concat)  # [B, afr_size, L]

        # 特征融合
        x_concat = self.AFR(x_concat)
        return x_concat


# ---------------------- 改进模块3：增强的Transformer ----------------------
class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.d_model = d_model
        self.pe = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x):
        seq_len = x.size(1)
        pe = self.pe[:seq_len, :].unsqueeze(0)
        return x + pe


class EnhancedTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, afr_size):
        super().__init__()
        self.pos_encoder = RelativePositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.adapt_conv = nn.Conv1d(afr_size, d_model, 1)

    def forward(self, x):
        x = self.adapt_conv(x)
        x = x.permute(2, 0, 1)  # [seq_len, batch, features]
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return x.permute(1, 2, 0)


# ---------------------- 最终模型整合 ----------------------
class AttnSleep(nn.Module):
    def __init__(self):
        super(AttnSleep, self).__init__()
        afr_size = 30
        d_model = 128
        num_classes = 5

        self.mrcnn = MRCNN(afr_size)
        self.transformer = EnhancedTransformer(
            d_model=d_model,
            nhead=8,
            num_layers=4,
            afr_size=afr_size
        )

        # 双向LSTM时序建模
        self.lstm = nn.LSTM(d_model, 128,
                            bidirectional=True,
                            batch_first=True,
                            num_layers=2)

        # 深度监督头
        self.ds_head = DeepSupervisionHead(256, num_classes)

        # 主分类器
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # 调整输入维度顺序
        if x.dim() == 3 and x.size(2) == 1:  # 输入形状为 [batch, seq_len, 1]
            x = x.permute(0, 2, 1)  # 转换为 [batch, 1, seq_len]
        elif x.dim() == 2:  # 输入形状为 [batch, seq_len]
            x = x.unsqueeze(1)  # 转换为 [batch, 1, seq_len]

        # 特征提取
        x = self.mrcnn(x)

        # Transformer编码
        x = self.transformer(x)

        # LSTM时序建模
        x = x.permute(0, 2, 1)  # [batch, seq, features]
        x, _ = self.lstm(x)

        # 深度监督
        ds_output = self.ds_head(x[:, -1, :])

        # 注意力池化
        attn_weights = torch.softmax(torch.matmul(x, x.mean(dim=1, keepdim=True).transpose(1, 2)), dim=1)
        context = torch.sum(attn_weights * x, dim=1)

        # 最终分类
        main_output = self.classifier(context)
        return main_output + 0.3 * ds_output  # 加权融合