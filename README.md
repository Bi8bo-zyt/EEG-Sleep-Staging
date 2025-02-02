# EEG-Sleep-Staging 🧠💤
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-%23EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Under Review at IEEE JBHI** | [预印本链接](https://arxiv.org/abs/xxxx) | [演示视频](https://youtu.be/xxxx)

基于多尺度注意力机制的睡眠分期模型，在Sleep-EDF数据集上达到**81%的Cohen's Kappa系数**。

## 📚 目录
- [创新点](#-创新点)
- [快速开始](#-快速开始)
- [可视化结果](#-可视化结果)
- [引用](#-引用)

## 🌟 创新点
1. **多尺度时空注意力**  
   - 低频段（δ/θ波）采用空洞卷积+多头注意力
   - 高频段（β波）使用因果局部注意力
   ```python
   # 核心代码片段
   class MultiScaleAttention(nn.Module):
       def __init__(self):
           self.low_att = DilatedAttention(dilation=4)
           self.high_att = LocalAttention(window_size=16)
   ```

2. **时频域混合数据增强**  
   - 时域：非平稳噪声注入
   - 频域：随机频段遮蔽
   ![增强对比图](docs/aug_comparison.png)

## ⚡ 快速开始
### 环境配置
```bash
git clone https://github.com/yourname/EEG-Sleep-Staging.git
pip install -r requirements.txt
```

### 训练模型
```python
python src/train.py --dataset Sleep-EDF --model ms_att
```

### 部署到树莓派
```bash
docker build -t sleep_stage .
docker run -it --device /dev/ttyACM0 sleep_stage
```

## 📊 可视化结果
| 模型架构 | 混淆矩阵 | 注意力热力图 |
|----------|----------|--------------|
| ![架构图](docs/architecture.png) | ![混淆矩阵](docs/confusion_matrix.png) | ![热力图](docs/heatmap.png) |

## 📄 引用
```bibtex
@article{yourname2023eeg,
  title={Multi-Scale Attention Network for Sleep Stage Classification},
  author={Your Name, Your Advisor},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2023}
}
```
