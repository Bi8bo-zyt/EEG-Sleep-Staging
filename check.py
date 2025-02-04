# 临时测试代码
from data_loaders import SleepDataset
import glob

dataset = SleepDataset(glob.glob(r'E:\science\EEG-Sleep-Staging\data\*.npz'))
sample = dataset[0]
print(f"单个样本形状: {sample[0].shape}")  # 应为 torch.Size([1, 3000])