import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import random


class SleepDataset(Dataset):
    def __init__(self, file_list, augment=True):
        self.augment = augment
        self.x_data = []
        self.y_data = []

        if not file_list:
            raise ValueError("输入文件列表不能为空")

        print(f"\n🔧 正在加载 {len(file_list)} 个数据文件...")
        valid_count = 0

        for idx, file in enumerate(file_list, 1):
            try:
                if not os.path.exists(file):
                    print(f"  ⚠️ 文件不存在: {file}")
                    continue

                with np.load(file) as data:
                    x = data['x'].astype(np.float32)
                    y = data['y'].astype(np.int64)

                    # 统一维度处理
                    if x.ndim == 3:
                        if x.shape[1] == 3000 and x.shape[2] == 1:
                            x = x.transpose(0, 2, 1)
                        elif x.shape[1] != 1:
                            raise ValueError(f"非常规三维数据: {x.shape}")
                    elif x.ndim == 2:
                        x = x[:, np.newaxis, :]
                    else:
                        raise ValueError(f"不支持的数据维度: {x.ndim}")

                    self.x_data.append(torch.from_numpy(x))
                    self.y_data.append(torch.from_numpy(y))
                    valid_count += 1

                    if idx % 5 == 0:
                        print(f"  已加载 {idx}/{len(file_list)} 个文件...")

            except Exception as e:
                print(f"  ❌ 加载错误 {file}: {str(e)}")
                continue

        if valid_count == 0:
            raise RuntimeError("没有加载到有效数据")

        self.x_data = torch.cat(self.x_data, dim=0)
        self.y_data = torch.cat(self.y_data, dim=0)

        # 标准化
        self.mean = torch.mean(self.x_data, dim=(0, 2), keepdim=True)
        self.std = torch.std(self.x_data, dim=(0, 2), keepdim=True)
        self.x_data = (self.x_data - self.mean) / (self.std + 1e-8)

        print(f"✅ 成功加载 {len(self)} 个样本")
        print(f"📐 数据形状: {self.x_data.shape}")

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx].clone()
        y = self.y_data[idx]

        if self.augment:
            # 随机缩放 (0.8-1.2倍)
            scale_factor = 0.8 + 0.4 * torch.rand(1)
            x = x * scale_factor

            # 随机片段混合（针对N1阶段增强）
            if random.random() < 0.3 and y == 1:
                mix_idx = random.randint(0, len(self) - 1)
                x_mix, _ = self.x_data[mix_idx], self.y_data[mix_idx]
                alpha = torch.rand(1)
                x = alpha * x + (1 - alpha) * x_mix

        return x.float(), y.long()


class NestedCVSplitter:
    def __init__(self, data_dir, n_splits=5, seed=42):
        self.data_dir = os.path.normpath(data_dir)
        print(f"\n🔍 正在扫描数据目录: {self.data_dir}")

        self.all_files = []
        for fname in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, fname)
            if fname.endswith('.npz') and os.path.isfile(file_path):
                if os.path.getsize(file_path) > 1024:
                    self.all_files.append(file_path)

        if not self.all_files:
            raise FileNotFoundError("没有找到有效的.npz文件")

        print(f"📂 找到 {len(self.all_files)} 个有效数据文件")
        self.n_splits = n_splits
        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def get_fold(self, fold_idx):
        splits = list(self.kfold.split(self.all_files))
        if fold_idx >= len(splits):
            raise ValueError("fold索引超出范围")

        train_val_indices, test_indices = splits[fold_idx]
        test_files = [self.all_files[i] for i in test_indices]

        inner_kfold = KFold(n_splits=self.n_splits - 1, shuffle=True, random_state=fold_idx)
        inner_splits = []

        for inner_train_idx, inner_val_idx in inner_kfold.split(train_val_indices):
            real_train = [train_val_indices[i] for i in inner_train_idx]
            real_val = [train_val_indices[i] for i in inner_val_idx]

            inner_splits.append({
                'train_files': [self.all_files[i] for i in real_train],
                'val_files': [self.all_files[i] for i in real_val]
            })

        return {
            'test_files': test_files,
            'train_val_splits': inner_splits
        }


def create_loaders(train_files, val_files, test_files, batch_size=32):
    def check_files(files, name):
        if not files and name != "测试集":
            return []
        valid_files = [f for f in files if os.path.exists(f)]
        if not valid_files and name == "测试集":
            raise ValueError("测试集不能为空")
        return valid_files

    train_set = SleepDataset(check_files(train_files, "训练集"), augment=True) if train_files else None
    val_set = SleepDataset(check_files(val_files, "验证集"), augment=False) if val_files else None
    test_set = SleepDataset(check_files(test_files, "测试集"), augment=False)

    loader_args = {
        'batch_size': batch_size,
        'num_workers': 0 if os.name == 'nt' else 4,
        'pin_memory': True,
        'persistent_workers': True
    }

    return (
        DataLoader(train_set, shuffle=True, **loader_args) if train_set else None,
        DataLoader(val_set, shuffle=False, **loader_args) if val_set else None,
        DataLoader(test_set, shuffle=False, **loader_args)
    )