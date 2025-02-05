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

        print(f"\n🔧 正在加载 {len(file_list)} 个数据文件...")
        valid_count = 0

        for idx, file in enumerate(file_list, 1):
            try:
                # 检查文件是否存在
                if not os.path.exists(file):
                    print(f"  ⚠️ 文件不存在: {file}")
                    continue

                # 加载数据
                with np.load(file) as data:
                    # 检查必要字段
                    if 'x' not in data or 'y' not in data:
                        print(f"  ⚠️ 文件格式错误: {file} 缺少x/y字段")
                        continue

                    x = data['x'].astype(np.float32)
                    y = data['y'].astype(np.int64)

                    # 统一维度处理
                    if x.ndim == 3:
                        # 情况1: (n, 1, 3000) -> 直接使用
                        if x.shape[1] == 1 and x.shape[2] == 3000:
                            pass
                        # 情况2: (n, 3000, 1) -> 转置为(n, 1, 3000)
                        elif x.shape[1] == 3000 and x.shape[2] == 1:
                            x = x.transpose(0, 2, 1)
                        else:
                            raise ValueError(f"非常规三维数据: {x.shape}")
                    # 处理二维数据 (n, 3000) -> (n, 1, 3000)
                    elif x.ndim == 2:
                        x = x[:, np.newaxis, :]
                    else:
                        raise ValueError(f"不支持的数据维度: {x.ndim}")

                    # 最终形状验证
                    if x.shape[1:] != (1, 3000):
                        print(f"  ⚠️ 维度校验失败: {file} 形状={x.shape}")
                        continue

                    # 转换为Tensor
                    self.x_data.append(torch.from_numpy(x))
                    self.y_data.append(torch.from_numpy(y))
                    valid_count += 1

                    # 进度显示
                    if idx % 5 == 0:
                        print(f"  已加载 {idx}/{len(file_list)} 个文件...")

            except Exception as e:
                print(f"  ❌ 加载错误 {file}: {str(e)}")
                continue

        # 检查有效数据
        if valid_count == 0:
            raise RuntimeError(f"❌ 没有加载到有效数据，请检查文件格式！")

        # 合并所有数据
        self.x_data = torch.cat(self.x_data, dim=0)
        self.y_data = torch.cat(self.y_data, dim=0)

        # 数据标准化
        self.mean = torch.mean(self.x_data, dim=(0, 2), keepdim=True)
        self.std = torch.std(self.x_data, dim=(0, 2), keepdim=True)
        self.x_data = (self.x_data - self.mean) / (self.std + 1e-8)

        # 最终维度验证
        print(f"✅ 成功加载 {len(self)} 个样本")
        print(f"📐 数据形状: {self.x_data.shape}")
        assert self.x_data.dim() == 3, f"数据维度错误: 当前维度{self.x_data.dim()}D，应为3D"
        assert self.x_data.shape[1] == 1, f"通道维度错误: 当前{self.x_data.shape[1]}，应为1"

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx].clone()
        y = self.y_data[idx]

        # 数据增强
        if self.augment:
            # 随机翻转
            if random.random() > 0.5:
                x = torch.flip(x, [-1])
            # 添加高斯噪声
            if random.random() > 0.5:
                x += torch.randn_like(x) * 0.05

        return x.float(), y.long()


class NestedCVSplitter:
    def __init__(self, data_dir, n_splits=5, seed=42):
        # 路径处理
        self.data_dir = os.path.normpath(data_dir)
        print(f"\n🔍 正在扫描数据目录: {self.data_dir}")

        # 获取有效文件列表
        self.all_files = []
        for fname in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, fname)
            if fname.endswith('.npz') and os.path.isfile(file_path):
                if os.path.getsize(file_path) > 1024:  # 1KB
                    self.all_files.append(file_path)
                else:
                    print(f"  ⚠️ 忽略空文件: {fname}")

        if not self.all_files:
            raise FileNotFoundError(f"❌ 目录中没有找到有效的.npz文件: {self.data_dir}")

        print(f"📂 找到 {len(self.all_files)} 个有效数据文件")
        self.n_splits = n_splits
        self.kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def get_fold(self, fold_idx):
        # 外层划分
        splits = list(self.kfold.split(self.all_files))
        if fold_idx >= len(splits):
            raise ValueError(f"fold索引{fold_idx}超出范围(总fold数{len(splits)})")

        train_val_indices, test_indices = splits[fold_idx]

        # 添加路径验证
        test_files = [self.all_files[i] for i in test_indices]
        if not test_files:
            raise ValueError(f"第{fold_idx + 1}折没有找到测试集文件")

        # 内层划分
        inner_kfold = KFold(n_splits=self.n_splits - 1, shuffle=True, random_state=fold_idx)
        inner_splits = []

        for inner_train_idx, inner_val_idx in inner_kfold.split(train_val_indices):
            # 转换到原始索引
            real_train = [train_val_indices[i] for i in inner_train_idx]
            real_val = [train_val_indices[i] for i in inner_val_idx]

            inner_splits.append({
                'train_files': [self.all_files[i] for i in real_train],
                'val_files': [self.all_files[i] for i in real_val]
            })

        return {
            'test_files': [self.all_files[i] for i in test_indices],
            'train_val_splits': inner_splits
        }


def create_loaders(train_files, val_files, test_files, batch_size=32):
    # 添加空列表检查
    def check_files(file_list, name):
        # 允许测试阶段的空列表
        if not file_list and name != "测试集":
            raise ValueError(f"{name}文件列表为空！")
        valid_files = [f for f in file_list if os.path.exists(f)]
        if not valid_files and name == "测试集":
            raise ValueError("测试集不能为空")
        return valid_files

    # 训练集和验证集允许为空（仅在测试阶段）
    train_set = None
    if train_files:
        train_set = SleepDataset(check_files(train_files, "训练集"), augment=True)

    val_set = None
    if val_files:
        val_set = SleepDataset(check_files(val_files, "验证集"), augment=False)

    # 测试集必须存在
    test_set = SleepDataset(check_files(test_files, "测试集"), augment=False)

    loader_args = {
        'batch_size': batch_size,
        'num_workers': 4 if os.cpu_count() > 4 else 2,
        'pin_memory': True,
        'persistent_workers': True
    }

    return (
        DataLoader(train_set, shuffle=True, **loader_args) if train_set else None,
        DataLoader(val_set, shuffle=False, **loader_args) if val_set else None,
        DataLoader(test_set, shuffle=False, **loader_args)
    )