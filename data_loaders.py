import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight


class EnhancedSleepDataset(Dataset):
    """支持数据增强和动态标准化的增强型数据集加载器"""

    def __init__(self, file_list, mode='train', augment_prob=0.5, noise_scale=0.1):
        """
        Args:
            file_list (list): .npz文件路径列表
            mode (str): 数据集模式 [train/val/test]
            augment_prob (float): 数据增强应用概率
            noise_scale (float): 高斯噪声标准差系数
        """
        self.mode = mode
        self.augment_prob = augment_prob if mode == 'train' else 0.0
        self.noise_scale = noise_scale

        # 加载并整合所有数据
        self.x_data, self.y_data = self._load_and_process(file_list)

        # 计算全局标准化参数（基于训练集）
        if mode == 'train':
            self.mean = np.mean(self.x_data)
            self.std = np.std(self.x_data)
        else:
            self.mean = None
            self.std = None

        # 转换为Tensor
        self.x_data = torch.FloatTensor(self.x_data)
        self.y_data = torch.LongTensor(self.y_data)

        # 调整维度 (batch, channel, seq_len)
        if len(self.x_data.shape) == 2:
            self.x_data = self.x_data.unsqueeze(1)
        elif self.x_data.shape[1] != 1:
            self.x_data = self.x_data.permute(0, 2, 1)

    def _load_and_process(self, file_list):
        """加载并预处理数据"""
        x_list, y_list = [], []
        for file_path in file_list:
            with np.load(file_path) as data:
                x = data['x']
                y = data['y']

                # 数据清洗：移除NaN值
                valid_idx = ~np.isnan(x).any(axis=1)
                x = x[valid_idx]
                y = y[valid_idx]

                x_list.append(x)
                y_list.append(y)

        return np.vstack(x_list), np.concatenate(y_list)

    def _augment(self, x):
        """应用数据增强"""
        if torch.rand(1) < self.augment_prob:
            # 高斯噪声
            x += torch.randn_like(x) * self.noise_scale * x.std()

        if torch.rand(1) < self.augment_prob:
            # 随机时移
            shift = torch.randint(-10, 10, (1,)).item()
            x = torch.roll(x, shifts=shift, dims=1)

        if torch.rand(1) < self.augment_prob:
            # 随机缩放
            scale = torch.FloatTensor(1).uniform_(0.8, 1.2)
            x *= scale

        return x

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]

        # 标准化
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std

        # 数据增强
        if self.mode == 'train':
            x = self._augment(x)

        return x, y

    def __len__(self):
        return len(self.y_data)

    def get_class_weights(self):
        """计算类别权重用于损失函数"""
        y_np = self.y_data.numpy()
        classes = np.unique(y_np)
        weights = compute_class_weight('balanced', classes=classes, y=y_np)
        return torch.FloatTensor(weights)


def create_data_loaders(train_files, val_files, test_files, batch_size=64,
                        num_workers=4, augment_prob=0.5):
    """
    创建数据加载器三元组
    Returns:
        (train_loader, val_loader, test_loader, class_weights)
    """
    # 创建数据集
    train_dataset = EnhancedSleepDataset(train_files, mode='train',
                                         augment_prob=augment_prob)
    val_dataset = EnhancedSleepDataset(val_files, mode='val')
    test_dataset = EnhancedSleepDataset(test_files, mode='test')

    # 获取类别权重
    class_weights = train_dataset.get_class_weights()

    # 创建DataLoader
    loader_args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
        'persistent_workers': True
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    return train_loader, val_loader, test_loader, class_weights


def stratified_kfold_split(file_list, n_splits=5, seed=42):
    """分层K折划分（文件级别）"""
    from sklearn.model_selection import StratifiedKFold
    # 获取每个文件的标签（取第一个样本的标签）
    file_labels = [np.load(f)['y'][0] for f in file_list]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(file_list, file_labels):
        yield [file_list[i] for i in train_idx], [file_list[i] for i in test_idx]


# 使用示例
if __name__ == "__main__":
    # 加载所有数据文件
    data_dir = "E:\\science\\EEG-Sleep-Staging\\data"
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npz')]

    # 进行5折划分
    for fold, (train_files, test_files) in enumerate(stratified_kfold_split(all_files)):
        print(f"Fold {fold + 1}:")
        print(f"  Train files: {len(train_files)}")
        print(f"  Test files: {len(test_files)}")

        # 创建数据加载器
        train_loader, val_loader, test_loader, class_weights = create_data_loaders(
            train_files, [], test_files, batch_size=64
        )

        # 验证数据流
        for x, y in train_loader:
            print(f"Batch shape: {x.shape}, Labels: {y.shape}")
            break