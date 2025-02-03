import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, StratifiedKFold
from models.model import AttnSleep_Improved  # 导入你改进后的模型
from data_loaders import LoadDataset_from_numpy
import time
import json
import copy


# 实验配置
class Config:
    data_path = r"E:\science\EEG-Sleep-Staging\data"  # 数据路径
    num_classes = 5
    batch_size = 64
    epochs = 100
    lr = 1e-4
    weight_decay = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    save_dir = "./experiments"
    outer_folds = 5  # 外层交叉验证折数
    inner_folds = 5  # 内层交叉验证折数


# 固定随机种子
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_data_files(data_path):
    """加载所有npz文件路径"""
    all_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".npz"):
                all_files.append(os.path.join(root, file))
    return all_files


class Trainer:
    def __init__(self, config, train_files, val_files):
        self.config = config
        self.train_dataset = LoadDataset_from_numpy(train_files)
        self.val_dataset = LoadDataset_from_numpy(val_files)

        # 计算类别权重
        all_labels = np.concatenate([np.load(f)["y"] for f in train_files])
        class_counts = np.bincount(all_labels)
        class_weights = 1. / class_counts
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(config.device)

        self.model = AttnSleep_Improved().to(config.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=config.lr,
                                           weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        correct = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(self.config.device)
            labels = labels.to(self.config.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

        return total_loss / len(train_loader), correct / len(self.train_dataset)

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.config.device)
                labels = labels.to(self.config.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()

        return total_loss / len(data_loader), correct / len(self.val_dataset)

    def train(self, fold_idx, inner_fold=None):
        best_val_acc = 0.0
        early_stop_counter = 0

        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.config.batch_size,
                                  shuffle=True,
                                  num_workers=4)
        val_loader = DataLoader(self.val_dataset,
                                batch_size=self.config.batch_size,
                                shuffle=False,
                                num_workers=4)

        history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        for epoch in range(self.config.epochs):
            start_time = time.time()
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # 早停机制
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                early_stop_counter = 0
                # 保存最佳模型
                model_name = f"fold{fold_idx}_best.pth"
                if inner_fold is not None:
                    model_name = f"outer{fold_idx}_inner{inner_fold}_best.pth"
                torch.save(self.model.state_dict(),
                           os.path.join(self.config.save_dir, model_name))
            else:
                early_stop_counter += 1
                if early_stop_counter >= 15:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            print(f"Fold {fold_idx} Epoch {epoch + 1}/{self.config.epochs} | "
                  f"Time: {time.time() - start_time:.1f}s | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        return history, best_val_acc


def nested_cross_validation(config):
    set_seed(config.seed)
    os.makedirs(config.save_dir, exist_ok=True)

    all_files = load_data_files(config.data_path)
    file_labels = [np.load(f)["y"][0] for f in all_files]  # 获取每个文件的标签用于分层划分

    # 外层交叉验证
    outer_kfold = StratifiedKFold(n_splits=config.outer_folds, shuffle=True, random_state=config.seed)
    outer_results = []

    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_kfold.split(all_files, file_labels)):
        # 划分外层训练集和测试集
        outer_train_files = [all_files[i] for i in outer_train_idx]
        outer_test_files = [all_files[i] for i in outer_test_idx]

        # 内层交叉验证
        inner_kfold = StratifiedKFold(n_splits=config.inner_folds, shuffle=True, random_state=config.seed)
        inner_best_acc = 0.0
        best_inner_model = None

        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
                inner_kfold.split(outer_train_files, [file_labels[i] for i in outer_train_idx])):

            # 划分内层训练集和验证集
            inner_train_files = [outer_train_files[i] for i in inner_train_idx]
            inner_val_files = [outer_train_files[i] for i in inner_val_idx]

            # 训练内层模型
            trainer = Trainer(config, inner_train_files, inner_val_files)
            history, val_acc = trainer.train(outer_fold, inner_fold)

            if val_acc > inner_best_acc:
                inner_best_acc = val_acc
                best_inner_model = copy.deepcopy(trainer.model.state_dict())

        # 使用最佳内层模型初始化外层模型
        final_model = Trainer(config, outer_train_files, outer_test_files)
        final_model.model.load_state_dict(best_inner_model)

        # 在外层测试集上评估
        test_loader = DataLoader(LoadDataset_from_numpy(outer_test_files),
                                 batch_size=config.batch_size,
                                 shuffle=False)
        test_loss, test_acc = final_model.evaluate(test_loader)
        outer_results.append(test_acc)

        # 保存结果
        result = {
            "outer_fold": outer_fold,
            "test_acc": test_acc,
            "inner_best_acc": inner_best_acc
        }
        with open(os.path.join(config.save_dir, f"result_outer{outer_fold}.json"), "w") as f:
            json.dump(result, f)

        print(f"Outer Fold {outer_fold} | Test Acc: {test_acc:.4f}")

    # 汇总最终结果
    print("\nFinal Results:")
    print(f"Mean Accuracy: {np.mean(outer_results):.4f} ± {np.std(outer_results):.4f}")


if __name__ == "__main__":
    config = Config()
    nested_cross_validation(config)