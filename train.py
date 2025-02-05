import torch
import torch.nn as nn
import numpy as np
import time
import copy
from itertools import islice
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from data_loaders import NestedCVSplitter, create_loaders
from models.model import AttnSleep

# 配置参数
config = {
    'data_dir': r'/home/Wsh/ZYT/Sleep-EDF-20/fpzcz',
    'n_outer_folds': 3,
    'epochs': 50,
    'batch_size': 64,
    'patience': 15,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42
}

# 初始化交叉验证
splitter = NestedCVSplitter(config['data_dir'],
                            n_splits=config['n_outer_folds'],
                            seed=config['seed'])

# 存储所有结果
all_results = []

for outer_fold in range(config['n_outer_folds']):
    print(f"\n=== Processing Outer Fold {outer_fold + 1}/{config['n_outer_folds']} ===")

    # 获取当前fold划分
    fold_data = splitter.get_fold(outer_fold)
    test_files = fold_data['test_files']

    best_inner_models = []

    # 内层交叉验证
    for inner_fold, inner_split in enumerate(fold_data['train_val_splits']):
        print(f"\n--- Inner Fold {inner_fold + 1}/{len(fold_data['train_val_splits'])} ---")

        # 创建数据加载器
        train_loader, val_loader, _ = create_loaders(
            inner_split['train_files'],
            inner_split['val_files'],
            test_files,
            config['batch_size']
        )

        # 初始化模型
        model = AttnSleep().to(config['device'])
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=config['lr'],
                                      weight_decay=config['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0, 1.0, 1.0, 1.0]).to(config['device']))

        best_val_loss = float('inf')
        best_model = None
        patience_counter = 0

        # 训练循环
        for epoch in range(config['epochs']):
            start_time = time.time()

            # 训练阶段
            model.train()
            train_loss = 0
            for x, y in train_loader:
                # 输入形状验证
                assert x.shape[1] == 1, f"Invalid input shape: {x.shape}, expected channel dimension at index 1"

                x = x.to(config['device'], non_blocking=True)
                y = y.to(config['device'], non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item() * x.size(0)

            # 验证阶段
            model.eval()
            val_loss = 0
            preds, truths = [], []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(config['device'])
                    y = y.to(config['device'])

                    outputs = model(x)
                    loss = criterion(outputs, y)

                    val_loss += loss.item() * x.size(0)
                    preds.append(torch.argmax(outputs, 1).cpu())
                    truths.append(y.cpu())

            # 计算指标
            train_loss /= len(train_loader.dataset)
            val_loss /= len(val_loader.dataset)
            preds = torch.cat(preds).numpy()
            truths = torch.cat(truths).numpy()

            val_acc = accuracy_score(truths, preds)
            val_f1 = f1_score(truths, preds, average='macro')

            # 学习率调整
            scheduler.step()

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            # 打印进度
            print(f"Epoch {epoch + 1}/{config['epochs']} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Acc: {val_acc:.4f} | F1: {val_f1:.4f} | "
                  f"Time: {time.time() - start_time:.2f}s")

            if patience_counter >= config['patience']:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # 保存最佳内层模型
        best_inner_models.append(best_model)
        torch.cuda.empty_cache()

    # 外层测试集评估
    print("\n--- Testing on Outer Fold ---")

    # 创建测试集加载器（只传入测试集文件）
    _, _, test_loader = create_loaders(
        train_files=None,  # 显式传递None代替空列表
        val_files=None,
        test_files=test_files,
        batch_size=config['batch_size']
    )

    # 添加空值检查
    if test_loader is None:
        raise RuntimeError("测试集加载失败")

    # 模型集成
    final_preds = []
    truths = []
    for model_state in best_inner_models:
        model.load_state_dict(model_state)
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(config['device'])
                outputs = model(x)
                fold_preds.append(torch.argmax(outputs, 1).cpu().numpy())
                if len(truths) == 0:
                    truths.append(y.numpy())

        final_preds.append(np.concatenate(fold_preds))

    # 投票集成
    final_preds = np.stack(final_preds)
    ensemble_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 0, final_preds)
    truths = np.concatenate(truths)

    # 计算指标
    test_acc = accuracy_score(truths, ensemble_preds)
    test_f1 = f1_score(truths, ensemble_preds, average='macro')
    cm = confusion_matrix(truths, ensemble_preds)

    print(f"Outer Fold {outer_fold + 1} Results:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Macro F1: {test_f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    all_results.append({
        'fold': outer_fold,
        'accuracy': test_acc,
        'f1': test_f1,
        'cm': cm
    })

# 打印最终结果
print("\n=== Final Results ===")
accuracies = [res['accuracy'] for res in all_results]
f1_scores = [res['f1'] for res in all_results]

print(f"Average Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Average Macro F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")