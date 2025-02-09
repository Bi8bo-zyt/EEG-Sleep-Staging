import os
import torch
import numpy as np
import time
import copy
from itertools import islice
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from data_loaders import NestedCVSplitter, create_loaders
from models.model import AttnSleep, FocalLoss

# 配置参数
config = {
    'data_dir': r'/home/Wsh/ZYT/EEG-Sleep-Staging/data',
    'checkpoint_dir': r'/home/Wsh/ZYT/EEG-Sleep-Staging/checkpoints',  # 新增检查点目录
    'n_outer_folds': 5,
    'epochs': 50,
    'batch_size': 256,
    'patience': 10,
    'lr': 3e-4,
    'weight_decay': 1e-3,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'class_weights': [1.0, 5.0, 1.0, 2.0, 3.0],  # W,N1,N2,N3,REM
    'focal_loss': True,
    'gamma': 2,
    'seed': 42,
    'resume': True  # 是否启用断点续训
}


def save_checkpoint(state, filename='checkpoint.pth'):
    """保存训练状态"""
    torch.save(state, os.path.join(config['checkpoint_dir'], filename))
    print(f"✅ 检查点已保存: {filename}")


def load_checkpoint():
    """加载最近的检查点"""
    try:
        # 获取最新检查点
        checkpoints = [f for f in os.listdir(config['checkpoint_dir']) if f.endswith('.pth')]
        if not checkpoints:
            return None
        latest = max(checkpoints, key=lambda x: os.path.getmtime(os.path.join(config['checkpoint_dir'], x)))

        checkpoint = torch.load(os.path.join(config['checkpoint_dir'], latest))
        print(f"🔍 从检查点恢复: {latest}")
        return checkpoint
    except Exception as e:
        print(f"❌ 加载检查点失败: {str(e)}")
        return None


def main():
    # 创建检查点目录
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    # 初始化交叉验证
    splitter = NestedCVSplitter(config['data_dir'],
                                n_splits=config['n_outer_folds'],
                                seed=config['seed'])

    # 尝试加载全局检查点
    global_state = None
    if config['resume']:
        global_state = load_checkpoint()

    # 初始化结果存储
    all_results = []
    start_outer_fold = 0
    start_inner_fold = 0

    # 恢复全局状态
    if global_state:
        all_results = global_state['results']
        start_outer_fold = global_state['outer_fold']
        start_inner_fold = global_state['inner_fold']
        print(f"↩️ 从第 {start_outer_fold + 1} 折外层, 第 {start_inner_fold + 1} 折内层恢复")

    # 外层交叉验证循环
    for outer_fold in range(start_outer_fold, config['n_outer_folds']):
        print(f"\n=== Processing Outer Fold {outer_fold + 1}/{config['n_outer_folds']} ===")

        fold_data = splitter.get_fold(outer_fold)
        test_files = fold_data['test_files']

        best_inner_models = []

        # 内层交叉验证
        for inner_idx, inner_split in enumerate(fold_data['train_val_splits']):
            if outer_fold == start_outer_fold and inner_idx < start_inner_fold:
                continue  # 跳过已完成的inner fold

            start_inner_fold = 0  # 重置内层索引
            print(f"\n--- Inner Fold {inner_idx + 1}/{len(fold_data['train_val_splits'])} ---")

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

            # 改进的损失函数
            if config['focal_loss']:
                # 转换类别权重为Tensor
                class_weights = torch.tensor(config['class_weights'],
                                             device=config['device'])
                criterion = FocalLoss(alpha=class_weights,
                                      gamma=config['gamma'])
            else:
                class_weights = torch.tensor(config['class_weights'],
                                             device=config['device'])
                criterion = nn.CrossEntropyLoss(weight=class_weights)

            # 尝试加载模型检查点
            start_epoch = 0
            best_val_loss = float('inf')
            if config['resume']:
                checkpoint = load_checkpoint()
                if checkpoint and checkpoint['outer_fold'] == outer_fold and checkpoint['inner_fold'] == inner_idx:
                    model.load_state_dict(checkpoint['model'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    scheduler.load_state_dict(checkpoint['scheduler'])
                    start_epoch = checkpoint['epoch'] + 1
                    best_val_loss = checkpoint['best_val_loss']
                    print(f"↩️ 从第 {start_epoch} 轮恢复训练")

            # 训练循环
            best_model = None
            patience_counter = 0
            for epoch in range(start_epoch, config['epochs']):
                start_time = time.time()

                # 训练步骤
                model.train()
                train_loss = 0
                for x, y in train_loader:
                    x = x.to(config['device'])
                    y = y.to(config['device'])

                    optimizer.zero_grad()
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * x.size(0)

                # 验证步骤
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
                train_loss = train_loss / len(train_loader.dataset)
                val_loss = val_loss / len(val_loader.dataset)
                preds = torch.cat(preds).numpy()
                truths = torch.cat(truths).numpy()

                val_acc = accuracy_score(truths, preds)
                val_f1 = f1_score(truths, preds, average='macro')

                # 学习率调整
                scheduler.step()

                # 保存检查点
                checkpoint = {
                    'outer_fold': outer_fold,
                    'inner_fold': inner_idx,
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'results': all_results
                }
                save_checkpoint(checkpoint, f'fold_{outer_fold}_inner_{inner_idx}_epoch_{epoch}.pth')

                # 早停机制
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1

                # 打印进度
                print(f"Epoch {epoch + 1}/{config['epochs']} | "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Acc: {val_acc:.4f} | F1: {val_f1:.4f} | Time: {time.time() - start_time:.2f}s")

                if patience_counter >= config['patience']:
                    print(f"⏹️ 第 {epoch + 1} 轮触发早停")
                    break

            # 保存最佳内层模型
            best_inner_models.append(best_model)

            # 保存全局状态
            global_state = {
                'outer_fold': outer_fold,
                'inner_fold': inner_idx,
                'results': all_results
            }
            save_checkpoint(global_state, 'global_state.pth')

        # 外层测试集评估
        print("\n--- Testing on Outer Fold ---")
        _, _, test_loader = create_loaders(
            train_files=None,
            val_files=None,
            test_files=test_files,
            batch_size=config['batch_size']
        )

        # 模型集成
        final_preds = []
        truths = None  # 初始化truths变量

        for model_state in best_inner_models:
            model.load_state_dict(model_state)
            model.eval()

            fold_preds = []
            fold_truths = []

            with torch.no_grad():
                for x, y in test_loader:
                    x = x.to(config['device'])
                    outputs = model(x)
                    fold_preds.append(torch.argmax(outputs, 1).cpu().numpy())
                    fold_truths.append(y.numpy())

            # 确保每个模型的预测结果一致
            fold_preds = np.concatenate(fold_preds)
            fold_truths = np.concatenate(fold_truths)

            if truths is None:
                truths = fold_truths
            else:
                # 验证不同模型的真实标签是否一致
                assert np.array_equal(truths, fold_truths), "不同模型的真实标签不一致"

            final_preds.append(fold_preds)

        # 投票集成
        final_preds = np.stack(final_preds)
        ensemble_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 0, final_preds)

        # 最终验证
        assert len(truths) == len(ensemble_preds), f"标签数量不匹配: {len(truths)} vs {len(ensemble_preds)}"

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

        save_checkpoint({'results': all_results}, 'final_results.pth')

    # 打印最终结果
    print("\n=== Final Results ===")
    accuracies = [res['accuracy'] for res in all_results]
    f1_scores = [res['f1'] for res in all_results]

    print(f"Average Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Average Macro F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")


if __name__ == '__main__':
    main()