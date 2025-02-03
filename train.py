import argparse
import collections
import numpy as np
from sklearn.model_selection import KFold
from data_loaders import *
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils.util import *
import torch
import torch.nn as nn

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def nested_cv_main(config, np_data_dir, num_outer_folds=5, num_inner_folds=5):
    # 加载完整数据集
    all_files = [os.path.join(np_data_dir, f) for f in os.listdir(np_data_dir) if f.endswith('.npz')]
    X = np.concatenate([np.load(f)['x'] for f in all_files], axis=0)
    y = np.concatenate([np.load(f)['y'] for f in all_files], axis=0)

    outer_kfold = KFold(n_splits=num_outer_folds, shuffle=True, random_state=SEED)

    outer_fold_results = []
    for outer_fold, (train_val_idx, test_idx) in enumerate(outer_kfold.split(X)):
        logger = config.get_logger(f'nested_cv_outer_{outer_fold}')

        # Outer fold划分
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        y_train_val, y_test = y[train_val_idx], y[test_idx]

        # Inner交叉验证
        inner_kfold = KFold(n_splits=num_inner_folds, shuffle=True, random_state=SEED)
        best_inner_metrics = []

        for inner_fold, (train_idx, val_idx) in enumerate(inner_kfold.split(X_train_val)):
            # 构建数据加载器
            train_files = [f"fold_{outer_fold}_inner_{inner_fold}_train.npz"]
            val_files = [f"fold_{outer_fold}_inner_{inner_fold}_val.npz"]

            # 保存临时数据（实际使用时建议使用内存数据集）
            np.savez(train_files[0], x=X_train_val[train_idx], y=y_train_val[train_idx])
            np.savez(val_files[0], x=X_train_val[val_idx], y=y_train_val[val_idx])

            # 初始化模型
            model = config.init_obj('arch', module_arch)
            model.apply(weights_init_normal)

            # 优化器和损失函数
            trainable_params = filter(lambda p: p.requires_grad, model.parameters())
            optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
            criterion = getattr(module_loss, config['loss'])
            metrics = [getattr(module_metric, met) for met in config['metrics']]

            # 数据加载
            train_loader, val_loader, data_count = data_generator_np(train_files, val_files,
                                                                     config["data_loader"]["args"]["batch_size"])

            # 类别权重计算
            class_weights = calc_class_weight(data_count)

            # 训练器
            inner_trainer = InnerTrainer(model, criterion, metrics, optimizer,
                                         config=config,
                                         data_loader=train_loader,
                                         valid_data_loader=val_loader,
                                         class_weights=class_weights)

            # 训练并验证
            best_metric = inner_trainer.train()
            best_inner_metrics.append(best_metric)

            # 清理临时文件
            os.remove(train_files[0])
            os.remove(val_files[0])

        # 选择最佳内层模型
        avg_best_metric = np.mean([m['val_acc'] for m in best_inner_metrics])
        logger.info(f'Outer Fold {outer_fold} | Avg Inner Val Acc: {avg_best_metric:.4f}')

        # 在完整训练集上训练最终模型
        final_model = config.init_obj('arch', module_arch)
        final_model.apply(weights_init_normal)

        # 重新初始化优化器
        trainable_params = filter(lambda p: p.requires_grad, final_model.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

        # 构建最终数据加载器
        final_train_files = [f"final_train_{outer_fold}.npz"]
        np.savez(final_train_files[0], x=X_train_val, y=y_train_val)
        train_loader, _, _ = data_generator_np(final_train_files, [],
                                               config["data_loader"]["args"]["batch_size"])

        # 最终训练
        final_trainer = Trainer(final_model, criterion, metrics, optimizer,
                                config=config,
                                data_loader=train_loader)
        final_trainer.train()

        # 在测试集上评估
        test_files = [f"test_{outer_fold}.npz"]
        np.savez(test_files[0], x=X_test, y=y_test)
        _, test_loader, _ = data_generator_np([], test_files,
                                              config["data_loader"]["args"]["batch_size"])

        test_metrics = final_trainer.evaluate(test_loader)
        outer_fold_results.append(test_metrics)
        logger.info(f'Outer Fold {outer_fold} | Test Acc: {test_metrics["acc"]:.4f}')

    # 汇总结果
    final_results = {
        'mean_acc': np.mean([r['acc'] for r in outer_fold_results]),
        'std_acc': np.std([r['acc'] for r in outer_fold_results]),
        'f1_scores': np.mean([r['f1'] for r in outer_fold_results], axis=0)
    }
    logger.info(f'Final Nested CV Results | Acc: {final_results["mean_acc"]:.4f} ± {final_results["std_acc"]:.4f}')
    return final_results


class InnerTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_metric = 0.0

    def _train_epoch(self, epoch):
        # 添加渐进式学习率调度
        if self.config['scheduler']['type'] == 'OneCycleLR':
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config['scheduler']['args']['max_lr'],
                steps_per_epoch=len(self.data_loader),
                epochs=self.config['trainer']['epochs']
            )

        return super()._train_epoch(epoch)

    def _valid_epoch(self, epoch):
        log = super()._valid_epoch(epoch)
        if log['acc'] > self.best_metric:
            self.best_metric = log['acc']
        return log


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Nested CV Training')
    args.add_argument('-c', '--config', default="config.json", type=str)
    args.add_argument('-da', '--np_data_dir', required=True, type=str)
    args.add_argument('-o', '--num_outer_folds', default=5, type=int)
    args.add_argument('-i', '--num_inner_folds', default=5, type=int)

    config = ConfigParser.from_args(args)
    nested_cv_main(config,
                   args.parse_args().np_data_dir,
                   num_outer_folds=args.parse_args().num_outer_folds,
                   num_inner_folds=args.parse_args().num_inner_folds)