"""
ToxD4C 真实数据训练脚本
使用真实的LMDB数据集进行训练

作者: AI助手
日期: 2024-06-11
"""

import os
import sys
import json
import logging
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
import traceback

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from data.lmdb_dataset import create_lmdb_dataloaders
from models.toxd4c import ToxD4C
from configs.toxd4c_config import get_enhanced_toxd4c_config

# 设置警告过滤
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_real_data.log')
    ]
)
logger = logging.getLogger(__name__)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def check_for_nan_inf(tensor, name="tensor"):
    """检查张量中的NaN和Inf"""
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan:
        logger.warning(f"{name} contains NaN values!")
    if has_inf:
        logger.warning(f"{name} contains Inf values!")
    
    return has_nan or has_inf


def safe_loss_computation(pred, target, mask, loss_fn):
    """安全的损失计算"""
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    # 只计算有效标签的损失
    valid_pred = pred[mask]
    valid_target = target[mask]
    
    # 检查NaN
    if check_for_nan_inf(valid_pred, "prediction") or check_for_nan_inf(valid_target, "target"):
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    loss = loss_fn(valid_pred, valid_target)
    
    # 检查损失
    if check_for_nan_inf(loss, "loss"):
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    
    return loss


def compute_metrics(predictions, targets, masks, task_type='classification'):
    """计算评估指标"""
    metrics = {}
    
    for task_idx in range(predictions.shape[1]):
        task_pred = predictions[:, task_idx]
        task_target = targets[:, task_idx]
        task_mask = masks[:, task_idx]
        
        if task_mask.sum() == 0:
            continue
        
        valid_pred = task_pred[task_mask].cpu().numpy()
        valid_target = task_target[task_mask].cpu().numpy()
        
        if len(valid_pred) == 0:
            continue
        
        try:
            if task_type == 'classification':
                # 二分类指标
                pred_binary = (valid_pred > 0.5).astype(int)
                target_binary = valid_target.astype(int)
                
                acc = accuracy_score(target_binary, pred_binary)
                metrics[f'task_{task_idx}_accuracy'] = acc
                
                if len(np.unique(target_binary)) > 1:
                    auc = roc_auc_score(target_binary, valid_pred)
                    metrics[f'task_{task_idx}_auc'] = auc
            else:
                # 回归指标
                mse = mean_squared_error(valid_target, valid_pred)
                # 检查真实标签的方差以避免R²为NaN
                if np.var(valid_target) < 1e-6:
                    r2 = 0.0
                else:
                    r2 = r2_score(valid_target, valid_pred)
                rmse = np.sqrt(mse)
                
                metrics[f'task_{task_idx}_mse'] = mse
                metrics[f'task_{task_idx}_rmse'] = rmse
                metrics[f'task_{task_idx}_r2'] = r2
        except Exception as e:
            logger.warning(f"Error computing metrics for task {task_idx}: {e}")
            continue
    
    return metrics


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    num_batches = 0
    
    classification_criterion = nn.BCEWithLogitsLoss(reduction='none')
    regression_criterion = nn.MSELoss(reduction='none')
    
    for batch_idx, batch in enumerate(dataloader):
        try:
            # 移动数据到设备
            data = {
                'atom_features': batch['atom_features'].to(device),
                'edge_index': batch['edge_index'].to(device),
                'coordinates': batch['coordinates'].to(device),
                'batch': batch['batch'].to(device)
            }
            
            cls_labels = batch['classification_labels'].to(device)
            reg_labels = batch['regression_labels'].to(device)
            cls_mask = batch['classification_mask'].to(device)
            reg_mask = batch['regression_mask'].to(device)
            smiles_list = batch['smiles']
            
            # 前向传播
            optimizer.zero_grad()
            
            outputs = model(data, smiles_list)
            
            # 检查输出
            cls_preds = outputs['predictions']['classification']
            reg_preds = outputs['predictions']['regression']

            if check_for_nan_inf(cls_preds, "classification_output"):
                logger.warning(f"NaN in classification output at batch {batch_idx}")
                continue
            
            if check_for_nan_inf(reg_preds, "regression_output"):
                logger.warning(f"NaN in regression output at batch {batch_idx}")
                continue
            
            # 计算损失
            cls_loss = safe_loss_computation(
                cls_preds, cls_labels, cls_mask,
                lambda p, t: classification_criterion(p, t).mean()
            )
            
            reg_loss = safe_loss_computation(
                reg_preds, reg_labels, reg_mask,
                lambda p, t: regression_criterion(p, t).mean()
            )
            
            total_loss_batch = cls_loss + reg_loss
            
            # 检查总损失
            if check_for_nan_inf(total_loss_batch, "total_loss"):
                logger.warning(f"NaN in total loss at batch {batch_idx}")
                continue
            
            # 反向传播
            total_loss_batch.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # 检查梯度
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and check_for_nan_inf(param.grad, f"grad_{name}"):
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                logger.warning(f"NaN in gradients at batch {batch_idx}")
                continue
            
            optimizer.step()
            
            # 更新学习率
            if scheduler is not None:
                scheduler.step()
            
            # 更新统计
            total_loss += total_loss_batch.item()
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx}/{len(dataloader)}: "
                          f"Loss={total_loss_batch.item():.4f}, "
                          f"Cls={cls_loss.item():.4f}, "
                          f"Reg={reg_loss.item():.4f}")
        
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {e}")
            logger.error(traceback.format_exc())
            continue
    
    if num_batches == 0:
        return 0.0, 0.0, 0.0
    
    return total_loss / num_batches, total_cls_loss / num_batches, total_reg_loss / num_batches


def evaluate_model(model, dataloader, device):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    num_batches = 0
    
    all_cls_preds = []
    all_cls_targets = []
    all_cls_masks = []
    all_reg_preds = []
    all_reg_targets = []
    all_reg_masks = []
    
    classification_criterion = nn.BCEWithLogitsLoss(reduction='none')
    regression_criterion = nn.MSELoss(reduction='none')
    
    with torch.no_grad():
        for batch in dataloader:
            try:
                # 移动数据到设备
                data = {
                    'atom_features': batch['atom_features'].to(device),
                    'edge_index': batch['edge_index'].to(device),
                    'coordinates': batch['coordinates'].to(device),
                    'batch': batch['batch'].to(device)
                }
                
                cls_labels = batch['classification_labels'].to(device)
                reg_labels = batch['regression_labels'].to(device)
                cls_mask = batch['classification_mask'].to(device)
                reg_mask = batch['regression_mask'].to(device)
                smiles_list = batch['smiles']
                
                # 前向传播
                outputs = model(data, smiles_list)
                
                # 检查输出
                cls_preds = outputs['predictions']['classification']
                reg_preds = outputs['predictions']['regression']

                if check_for_nan_inf(cls_preds, "eval_classification"):
                    continue
                if check_for_nan_inf(reg_preds, "eval_regression"):
                    continue
                
                # 计算损失
                cls_loss = safe_loss_computation(
                    cls_preds, cls_labels, cls_mask,
                    lambda p, t: classification_criterion(p, t).mean()
                )
                
                reg_loss = safe_loss_computation(
                    reg_preds, reg_labels, reg_mask,
                    lambda p, t: regression_criterion(p, t).mean()
                )
                
                total_loss += (cls_loss + reg_loss).item()
                total_cls_loss += cls_loss.item()
                total_reg_loss += reg_loss.item()
                num_batches += 1
                
                # 收集预测结果
                cls_probs = torch.sigmoid(cls_preds)
                all_cls_preds.append(cls_probs.cpu())
                all_cls_targets.append(cls_labels.cpu())
                all_cls_masks.append(cls_mask.cpu())
                
                all_reg_preds.append(reg_preds.cpu())
                all_reg_targets.append(reg_labels.cpu())
                all_reg_masks.append(reg_mask.cpu())
                
            except Exception as e:
                logger.error(f"Error in evaluation batch: {e}")
                continue
    
    if num_batches == 0:
        return {}, 0.0, 0.0, 0.0
    
    # 计算指标
    metrics = {}
    
    if all_cls_preds:
        cls_preds = torch.cat(all_cls_preds, dim=0)
        cls_targets = torch.cat(all_cls_targets, dim=0)
        cls_masks = torch.cat(all_cls_masks, dim=0)
        
        cls_metrics = compute_metrics(cls_preds, cls_targets, cls_masks, 'classification')
        metrics.update(cls_metrics)
    
    if all_reg_preds:
        reg_preds = torch.cat(all_reg_preds, dim=0)
        reg_targets = torch.cat(all_reg_targets, dim=0)
        reg_masks = torch.cat(all_reg_masks, dim=0)
        
        reg_metrics = compute_metrics(reg_preds, reg_targets, reg_masks, 'regression')
        metrics.update(reg_metrics)
    
    avg_loss = total_loss / num_batches
    avg_cls_loss = total_cls_loss / num_batches
    avg_reg_loss = total_reg_loss / num_batches
    
    return metrics, avg_loss, avg_cls_loss, avg_reg_loss


def main():
    parser = argparse.ArgumentParser(description='ToxScan Enhanced 真实数据训练')
    parser.add_argument('--data_dir', type=str, default='data/dataset', help='LMDB数据目录')
    parser.add_argument('--experiment_name', type=str, default='toxscan_enhanced_real', help='实验名称')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--max_atoms', type=int, default=64, help='最大原子数')
    parser.add_argument('--warmup_ratio', type=float, default=0.06, help='学习率预热比例')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU总内存: {gpu_memory:.2f} GB")
    
    # 创建输出目录
    output_dir = Path("checkpoints_real")
    output_dir.mkdir(exist_ok=True)
    
    # 获取配置
    config = get_enhanced_toxd4c_config()
    config['batch_size'] = args.batch_size
    
    # 保存配置
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"配置已保存: {config_path}")
    
    # 创建数据加载器
    logger.info(f"加载LMDB数据集: {args.data_dir}")
    try:
        train_loader, valid_loader, test_loader = create_lmdb_dataloaders(
            args.data_dir, 
            batch_size=args.batch_size,
            max_atoms=args.max_atoms
        )
        
        logger.info(f"训练集批次数: {len(train_loader)}")
        logger.info(f"验证集批次数: {len(valid_loader)}")
        logger.info(f"测试集批次数: {len(test_loader)}")
        
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        return
    
    # 创建模型
    logger.info("创建ToxD4C模型...")
    model = ToxD4C(config, device=device).to(device)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型创建成功，参数数量: {total_params:,}")
    
    # 测试模型前向传播
    logger.info("测试模型前向传播...")
    model.eval()
    with torch.no_grad():
        try:
            sample_batch = next(iter(train_loader))
            data = {
                'atom_features': sample_batch['atom_features'].to(device),
                'edge_index': sample_batch['edge_index'].to(device),
                'coordinates': sample_batch['coordinates'].to(device),
                'batch': sample_batch['batch'].to(device)
            }
            smiles_list = sample_batch['smiles']
            
            test_outputs = model(data, smiles_list)
            
            logger.info("测试输出形状:")
            for key, value in test_outputs['predictions'].items():
                logger.info(f"  {key}: {value.shape}")
                has_nan = torch.isnan(value).any().item()
                has_inf = torch.isinf(value).any().item()
                logger.info(f"  {key} 是否包含NaN: {has_nan}")
                logger.info(f"  {key} 是否包含Inf: {has_inf}")
            
            if any(torch.isnan(v).any() for v in test_outputs['predictions'].values()):
                logger.error("模型输出包含NaN，请检查模型实现")
                return
            
            logger.info("模型前向传播测试通过！")
            
        except Exception as e:
            logger.error(f"模型前向传播测试失败: {e}")
            logger.error(traceback.format_exc())
            return
    
    # 创建优化器和调度器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5,
        eps=1e-8
    )
    
    # 学习率调度器 (带预热)
    num_training_steps = len(train_loader) * args.num_epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # 训练循环
    logger.info("开始训练...")
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # 训练
        train_loss, train_cls_loss, train_reg_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )
        
        # 验证
        val_metrics, val_loss, val_cls_loss, val_reg_loss = evaluate_model(
            model, valid_loader, device
        )
        
        # 详细的训练监控信息
        logger.info(f"=== Epoch {epoch + 1} 训练结果 ===")
        logger.info(f"训练损失: {train_loss:.4f} (分类: {train_cls_loss:.4f}, 回归: {train_reg_loss:.4f})")
        logger.info(f"验证损失: {val_loss:.4f} (分类: {val_cls_loss:.4f}, 回归: {val_reg_loss:.4f})")
        
        # 计算并显示关键评估指标
        if val_metrics:
            # 分类指标统计
            cls_accs = [v for k, v in val_metrics.items() if 'accuracy' in k]
            cls_aucs = [v for k, v in val_metrics.items() if 'auc' in k]
            
            # 回归指标统计
            r2_scores = [v for k, v in val_metrics.items() if 'r2' in k]
            rmse_scores = [v for k, v in val_metrics.items() if 'rmse' in k]
            
            # 显示分类性能
            if cls_accs:
                avg_acc = np.mean(cls_accs)
                logger.info(f"平均分类准确率: {avg_acc:.4f} (有效任务: {len(cls_accs)}/26)")
            
            if cls_aucs:
                avg_auc = np.mean(cls_aucs)
                logger.info(f"平均AUC: {avg_auc:.4f}")
            
            # 显示回归性能
            if r2_scores:
                avg_r2 = np.mean(r2_scores)
                logger.info(f"平均R²: {avg_r2:.4f} (有效任务: {len(r2_scores)}/5)")
            
            if rmse_scores:
                avg_rmse = np.mean(rmse_scores)
                logger.info(f"平均RMSE: {avg_rmse:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # 保存模型
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config,
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }
            
            checkpoint_path = output_dir / f"{args.experiment_name}_best.pth"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"保存最佳模型: {checkpoint_path}")
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= patience:
            logger.info(f"验证损失连续{patience}个epoch未改善，早停")
            break
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f"{args.experiment_name}_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config,
                'val_loss': val_loss,
                'val_metrics': val_metrics
            }, checkpoint_path)
            logger.info(f"保存检查点: {checkpoint_path}")
    
    logger.info("训练完成！")
    
    # 最终评估
    logger.info("进行最终评估...")
    
    # 加载最佳模型
    best_checkpoint_path = output_dir / f"{args.experiment_name}_best.pth"
    if best_checkpoint_path.exists():
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"加载最佳模型: {best_checkpoint_path}")
    
    # 最终验证
    final_metrics, final_loss, final_cls_loss, final_reg_loss = evaluate_model(
        model, valid_loader, device
    )
    
    logger.info(f"最终验证结果:")
    logger.info(f"  总损失: {final_loss:.4f}")
    logger.info(f"  分类损失: {final_cls_loss:.4f}")
    logger.info(f"  回归损失: {final_reg_loss:.4f}")
    
    if final_metrics:
        # 分类指标汇总
        cls_accs = [v for k, v in final_metrics.items() if 'accuracy' in k]
        cls_aucs = [v for k, v in final_metrics.items() if 'auc' in k]
        
        if cls_accs:
            logger.info(f"  平均分类准确率: {np.mean(cls_accs):.4f}")
        if cls_aucs:
            logger.info(f"  平均AUC: {np.mean(cls_aucs):.4f}")
        
        # 回归指标汇总
        r2_scores = [v for k, v in final_metrics.items() if 'r2' in k]
        rmse_scores = [v for k, v in final_metrics.items() if 'rmse' in k]
        
        if r2_scores:
            logger.info(f"  平均R²: {np.mean(r2_scores):.4f}")
        if rmse_scores:
            logger.info(f"  平均RMSE: {np.mean(rmse_scores):.4f}")
    
    # 保存最终结果
    results = {
        'experiment_name': args.experiment_name,
        'config': config,
        'final_metrics': final_metrics,
        'final_loss': final_loss,
        'final_cls_loss': final_cls_loss,
        'final_reg_loss': final_reg_loss,
        'model_params': total_params
    }
    
    results_path = output_dir / f"{args.experiment_name}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"结果已保存: {results_path}")


if __name__ == "__main__":
    main()