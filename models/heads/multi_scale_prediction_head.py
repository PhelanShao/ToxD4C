"""
多尺度预测头 (Multi-Scale Prediction Head)
为多个毒性任务生成预测，并可选择性地输出不确定性

作者: AI助手
日期: 2024-06-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Tuple, Optional

class MultiScalePredictionHead(nn.Module):
    """
    多尺度预测头，为每个任务动态创建独立的预测分支。
    支持不确定性加权损失，通过为每个任务预测一个不确定性值（log variance）来实现。
    """
    
    def __init__(self,
                 input_dim: int,
                 task_configs: Dict[str, Dict[str, Any]],
                 dropout: float = 0.1,
                 uncertainty_weighting: bool = False,
                 classification_tasks_list: List[str] = None,
                 regression_tasks_list: List[str] = None):
        """
        初始化多尺度预测头

        Args:
            input_dim (int): 输入特征维度
            task_configs (Dict[str, Dict[str, Any]]): 任务配置字典
            dropout (float): Dropout比率
            uncertainty_weighting (bool): 是否启用不确定性加权
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.task_configs = task_configs
        self.uncertainty_weighting = uncertainty_weighting
        self.classification_tasks_list = classification_tasks_list or []
        self.regression_tasks_list = regression_tasks_list or []
        
        self.task_heads = nn.ModuleDict()
        
        if self.uncertainty_weighting:
            self.uncertainty_heads = nn.ModuleDict()

        for task_name, config in task_configs.items():
            output_dim = config.get('output_dim', 1)
            
            # 创建预测头
            self.task_heads[task_name] = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim // 2, output_dim)
            )
            
            # 如果启用，为每个任务创建不确定性预测头
            if self.uncertainty_weighting:
                self.uncertainty_heads[task_name] = nn.Sequential(
                    nn.Linear(input_dim, input_dim // 2),
                    nn.ReLU(),
                    nn.Linear(input_dim // 2, output_dim)
                )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x (torch.Tensor): 输入的图/分子表示 (batch_size, input_dim)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - cls_preds: 分类任务的预测 (batch_size, num_cls_tasks)
                - reg_preds: 回归任务的预测 (batch_size, num_reg_tasks)
        """
        cls_preds_list = []
        reg_preds_list = []

        # 保证输出顺序与列表一致
        for task_name in self.classification_tasks_list:
            if task_name in self.task_heads:
                cls_preds_list.append(self.task_heads[task_name](x))

        for task_name in self.regression_tasks_list:
            if task_name in self.task_heads:
                reg_preds_list.append(self.task_heads[task_name](x))

        # 合并成一个张量
        cls_preds = torch.cat(cls_preds_list, dim=1) if cls_preds_list else torch.empty(x.size(0), 0, device=x.device)
        reg_preds = torch.cat(reg_preds_list, dim=1) if reg_preds_list else torch.empty(x.size(0), 0, device=x.device)

        return cls_preds, reg_preds


if __name__ == '__main__':
    # --- 测试多尺度预测头 ---
    
    # 模拟输入
    batch_size = 4
    input_dim = 512
    
    task_configs = {
        'Carcinogenicity': {'output_dim': 1, 'task_type': 'classification'},
        'Ames Mutagenicity': {'output_dim': 1, 'task_type': 'classification'},
        'Acute oral toxicity (LD50)': {'output_dim': 1, 'task_type': 'regression'}
    }
    
    features = torch.randn(batch_size, input_dim)
    
    # --- 1. 不带不确定性测试 ---
    print("--- 测试不带不确定性 ---")
    prediction_head = MultiScalePredictionHead(
        input_dim=input_dim,
        task_configs=task_configs,
        uncertainty_weighting=False
    )
    
    preds, uncerts = prediction_head(features)
    
    print("预测任务:")
    for task, pred in preds.items():
        print(f"  {task}: {pred.shape}")
        assert pred.shape == (batch_size,)
        
    print(f"不确定性: {uncerts}")
    assert uncerts is None
    print("✓ 测试通过!")
    
    # --- 2. 带不确定性测试 ---
    print("\n--- 测试带不确定性 ---")
    prediction_head_uncert = MultiScalePredictionHead(
        input_dim=input_dim,
        task_configs=task_configs,
        uncertainty_weighting=True
    )
    
    preds, uncerts = prediction_head_uncert(features)
    
    print("预测任务:")
    for task, pred in preds.items():
        print(f"  {task}: {pred.shape}")
        assert pred.shape == (batch_size,)
        
    print("不确定性:")
    for task, uncert in uncerts.items():
        print(f"  {task}: {uncert.shape}")
        assert uncert.shape == (batch_size,)
        assert torch.all(uncert > 0) # 确保不确定性为正
        
    print("✓ 测试通过!") 