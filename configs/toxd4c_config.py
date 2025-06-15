"""
ToxD4C数据集任务配置
包含26个分类任务和5个回归任务的完整配置

作者: AI助手
日期: 2024-06-11
"""

from typing import Dict, List, Any


# 分类任务列表 (26个)
CLASSIFICATION_TASKS = [
    'Carcinogenicity', 'Ames Mutagenicity', 'Respiratory toxicity', 
    'Eye irritation', 'Eye corrosion', 'Cardiotoxicity1', 'Cardiotoxicity10', 
    'Cardiotoxicity30', 'Cardiotoxicity5', 'CYP1A2', 'CYP2C19', 'CYP2C9', 
    'CYP2D6', 'CYP3A4', 'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 
    'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 
    'SR-HSE', 'SR-MMP', 'SR-p53'
]

# 回归任务列表 (5个)
REGRESSION_TASKS = [
    'Acute oral toxicity (LD50)', 'LC50DM', 'BCF', 'LC50', 'IGC50'
]

# 任务配置字典
def get_toxd4c_task_configs() -> Dict[str, Dict[str, Any]]:
    """获取ToxD4C数据集的任务配置"""
    task_configs = {}
    
    # 分类任务配置 (二分类，输出维度为1)
    for task in CLASSIFICATION_TASKS:
        task_configs[task] = {
            'output_dim': 1,
            'task_type': 'classification',
            'activation': 'sigmoid',
            'loss_function': 'binary_cross_entropy'
        }
    
    # 回归任务配置
    for task in REGRESSION_TASKS:
        task_configs[task] = {
            'output_dim': 1,
            'task_type': 'regression',
            'activation': 'linear',
            'loss_function': 'mse'
        }
    
    return task_configs


# 任务分组
TASK_GROUPS = {
    'cardiotoxicity': ['Cardiotoxicity1', 'Cardiotoxicity10', 'Cardiotoxicity30', 'Cardiotoxicity5'],
    'cyp_enzymes': ['CYP1A2', 'CYP2C19', 'CYP2C9', 'CYP2D6', 'CYP3A4'],
    'nuclear_receptors': ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma'],
    'stress_response': ['SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'],
    'toxicity_endpoints': ['Carcinogenicity', 'Ames Mutagenicity', 'Respiratory toxicity', 'Eye irritation', 'Eye corrosion'],
    'environmental_fate': ['LC50DM', 'BCF', 'LC50', 'IGC50'],
    'acute_toxicity': ['Acute oral toxicity (LD50)']
}

# 任务权重 (基于生物学重要性)
TASK_WEIGHTS = {
    # 高优先级毒性终点
    'Carcinogenicity': 2.0,
    'Ames Mutagenicity': 2.0,
    'Acute oral toxicity (LD50)': 2.0,
    
    # 心脏毒性相关
    'Cardiotoxicity1': 1.5,
    'Cardiotoxicity5': 1.5,
    'Cardiotoxicity10': 1.3,
    'Cardiotoxicity30': 1.3,
    
    # CYP酶抑制
    'CYP3A4': 1.5,  # 最重要的CYP酶
    'CYP2D6': 1.3,
    'CYP1A2': 1.2,
    'CYP2C9': 1.2,
    'CYP2C19': 1.2,
    
    # 环境毒性
    'LC50': 1.3,
    'IGC50': 1.3,
    'BCF': 1.2,
    'LC50DM': 1.2,
    
    # 其他终点默认权重为1.0
}

# 为没有指定权重的任务设置默认权重
def get_task_weights() -> Dict[str, float]:
    """获取完整的任务权重字典"""
    weights = TASK_WEIGHTS.copy()
    all_tasks = CLASSIFICATION_TASKS + REGRESSION_TASKS
    
    for task in all_tasks:
        if task not in weights:
            weights[task] = 1.0
    
    return weights


# 模型配置
def get_enhanced_toxd4c_config() -> Dict[str, Any]:
    """获取ToxD4C的完整配置"""
    return {
        # 数据配置
        'atom_features_dim': 119,
        'bond_features_dim': 12,
        'edge_features_dim': 12,
        
        # 模型架构配置
        'hidden_dim': 512,
        'num_encoder_layers': 6,
        'num_attention_heads': 8,
        'dropout': 0.1,
        
        # 几何编码器配置
        'use_geometric_encoder': True,
        'geometric_hidden_dim': 256,
        'max_distance': 10.0,
        'num_rbf': 50,
        
        # 分层编码器配置
        'use_hierarchical_encoder': True,
        'hierarchy_levels': [2, 4, 8],
        
        # GNN-Transformer混合架构配置
        'use_hybrid_architecture': True,
        'gnn_layers': 3,
        'transformer_layers': 3,
        'fusion_method': 'cross_attention',
        
        # 分子指纹配置
        'use_fingerprints': True,
        'fingerprint_dim': 512,
        'fingerprint_configs': {
            'ecfp': {'n_bits': 2048, 'radius': 2},
            'maccs': {'n_bits': 167},
            'rdkit_fp': {'n_bits': 2048},
            'descriptors': {'n_features': 15}
        },
        
        # 任务配置
        'task_configs': get_toxd4c_task_configs(),
        'task_weights': get_task_weights(),
        'task_groups': TASK_GROUPS,
        
        # 对比学习配置
        'use_contrastive_learning': True,
        'contrastive_temperature': 0.1,
        'contrastive_weight': 0.3,
        
        # 训练配置
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'batch_size': 64,
        'max_epochs': 100,
        'patience': 10,
        'warmup_steps': 1000,
        
        # 数据增强配置
        'use_data_augmentation': True,
        'augmentation_ratio': 0.1,
        
        # 评估配置
        'eval_metrics': ['accuracy', 'f1_score', 'auc', 'precision', 'recall'],
        'regression_metrics': ['rmse', 'mae', 'r2'],
        
        # 其他配置
        'uncertainty_weighting': True,
        'gradient_clipping': 1.0,
        'label_smoothing': 0.1,
    }


# 数据集分割配置
DATASET_SPLITS = {
    'train_ratio': 0.8,
    'val_ratio': 0.1,
    'test_ratio': 0.1,
    'random_seed': 42,
    'stratify': True  # 对分类任务进行分层采样
}

# 数据路径配置
DATA_PATHS = {
    'raw_data_dir': 'data/raw/',
    'processed_data_dir': 'data/processed/',
    'model_save_dir': 'checkpoints/',
    'results_dir': 'results/',
    'logs_dir': 'logs/'
}

if __name__ == "__main__":
    # 打印配置信息
    config = get_enhanced_toxd4c_config()
    print("ToxD4C 配置:")
    print(f"分类任务数量: {len(CLASSIFICATION_TASKS)}")
    print(f"回归任务数量: {len(REGRESSION_TASKS)}")
    print(f"总任务数量: {len(CLASSIFICATION_TASKS) + len(REGRESSION_TASKS)}")
    print(f"模型隐藏维度: {config['hidden_dim']}")
    print(f"注意力头数: {config['num_attention_heads']}")
    print(f"编码器层数: {config['num_encoder_layers']}")
    
    print("\n任务分组:")
    for group_name, tasks in TASK_GROUPS.items():
        print(f"{group_name}: {len(tasks)} 任务")
    
    print("\n高权重任务:")
    weights = get_task_weights()
    high_weight_tasks = {k: v for k, v in weights.items() if v > 1.0}
    for task, weight in sorted(high_weight_tasks.items(), key=lambda x: x[1], reverse=True):
        print(f"{task}: {weight}") 