"""
ToxD4C 主模型架构
集成了多种先进的分子表示学习模块

作者: AI助手
日期: 2024-06-11
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from torch_geometric.nn import GCN, global_mean_pool
from torch_geometric.utils import to_dense_batch

# --- 导入核心模块 ---
# 基础架构
from .architectures.gnn_transformer_hybrid import GNNTransformerHybrid

# 编码器
from .encoders.geometric_encoder import GeometricEncoder
from .encoders.hierarchical_encoder import HierarchicalEncoder

# 特征增强
from .fingerprints.molecular_fingerprint_enhanced import MolecularFingerprintModule

# 预测头
from .heads.multi_scale_prediction_head import MultiScalePredictionHead
from configs.toxd4c_config import CLASSIFICATION_TASKS, REGRESSION_TASKS

# 损失函数
from .losses.contrastive_loss import SupConLoss


class ToxD4C(nn.Module):
    """
    ToxD4C 统一模型
    
    该模型集成了以下先进功能：
    1. GNN-Transformer混合架构：捕捉局部和全局的分子结构信息。
    2. 几何编码器：利用3D空间坐标信息。
    3. 分层编码器：在多个尺度上提取图特征。
    4. 分子指纹增强：融合多种化学指纹和描述符。
    5. 多尺度预测头：为多个毒性任务生成预测。
    6. 对比学习：通过自监督学习提升表示质量。
    """
    
    def __init__(self, config: Dict[str, Any], device: str = 'cpu'):
        """
        初始化模型

        Args:
            config (Dict[str, Any]): 模型配置文件
            device (str): 运行设备 ('cpu' or 'cuda')
        """
        super().__init__()
        
        self.config = config
        self.device = device
        
        # --- 1. 输入层：将原始原子特征映射到隐藏维度 ---
        self.atom_embedding = nn.Linear(
            config['atom_features_dim'], config['hidden_dim']
        )
        
        # --- 2. 核心编码模块 ---
        
        # GNN-Transformer 混合编码器
        if config.get('use_hybrid_architecture', False):
            self.main_encoder = GNNTransformerHybrid(
                input_dim=config['hidden_dim'],
                hidden_dim=config['hidden_dim'],
                output_dim=config['hidden_dim'],
                gnn_layers=config.get('gnn_layers', 3),
                transformer_layers=config.get('transformer_layers', 3),
                num_heads=config['num_attention_heads'],
                dropout=config['dropout']
            )
        else:
            # 备用：使用简单的GCN编码器
            self.main_encoder = GCN(
                in_channels=config['hidden_dim'],
                hidden_channels=config['hidden_dim'],
                num_layers=config['num_encoder_layers'],
                dropout=config['dropout']
            )

        # 几何编码器 (可选)
        if config.get('use_geometric_encoder', False):
            self.geometric_encoder = GeometricEncoder(
                embed_dim=config['hidden_dim'],
                hidden_dim=config.get('geometric_hidden_dim', 256),
                num_rbf=config.get('num_rbf', 50),
                max_distance=config.get('max_distance', 10.0)
            )

        # 分层编码器 (可选)
        if config.get('use_hierarchical_encoder', False):
            self.hierarchical_encoder = HierarchicalEncoder(
                input_dim=config['hidden_dim'],
                hidden_dim=config['hidden_dim'],
                hierarchy_levels=config.get('hierarchy_levels', [2, 4, 8]),
                dropout=config['dropout']
            )
            
        # 分子指纹模块 (可选)
        if config.get('use_fingerprints', False):
            self.fingerprint_module = MolecularFingerprintModule(
                output_dim=config.get('fingerprint_dim', 512),
                fingerprint_configs=config.get('fingerprint_configs', {})
            )
            fp_dim = config.get('fingerprint_dim', 512)
        else:
            fp_dim = 0

        # --- 3. 特征融合 ---
        # 计算融合后的特征维度
        fusion_input_dim = config['hidden_dim']  # 主编码器输出
        if config.get('use_hierarchical_encoder', False):
            fusion_input_dim += config['hidden_dim']
        if config.get('use_fingerprints', False):
            fusion_input_dim += fp_dim
            
        self.fusion_layer = nn.Linear(fusion_input_dim, config['hidden_dim'])
        
        # --- 4. 预测头 ---
        self.prediction_head = MultiScalePredictionHead(
            input_dim=config['hidden_dim'],
            task_configs=config['task_configs'],
            dropout=config['dropout'],
            uncertainty_weighting=config.get('uncertainty_weighting', False),
            classification_tasks_list=CLASSIFICATION_TASKS,
            regression_tasks_list=REGRESSION_TASKS
        )
        
        # --- 5. 对比学习模块 (可选) ---
        if config.get('use_contrastive_learning', False):
            self.contrastive_loss = SupConLoss(
                temperature=config.get('contrastive_temperature', 0.1)
            )

    def forward(self, 
                data: Dict[str, torch.Tensor], 
                smiles_list: List[str]) -> Dict[str, Any]:
        """
        前向传播

        Args:
            data (Dict[str, torch.Tensor]): 包含图数据的字典
            smiles_list (List[str]): 分子SMILES列表

        Returns:
            Dict[str, Any]: 包含预测和中间表示的字典
        """
        # 从输入数据中解包
        x = data['atom_features']
        edge_index = data['edge_index']
        batch = data.get('batch')
        pos = data.get('coordinates')
        
        # 1. 初始原子嵌入
        atom_repr = self.atom_embedding(x)
        
        # 2. 几何编码 (如果启用)
        if hasattr(self, 'geometric_encoder') and pos is not None:
            atom_repr = self.geometric_encoder(atom_repr, pos, edge_index)
        
        # 3. 主编码器 (GNN-Transformer)
        if isinstance(self.main_encoder, GCN):
            # GCN返回节点级特征，需要手动池化
            node_repr = self.main_encoder(atom_repr, edge_index)
            graph_repr_main = global_mean_pool(node_repr, batch)
        else:
            # GNNTransformerHybrid 现在直接处理 torch_geometric 格式
            graph_repr_main, _ = self.main_encoder(atom_repr, edge_index, batch)
        
        # 4. 辅助编码器 (如果启用)
        all_graph_repr = [graph_repr_main]
        
        # 分层编码
        if hasattr(self, 'hierarchical_encoder'):
            graph_repr_hierarchical = self.hierarchical_encoder(atom_repr, edge_index, batch)
            all_graph_repr.append(graph_repr_hierarchical)
            
        # 分子指纹
        if hasattr(self, 'fingerprint_module'):
            fp_repr = self.fingerprint_module(smiles=smiles_list)
            all_graph_repr.append(fp_repr)
            
        # 5. 特征融合
        if len(all_graph_repr) > 1:
            fused_repr = torch.cat(all_graph_repr, dim=1)
            final_graph_repr = self.fusion_layer(fused_repr)
        else:
            final_graph_repr = graph_repr_main
            
        # 6. 预测头
        cls_preds, reg_preds = self.prediction_head(final_graph_repr)
        
        # 7. 构造输出
        output = {
            'predictions': {
                'classification': cls_preds,
                'regression': reg_preds
            },
            'graph_representation': final_graph_repr,
            'uncertainties': None # 暂时禁用不确定性
        }
        
            
        # 如果需要对比学习，添加相应的表示
        if hasattr(self, 'contrastive_loss'):
            # 假设主编码器的输出作为对比学习的表示
            output['contrastive_features'] = graph_repr_main
            
        return output

    def compute_contrastive_loss(self, 
                                 features: torch.Tensor, 
                                 labels: torch.Tensor) -> torch.Tensor:
        """
        计算对比损失
        
        Args:
            features (torch.Tensor): (batch_size, feature_dim)
            labels (torch.Tensor): (batch_size) - 用于区分正负样本
            
        Returns:
            torch.Tensor: 对比损失值
        """
        if hasattr(self, 'contrastive_loss'):
            # SupConLoss期望的输入是 (N, C, F)
            if features.ndim == 2:
                features = features.unsqueeze(1)
            return self.contrastive_loss(features, labels)
        
        return torch.tensor(0.0, device=self.device)


if __name__ == '__main__':
    # --- 测试 ToxD4C 模型 ---
    from configs.toxd4c_config import get_enhanced_toxd4c_config
    
    # 获取完整配置
    config = get_enhanced_toxd4c_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型实例
    model = ToxD4C(config, device=device).to(device)
    
    print("ToxD4C 模型测试:")
    print(f"设备: {device}")
    print(f"总参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 模拟输入数据
    batch_size = 4
    num_atoms_per_mol = 30
    
    data = {
        'atom_features': torch.randn(batch_size * num_atoms_per_mol, config['atom_features_dim']).to(device),
        'bond_features': torch.randn(batch_size * 50, config['bond_features_dim']).to(device),
        'edge_index': torch.randint(0, batch_size * num_atoms_per_mol, (2, 100)).to(device),
        'coordinates': torch.randn(batch_size * num_atoms_per_mol, 3).to(device),
        'batch': torch.repeat_interleave(torch.arange(batch_size), num_atoms_per_mol).to(device)
    }
    
    smiles_list = ["CCO"] * batch_size
    
    # 前向传播
    output = model(data, smiles_list)
    
    print("\n前向传播输出:")
    print(f"图表示形状: {output['graph_representation'].shape}")
    print(f"预测任务数量: {len(output['predictions'])}")
    
    # 检查一个任务的输出
    task_name = list(config['task_configs'].keys())[0]
    print(f"任务 '{task_name}' 的预测形状: {output['predictions'][task_name].shape}")
    
    # 验证输出形状
    assert output['graph_representation'].shape == (batch_size, config['hidden_dim'])
    assert output['predictions'][task_name].shape[0] == batch_size
    
    print("\n✓ 模型测试通过!") 