import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from torch_geometric.nn import GCN, global_mean_pool
from torch_geometric.utils import to_dense_batch
import numpy as np

from rdkit import Chem
from .architectures.gnn_transformer_hybrid import GNNTransformerHybrid
from .architectures.gcn_stack import GCNStack
from .encoders.geometric_encoder import GeometricEncoder
from .encoders.hierarchical_encoder import HierarchicalEncoder
from .encoders.quantum_descriptor_module import QuantumDescriptorModule
from .fingerprints.molecular_fingerprint_enhanced import MolecularFingerprintModule
from .heads.multi_scale_prediction_head import MultiScalePredictionHead
from configs.toxd4c_config import CLASSIFICATION_TASKS, REGRESSION_TASKS
from .losses.contrastive_loss import SupConLoss


class ToxD4C(nn.Module):
    def __init__(self, config: Dict[str, Any], device: str = 'cpu'):
        super().__init__()
        
        self.config = config
        self.device = device
        
        # Store task lists in config for easy access in train_epoch
        self.config['classification_tasks_list'] = CLASSIFICATION_TASKS
        self.config['regression_tasks_list'] = REGRESSION_TASKS
        
        self.atom_embedding = nn.Linear(
            config['atom_features_dim'], config['hidden_dim']
        )
        
        # Ablation toggles
        use_gnn = config.get('use_gnn', True)
        use_transformer = config.get('use_transformer', True)
        gnn_backbone = config.get('gnn_backbone', 'graph_attention')

        if config.get('use_hybrid_architecture', False):
            self.main_encoder = GNNTransformerHybrid(
                input_dim=config['hidden_dim'],
                hidden_dim=config['hidden_dim'],
                output_dim=config['hidden_dim'],
                gnn_layers=config.get('gnn_layers', 3),
                transformer_layers=config.get('transformer_layers', 3),
                num_heads=config['num_attention_heads'],
                dropout=config['dropout'],
                use_gnn=use_gnn,
                use_transformer=use_transformer,
                gnn_backbone=gnn_backbone,
                gcn_stack_layers=config.get('gcn_stack_layers', 3)
            )
        elif use_gnn and not use_transformer:
            # GNN only mode
            if gnn_backbone == 'pyg_gcn_stack':
                self.main_encoder = GCNStack(
                    in_channels=config['hidden_dim'],
                    hidden_dim=config['hidden_dim'],
                    num_layers=config.get('gcn_stack_layers', 3),
                    dropout=config['dropout'],
                    use_residual=True,
                )
            else:
                self.main_encoder = GCN(
                    in_channels=config['hidden_dim'],
                    hidden_channels=config['hidden_dim'],
                    num_layers=config['num_encoder_layers'],
                    dropout=config['dropout']
                )
        else:
            self.main_encoder = GCN(
                in_channels=config['hidden_dim'],
                hidden_channels=config['hidden_dim'],
                num_layers=config['num_encoder_layers'],
                dropout=config['dropout']
            )

        if config.get('use_geometric_encoder', False):
            self.geometric_encoder = GeometricEncoder(
                embed_dim=config['hidden_dim'],
                hidden_dim=config.get('geometric_hidden_dim', 256),
                num_rbf=config.get('num_rbf', 50),
                max_distance=config.get('max_distance', 10.0)
            )

        if config.get('use_hierarchical_encoder', False):
            self.hierarchical_encoder = HierarchicalEncoder(
                input_dim=config['hidden_dim'],
                hidden_dim=config['hidden_dim'],
                hierarchy_levels=config.get('hierarchy_levels', [2, 4, 8]),
                dropout=config['dropout']
            )
            
        if config.get('use_fingerprints', False):
            self.fingerprint_module = MolecularFingerprintModule(
                output_dim=config.get('fingerprint_dim', 512),
                fingerprint_configs=config.get('fingerprint_configs', {})
            )

        # Quantum Descriptor Module (可选)
        if config.get('use_quantum_descriptors', False):
            self.quantum_descriptor_module = QuantumDescriptorModule(
                num_descriptors=config.get('num_quantum_descriptors', 69),
                hidden_dim=config['hidden_dim'],
                output_dim=config.get('quantum_descriptor_dim', 256),
                graph_repr_dim=config['hidden_dim'],  # Match graph representation dimension
                num_decay_layers=config.get('quantum_decay_layers', 4),
                decay_rate=config.get('quantum_decay_rate', 0.1),
                dropout=config['dropout'],
                use_gating=config.get('quantum_use_gating', True),
            )

        # Define the fusion layer with the maximum possible input dimension
        fusion_input_dim = config['hidden_dim']
        if config.get('use_hierarchical_encoder', False):
            fusion_input_dim += config['hidden_dim']
        if config.get('use_fingerprints', False):
            fusion_input_dim += config.get('fingerprint_dim', 512)
        if config.get('use_quantum_descriptors', False):
            fusion_input_dim += config.get('quantum_descriptor_dim', 256)

        self.fusion_layer = nn.Linear(fusion_input_dim, config['hidden_dim'])
        
        self.prediction_head = MultiScalePredictionHead(
            input_dim=config['hidden_dim'],
            task_configs=config['task_configs'],
            dropout=config['dropout'],
            uncertainty_weighting=config.get('uncertainty_weighting', False),
            classification_tasks_list=CLASSIFICATION_TASKS,
            regression_tasks_list=REGRESSION_TASKS
        )
        
        if config.get('use_contrastive_learning', False):
            self.contrastive_loss = SupConLoss(
                temperature=config.get('contrastive_temperature', 0.1)
            )

    def forward(self,
                data: Dict[str, torch.Tensor],
                smiles_list: List[str]) -> Dict[str, Any]:
        x = data['atom_features']
        edge_index = data['edge_index']
        batch = data.get('batch')
        pos = data.get('coordinates')
        
        interp_data = {}

        atom_repr = self.atom_embedding(x)
        
        if hasattr(self, 'geometric_encoder') and pos is not None:
            atom_repr = self.geometric_encoder(atom_repr, pos, edge_index)
        
        # Handle different encoder types with different return signatures
        if isinstance(self.main_encoder, (GCN, GCNStack)):
            # GCN and GCNStack only take (x, edge_index) and return node features
            node_repr = self.main_encoder(atom_repr, edge_index)
            graph_repr_main = global_mean_pool(node_repr, batch)
            interp_data['main_encoder'] = None
        elif isinstance(self.main_encoder, GNNTransformerHybrid):
            # GNNTransformerHybrid returns (graph_repr, interpretability_data)
            graph_repr_main, main_interp_data = self.main_encoder(atom_repr, edge_index, batch)
            interp_data['main_encoder'] = main_interp_data
        else:
            # Fallback: try to call with batch and unpack tuple
            try:
                graph_repr_main, main_interp_data = self.main_encoder(atom_repr, edge_index, batch)
                interp_data['main_encoder'] = main_interp_data
            except (TypeError, ValueError):
                # If that fails, assume it's a simple encoder
                node_repr = self.main_encoder(atom_repr, edge_index)
                graph_repr_main = global_mean_pool(node_repr, batch)
                interp_data['main_encoder'] = None

        all_graph_repr = [graph_repr_main]
        
        if hasattr(self, 'hierarchical_encoder'):
            graph_repr_hierarchical = self.hierarchical_encoder(atom_repr, edge_index, batch)
            all_graph_repr.append(graph_repr_hierarchical)
            
        # Check if the input is likely to be SMILES before calling the fingerprint module
        is_smiles_input = False
        if smiles_list and Chem.MolFromSmiles(smiles_list[0]) is not None:
            is_smiles_input = True

        if hasattr(self, 'fingerprint_module') and is_smiles_input:
            fp_result = self.fingerprint_module(smiles=smiles_list)
            # Handle both single return value and tuple return value
            if isinstance(fp_result, tuple):
                fp_repr, fp_attn_weights = fp_result
                interp_data['fingerprint_attention'] = fp_attn_weights
            else:
                fp_repr = fp_result
                interp_data['fingerprint_attention'] = None
            all_graph_repr.append(fp_repr)

        # Quantum Descriptor Module (如果启用且提供了描述符)
        quantum_sparsity_loss = torch.tensor(0.0, device=x.device)
        if hasattr(self, 'quantum_descriptor_module'):
            quantum_descriptors = data.get('quantum_descriptors')
            if quantum_descriptors is not None:
                qd_result = self.quantum_descriptor_module(
                    quantum_descriptors,
                    graph_repr=graph_repr_main
                )
                all_graph_repr.append(qd_result['descriptor_repr'])
                interp_data['quantum_gate_values'] = qd_result['gate_values']
                quantum_sparsity_loss = qd_result['sparsity_loss']

        if len(all_graph_repr) > 1:
            fused_repr = torch.cat(all_graph_repr, dim=1)
            
            # If the fusion layer was initialized with a larger dimension (i.e., with fingerprints),
            # and we are now running without them, we need to slice the layer's weight.
            expected_dim = self.fusion_layer.in_features
            if fused_repr.shape[1] < expected_dim:
                # This case happens when running XYZ input on a model trained with fingerprints
                sliced_weight = self.fusion_layer.weight[:, :fused_repr.shape[1]]
                final_graph_repr = F.linear(fused_repr, sliced_weight, self.fusion_layer.bias)
            else:
                final_graph_repr = self.fusion_layer(fused_repr)
        else:
            final_graph_repr = graph_repr_main
            
        predictions, uncertainties = self.prediction_head(final_graph_repr)
        
        output = {
            'predictions': predictions,
            'graph_representation': final_graph_repr,
            'interpretation': interp_data,
            'uncertainties': uncertainties
        }

        # 添加量子描述符稀疏性损失（用于训练）
        if hasattr(self, 'quantum_descriptor_module'):
            output['quantum_sparsity_loss'] = quantum_sparsity_loss

        if hasattr(self, 'contrastive_loss'):
            output['contrastive_features'] = graph_repr_main

        return output

    def compute_contrastive_loss(self,
                                 features: torch.Tensor,
                                 labels: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'contrastive_loss'):
            if features.ndim == 2:
                features = features.unsqueeze(1)
            return self.contrastive_loss(features, labels)

        return torch.tensor(0.0, device=self.device)

    # ==================== 注意力可视化方法 ====================

    def enable_attention_storage(self, enable: bool = True):
        """
        启用/禁用注意力权重存储用于可视化。

        Args:
            enable: True 启用存储，False 禁用
        """
        for name, module in self.named_modules():
            if hasattr(module, 'store_attention'):
                module.store_attention = enable
                if not enable:
                    module.attention_weights_cache = None

    def get_all_attention_weights(self) -> Dict[str, torch.Tensor]:
        """
        获取所有注意力层的缓存权重。

        Returns:
            Dict: {层名称: 注意力权重张量 [batch, heads, atoms, atoms]}
        """
        attention_dict = {}
        for name, module in self.named_modules():
            if hasattr(module, 'attention_weights_cache') and module.attention_weights_cache is not None:
                attention_dict[name] = module.attention_weights_cache
        return attention_dict

    def clear_attention_cache(self):
        """清除所有注意力层的缓存。"""
        for name, module in self.named_modules():
            if hasattr(module, 'clear_attention_cache'):
                module.clear_attention_cache()
            elif hasattr(module, 'attention_weights_cache'):
                module.attention_weights_cache = None

    def visualize_attention(
        self,
        data: Dict[str, torch.Tensor],
        smiles: str,
        layer_name: Optional[str] = None,
        heads_to_show: List[int] = [0, 3, 7],
        save_path: Optional[str] = None,
        sample_idx: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        对单个分子进行注意力可视化。

        Args:
            data: 模型输入数据
            smiles: 分子的 SMILES 字符串
            layer_name: 要可视化的注意力层名称（None 则使用第一个）
            heads_to_show: 要显示的注意力头索引列表
            save_path: 图片保存路径
            sample_idx: batch 中的样本索引

        Returns:
            Dict: 包含注意力权重和距离矩阵的字典
        """
        # 启用注意力存储
        self.enable_attention_storage(True)

        # 前向传播
        self.eval()
        with torch.no_grad():
            _ = self.forward(data, [smiles])

        # 获取注意力权重
        attention_dict = self.get_all_attention_weights()

        if not attention_dict:
            print("警告: 未找到缓存的注意力权重")
            return None

        # 选择要可视化的层
        if layer_name and layer_name in attention_dict:
            attention = attention_dict[layer_name]
        else:
            # 使用第一个可用的注意力层
            layer_name = list(attention_dict.keys())[0]
            attention = attention_dict[layer_name]

        # 提取单个样本的注意力
        if attention.dim() == 4:
            attention = attention[sample_idx]  # [heads, atoms, atoms]

        # 获取 3D 坐标
        coordinates = data.get('coordinates')
        if coordinates is not None:
            coordinates = coordinates.cpu().numpy()
            # 如果是 batch 数据，需要提取对应样本
            if coordinates.ndim == 2:
                batch = data.get('batch')
                if batch is not None:
                    mask = (batch == sample_idx).cpu().numpy()
                    coordinates = coordinates[mask]

        # 清除缓存
        self.clear_attention_cache()
        self.enable_attention_storage(False)

        result = {
            'layer_name': layer_name,
            'attention_weights': attention.cpu().numpy(),
            'coordinates': coordinates,
            'smiles': smiles,
            'heads_to_show': heads_to_show
        }

        # 如果提供了保存路径，生成可视化图
        if save_path:
            try:
                from attention_visualization import plot_attention_visualization
                plot_attention_visualization(
                    smiles=smiles,
                    coordinates=coordinates,
                    attention_weights=attention,
                    heads_to_show=heads_to_show,
                    save_path=save_path
                )
            except ImportError:
                print("请确保 attention_visualization.py 在当前路径")

        return result


if __name__ == '__main__':
    from configs.toxd4c_config import get_enhanced_toxd4c_config
    
    config = get_enhanced_toxd4c_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = ToxD4C(config, device=device).to(device)
    
    print("ToxD4C Model Test:")
    print(f"Device: {device}")
    print(f"Total number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
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
    
    output = model(data, smiles_list)
    
    print("\nForward pass output:")
    print(f"Graph representation shape: {output['graph_representation'].shape}")
    
    # Check the number of prediction tasks based on what's available
    num_cls_tasks = output['predictions']['classification'].shape[1] if 'classification' in output['predictions'] and output['predictions']['classification'].numel() > 0 else 0
    num_reg_tasks = output['predictions']['regression'].shape[1] if 'regression' in output['predictions'] and output['predictions']['regression'].numel() > 0 else 0
    print(f"Number of prediction tasks: {num_cls_tasks + num_reg_tasks}")

    # Check output for a specific task if available
    if num_cls_tasks > 0:
        task_name = CLASSIFICATION_TASKS[0]
        print(f"Prediction shape for task '{task_name}': {output['predictions']['classification'][:, 0].shape}")
        assert output['predictions']['classification'].shape[0] == batch_size
    elif num_reg_tasks > 0:
        task_name = REGRESSION_TASKS[0]
        print(f"Prediction shape for task '{task_name}': {output['predictions']['regression'][:, 0].shape}")
        assert output['predictions']['regression'].shape[0] == batch_size

    assert output['graph_representation'].shape == (batch_size, config['hidden_dim'])
    
    print("\n✓ Model test passed!")