import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from torch_geometric.nn import GCN, global_mean_pool
from torch_geometric.utils import to_dense_batch

from .architectures.gnn_transformer_hybrid import GNNTransformerHybrid
from .encoders.geometric_encoder import GeometricEncoder
from .encoders.hierarchical_encoder import HierarchicalEncoder
from .fingerprints.molecular_fingerprint_enhanced import MolecularFingerprintModule
from .heads.multi_scale_prediction_head import MultiScalePredictionHead
from configs.toxd4c_config import CLASSIFICATION_TASKS, REGRESSION_TASKS
from .losses.contrastive_loss import SupConLoss


class ToxD4C(nn.Module):
    def __init__(self, config: Dict[str, Any], device: str = 'cpu'):
        super().__init__()
        
        self.config = config
        self.device = device
        
        self.atom_embedding = nn.Linear(
            config['atom_features_dim'], config['hidden_dim']
        )
        
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
            fp_dim = config.get('fingerprint_dim', 512)
        else:
            fp_dim = 0

        fusion_input_dim = config['hidden_dim']
        if config.get('use_hierarchical_encoder', False):
            fusion_input_dim += config['hidden_dim']
        if config.get('use_fingerprints', False):
            fusion_input_dim += fp_dim
            
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
        
        atom_repr = self.atom_embedding(x)
        
        if hasattr(self, 'geometric_encoder') and pos is not None:
            atom_repr = self.geometric_encoder(atom_repr, pos, edge_index)
        
        if isinstance(self.main_encoder, GCN):
            node_repr = self.main_encoder(atom_repr, edge_index)
            graph_repr_main = global_mean_pool(node_repr, batch)
        else:
            graph_repr_main, _ = self.main_encoder(atom_repr, edge_index, batch)
        
        all_graph_repr = [graph_repr_main]
        
        if hasattr(self, 'hierarchical_encoder'):
            graph_repr_hierarchical = self.hierarchical_encoder(atom_repr, edge_index, batch)
            all_graph_repr.append(graph_repr_hierarchical)
            
        if hasattr(self, 'fingerprint_module'):
            fp_repr = self.fingerprint_module(smiles=smiles_list)
            all_graph_repr.append(fp_repr)
            
        if len(all_graph_repr) > 1:
            fused_repr = torch.cat(all_graph_repr, dim=1)
            final_graph_repr = self.fusion_layer(fused_repr)
        else:
            final_graph_repr = graph_repr_main
            
        cls_preds, reg_preds = self.prediction_head(final_graph_repr)
        
        output = {
            'predictions': {
                'classification': cls_preds,
                'regression': reg_preds
            },
            'graph_representation': final_graph_repr,
            'uncertainties': None
        }
        
            
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