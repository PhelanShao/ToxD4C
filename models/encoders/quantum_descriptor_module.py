#!/usr/bin/env python
"""
Quantum Descriptor Module with Adaptive Feature Gating and Hierarchical Decay
==============================================================================

This module enables integration of pre-computed quantum chemical descriptors
(e.g., HOMO, LUMO, Mulliken charges, dipole moments) into the ToxD4C model.

Key Features
------------
1. **Adaptive Feature Gating**: Automatically learns feature importance through
   a learnable gating mechanism. Combines dynamic (input-dependent) and static
   (learned) gates to identify which descriptors are most relevant for the task.

2. **Hierarchical Decay**: Irrelevant features are progressively suppressed
   through multiple decay layers. Each layer applies a decay factor that
   increases with depth, ensuring that noise is filtered out while preserving
   important signals.

3. **Interpretability**: Gate values can be extracted after training to
   understand which quantum descriptors contribute most to predictions.

4. **Flexible Fusion**: Can operate standalone or fuse with graph representations
   through a learned fusion gate.

Architecture
------------
```
Input Descriptors [batch, num_descriptors]
        |
        v
  Input Projection (Linear + LayerNorm + GELU)
        |
        v
  Feature Gating Layer (learns importance weights)
        |
        v
  Hierarchical Decay Blocks (x N layers)
        |
        v
  Output Projection (Linear + LayerNorm)
        |
        v
  (Optional) Fusion with Graph Representation
```

Usage Example
-------------
```python
from models.encoders.quantum_descriptor_module import QuantumDescriptorModule

# Initialize module
module = QuantumDescriptorModule(
    num_descriptors=69,      # Number of input descriptors
    hidden_dim=256,          # Hidden dimension
    output_dim=256,          # Output dimension
    num_decay_layers=4,      # Number of decay layers
    decay_rate=0.1,          # Decay rate per layer
    use_gating=True,         # Enable feature gating
)

# Forward pass
descriptors = torch.randn(batch_size, 69)  # Pre-computed descriptors
graph_repr = torch.randn(batch_size, 256)  # Optional graph representation

result = module(descriptors, graph_repr)
# result['descriptor_repr']: Encoded descriptor representation
# result['fused_repr']: Fused representation (if graph_repr provided)
# result['gate_values']: Feature importance weights (for interpretability)
# result['sparsity_loss']: Regularization loss to add to training objective
```

Integration with ToxD4C
-----------------------
Enable in config:
```python
config = {
    'use_quantum_descriptors': True,
    'num_quantum_descriptors': 69,
    'quantum_descriptor_dim': 256,
    'quantum_decay_layers': 4,
    'quantum_decay_rate': 0.1,
    'quantum_use_gating': True,
    'quantum_sparsity_weight': 0.01,
}
```

In your data dict, include:
```python
data = {
    'atom_features': ...,
    'edge_index': ...,
    'quantum_descriptors': torch.tensor([...]),  # Shape: [batch, 69]
}
```

References
----------
- Feature gating inspired by attention mechanisms and mixture-of-experts
- Hierarchical decay inspired by residual networks with learnable gates
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class FeatureGatingLayer(nn.Module):
    """特征门控层 - 学习特征重要性并衰减无关特征"""
    
    def __init__(self, input_dim: int, temperature: float = 1.0, sparsity_reg: float = 0.01):
        super().__init__()
        self.temperature = temperature
        self.sparsity_reg = sparsity_reg
        
        # 门控网络：学习每个特征的重要性
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim),
        )
        # 可学习的基础门控值
        self.base_gate = nn.Parameter(torch.ones(input_dim) * 0.5)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, input_dim]
        Returns:
            gated_features, gate_values, sparsity_loss
        """
        gate_logits = self.gate_net(x)
        dynamic_gate = torch.sigmoid(gate_logits / self.temperature)
        gate = dynamic_gate * torch.sigmoid(self.base_gate)
        
        gated_x = x * gate
        sparsity_loss = self.sparsity_reg * torch.mean(gate)
        
        return gated_x, gate, sparsity_loss


class HierarchicalDecayBlock(nn.Module):
    """层级衰减块 - 每层逐步衰减无关信息"""
    
    def __init__(self, hidden_dim: int, decay_rate: float = 0.1):
        super().__init__()
        self.decay_rate = decay_rate
        
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.decay_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        decay_factor = max(1.0 - self.decay_rate * layer_idx, 0.3)
        h = self.transform(x)
        gate = self.decay_gate(x) * decay_factor
        return gate * h + self.residual_scale * x


class QuantumDescriptorModule(nn.Module):
    """
    量子描述符融合模块
    
    将预计算的量子化学描述符（如HOMO、LUMO、Mulliken电荷等）
    编码并与分子图表示融合。
    """
    
    def __init__(
        self,
        num_descriptors: int = 69,
        hidden_dim: int = 512,
        output_dim: int = 512,
        graph_repr_dim: int = 512,  # Dimension of graph representation for fusion
        num_decay_layers: int = 4,
        decay_rate: float = 0.1,
        dropout: float = 0.1,
        use_gating: bool = True,
    ):
        super().__init__()

        self.num_descriptors = num_descriptors
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.graph_repr_dim = graph_repr_dim
        self.use_gating = use_gating

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(num_descriptors, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 特征门控
        if use_gating:
            self.feature_gate = FeatureGatingLayer(
                hidden_dim, temperature=0.5, sparsity_reg=0.02
            )
        
        # 层级衰减块
        self.decay_blocks = nn.ModuleList([
            HierarchicalDecayBlock(hidden_dim, decay_rate)
            for _ in range(num_decay_layers)
        ])
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        # 图表示投影层：将图表示投影到与描述符相同的维度
        if graph_repr_dim != output_dim:
            self.graph_proj = nn.Linear(graph_repr_dim, output_dim)
        else:
            self.graph_proj = nn.Identity()

        # 融合门：控制描述符信息与主表示的融合比例
        # 输入维度为 output_dim * 2 (projected_graph + descriptor_repr)
        self.fusion_gate = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Sigmoid()
        )

    def forward(
        self,
        descriptors: torch.Tensor,
        graph_repr: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            descriptors: [batch, num_descriptors] 预计算的量子描述符
            graph_repr: [batch, output_dim] 可选的图表示，用于融合

        Returns:
            dict包含:
            - 'descriptor_repr': 编码后的描述符表示
            - 'fused_repr': 融合后的表示（如果提供了graph_repr）
            - 'gate_values': 特征门控值（用于可解释性）
            - 'sparsity_loss': 稀疏性损失
        """
        # 输入投影
        h = self.input_proj(descriptors)

        # 特征门控
        gate_values = None
        sparsity_loss = torch.tensor(0.0, device=descriptors.device)

        if self.use_gating:
            h, gate_values, sparsity_loss = self.feature_gate(h)

        # 层级衰减
        for i, block in enumerate(self.decay_blocks):
            h = block(h, i)

        # 输出投影
        descriptor_repr = self.output_proj(h)

        result = {
            'descriptor_repr': descriptor_repr,
            'gate_values': gate_values,
            'sparsity_loss': sparsity_loss,
        }

        # 如果提供了图表示，进行融合
        if graph_repr is not None:
            # 投影图表示到与描述符相同的维度
            graph_repr_proj = self.graph_proj(graph_repr)
            concat = torch.cat([graph_repr_proj, descriptor_repr], dim=-1)
            fusion_weight = self.fusion_gate(concat)
            fused = fusion_weight * descriptor_repr + (1 - fusion_weight) * graph_repr_proj
            result['fused_repr'] = fused

        return result

    def get_gate_values_mean(self) -> torch.Tensor:
        """获取平均门控值（用于分析特征重要性）"""
        if self.use_gating and hasattr(self.feature_gate, 'base_gate'):
            return torch.sigmoid(self.feature_gate.base_gate)
        return None


# 预定义的描述符名称（可用于可解释性分析）
QUANTUM_DESCRIPTOR_NAMES = [
    # 轨道能量相关
    'HOMO', 'LUMO', 'HOMO-1', 'LUMO+1', 'gap',
    # 热力学
    'Gibbs_free_energy', 'electronic_energy', 'zero_point_energy',
    # 偶极矩
    'dipole_x', 'dipole_y', 'dipole_z', 'dipole_total',
    # Mulliken电荷（前57个原子）
    *[f'mulliken_charge_{i}' for i in range(57)],
]

