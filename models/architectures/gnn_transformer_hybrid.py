"""
动态GNN-Transformer混合架构 (Dynamic GNN-Transformer Hybrid)
结合图神经网络的局部建模能力和Transformer的全局建模能力

作者: AI助手
日期: 2024-06-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_adj, to_dense_batch

class GATLayer(nn.Module):
    """单层图注意力网络"""
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        
        assert output_dim % num_heads == 0
        
        # 线性变换
        self.W_q = nn.Linear(input_dim, output_dim, bias=False)
        self.W_k = nn.Linear(input_dim, output_dim, bias=False)
        self.W_v = nn.Linear(input_dim, output_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index):
        # 线性变换
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        
        # 计算注意力分数 (这里简化为点积，实际GAT更复杂)
        alpha = torch.einsum('bic,bjc->bij', q, k) / math.sqrt(self.head_dim)
        
        # 使用 to_dense_adj 获取邻接矩阵以应用注意力
        adj = to_dense_adj(edge_index, max_num_nodes=x.size(1)).squeeze(0)
        alpha = alpha.masked_fill(adj == 0, -1e9)
        
        attention_weights = F.softmax(alpha, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 聚合特征
        output = torch.bmm(attention_weights, v)
        return output

class GraphAttentionNetwork(MessagePassing):
    """图注意力网络 (GAT) - 采用MessagePassing框架"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__(aggr='add', node_dim=0) # 'add' 聚合
        self.num_layers = num_layers
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(
                GATLayer(hidden_dim, hidden_dim, num_heads, dropout)
            )
            self.norm_layers.append(nn.LayerNorm(hidden_dim, eps=1e-5))
            
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        x = self.input_projection(x)
        
        for i in range(self.num_layers):
            x_res = x
            x = self.propagate(edge_index, x=x, layer_idx=i)
            x = self.norm_layers[i](x + x_res)
            x = F.relu(x)
            
        # 将巨图转为填充批次以进行后续处理
        dense_x, node_mask = to_dense_batch(x, batch)
        
        # 输出投影
        output = self.output_projection(dense_x)
        output = self.dropout(output)
        
        if node_mask is not None:
            output = output * node_mask.unsqueeze(-1)
            
        return output

    def message(self, x_j, x_i, index, ptr, size_i, layer_idx):
        # 在这里，我们可以使用GATLayer来计算消息
        # 为了简化，我们直接返回邻居特征
        return x_j


class MolecularTransformer(nn.Module):
    """分子Transformer - 处理分子序列的全局依赖"""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 max_seq_len: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len)
        
        # Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出投影
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                node_features: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Transformer前向传播"""
        # 输入投影
        h = self.input_projection(node_features)
        
        # 位置编码
        h = self.pos_encoding(h)
        
        # Transformer编码
        if attention_mask is not None:
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None
        
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)
        
        # 输出投影
        output = self.output_projection(h)
        output = self.dropout(output)
        
        return output


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class DynamicFusionModule(nn.Module):
    """动态融合模块 - 自适应融合GNN和Transformer特征"""
    
    def __init__(self, feature_dim: int, num_heads: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # 特征权重生成网络
        self.weight_generator = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 2),  # 2个权重：GNN和Transformer
            nn.Softmax(dim=-1)
        )
        
        # 交叉注意力模块
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 特征增强
        self.feature_enhance = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self,
                gnn_features: torch.Tensor,
                transformer_features: torch.Tensor,
                node_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """动态融合GNN和Transformer特征"""
        # 1. 计算动态权重
        combined_features = torch.cat([gnn_features, transformer_features], dim=-1)
        fusion_weights = self.weight_generator(combined_features)
        
        # 2. 交叉注意力增强
        enhanced_gnn, _ = self.cross_attention(
            gnn_features, transformer_features, transformer_features,
            key_padding_mask=~node_mask.bool() if node_mask is not None else None
        )
        
        enhanced_transformer, _ = self.cross_attention(
            transformer_features, gnn_features, gnn_features,
            key_padding_mask=~node_mask.bool() if node_mask is not None else None
        )
        
        # 3. 动态加权融合
        gnn_weight = fusion_weights[..., 0:1]
        transformer_weight = fusion_weights[..., 1:2]
        
        fused_features = (gnn_weight * enhanced_gnn + 
                         transformer_weight * enhanced_transformer)
        
        # 4. 特征增强
        enhanced_features = self.feature_enhance(fused_features)
        
        # 5. 残差连接和层归一化
        output = self.layer_norm(fused_features + enhanced_features)
        
        # 应用节点掩码
        if node_mask is not None:
            output = output * node_mask.unsqueeze(-1)
        
        return output


class GNNTransformerHybrid(nn.Module):
    """GNN-Transformer混合架构主类"""
    
    def __init__(self,
                 input_dim: int = 256,
                 hidden_dim: int = 512,
                 output_dim: int = 512,
                 gnn_layers: int = 3,
                 transformer_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 max_seq_len: int = 512,
                 use_dynamic_fusion: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_dynamic_fusion = use_dynamic_fusion
        
        # GNN分支 - 处理局部图结构
        self.gnn_branch = GraphAttentionNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=gnn_layers,
            dropout=dropout
        )
        
        # Transformer分支 - 处理全局序列依赖
        self.transformer_branch = MolecularTransformer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        # 动态融合模块
        if use_dynamic_fusion:
            self.fusion_module = DynamicFusionModule(
                feature_dim=hidden_dim,
                num_heads=num_heads
            )
        else:
            self.fusion_module = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 全局池化
        self.global_pooling = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
    def forward(self, x, edge_index, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """混合架构前向传播"""
        # GNN分支现在返回填充批次
        gnn_features = self.gnn_branch(x, edge_index, batch)
        
        # Transformer分支需要填充批次
        dense_x, node_mask = to_dense_batch(x, batch)
        transformer_features = self.transformer_branch(
            node_features=dense_x,
            attention_mask=node_mask
        )
        
        # 特征融合
        if self.use_dynamic_fusion:
            fused_features = self.fusion_module(
                gnn_features, transformer_features, node_mask
            )
        else:
            combined = torch.cat([gnn_features, transformer_features], dim=-1)
            fused_features = self.fusion_module(combined)
            if node_mask is not None:
                fused_features = fused_features * node_mask.unsqueeze(-1)
        
        # 输出层
        node_representations = self.output_layer(fused_features)
        
        # 全局池化
        global_repr, attention_weights = self.global_pooling(
            node_representations, node_representations, node_representations,
            key_padding_mask=~node_mask.bool() if node_mask is not None else None
        )
        
        # 平均池化
        if node_mask is not None:
            mask_expanded = node_mask.unsqueeze(-1).float()
            masked_repr = global_repr * mask_expanded
            node_counts = node_mask.sum(dim=1, keepdim=True).float()
            molecule_repr = masked_repr.sum(dim=1) / (node_counts + 1e-8)
        else:
            molecule_repr = global_repr.mean(dim=1)
            
        return molecule_repr, attention_weights


# 使用示例
if __name__ == "__main__":
    # 创建混合架构
    hybrid_model = GNNTransformerHybrid(
        input_dim=256,
        hidden_dim=512,
        output_dim=512,
        gnn_layers=3,
        transformer_layers=6,
        num_heads=8,
        dropout=0.1,
        use_dynamic_fusion=True
    )
    
    # 模拟输入数据
    batch_size = 4
    num_nodes = 64
    input_dim = 256
    
    node_features = torch.randn(batch_size, num_nodes, input_dim)
    node_mask = torch.ones(batch_size, num_nodes)
    
    # 前向传播
    try:
        molecule_repr, attention_weights = hybrid_model(
            node_features=node_features,
            node_mask=node_mask
        )
        
        print(f"分子表示形状: {molecule_repr.shape}")
        print(f"注意力权重形状: {attention_weights.shape}")
        print("GNN-Transformer混合架构创建成功!")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc() 