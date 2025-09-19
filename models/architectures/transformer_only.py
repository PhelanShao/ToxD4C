"""
Transformer-only architecture for ablation studies
当禁用GNN组件时使用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_mean_pool
import math


class PositionalEncoding(nn.Module):
    """位置编码模块"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [seq_len, batch_size, d_model]
        """
        return x + self.pe[:x.size(0), :]


class TransformerOnly(nn.Module):
    """仅使用Transformer的编码器（用于消融实验）"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 max_atoms: int = 200):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_atoms = max_atoms
        
        # 输入投影
        if input_dim != hidden_dim:
            self.input_projection = nn.Linear(input_dim, hidden_dim)
        else:
            self.input_projection = nn.Identity()
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim, max_atoms)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=False  # [seq_len, batch, features]
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # 输出投影
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges] (在此架构中不使用)
            batch: 批次索引 [num_nodes]
        
        Returns:
            graph_representation: [batch_size, hidden_dim]
        """
        batch_size = batch.max().item() + 1 if batch is not None else 1
        
        # 投影到隐藏维度
        x = self.input_projection(x)  # [num_nodes, hidden_dim]
        
        # 转换为密集批次格式
        x_dense, mask = to_dense_batch(x, batch, max_num_nodes=self.max_atoms)
        # x_dense: [batch_size, max_atoms, hidden_dim]
        # mask: [batch_size, max_atoms]
        
        # 转换为Transformer期望的格式 [seq_len, batch, features]
        x_dense = x_dense.transpose(0, 1)  # [max_atoms, batch_size, hidden_dim]
        
        # 添加位置编码
        x_dense = self.pos_encoding(x_dense)
        
        # 创建注意力掩码（True表示忽略的位置）
        # mask: [batch_size, max_atoms] -> [batch_size, max_atoms]
        attn_mask = ~mask  # 反转掩码，True表示要忽略的位置
        
        # Transformer编码
        # 注意：nn.TransformerEncoder期望src_key_padding_mask的形状为[batch_size, seq_len]
        encoded = self.transformer_encoder(
            x_dense, 
            src_key_padding_mask=attn_mask
        )  # [max_atoms, batch_size, hidden_dim]
        
        # 转换回 [batch_size, max_atoms, hidden_dim]
        encoded = encoded.transpose(0, 1)
        
        # 应用掩码并进行全局平均池化
        mask_expanded = mask.unsqueeze(-1).expand_as(encoded)
        encoded_masked = encoded * mask_expanded.float()
        
        # 计算每个图的平均表示
        graph_lengths = mask.sum(dim=1, keepdim=True).float()  # [batch_size, 1]
        graph_representation = encoded_masked.sum(dim=1) / graph_lengths.clamp(min=1)
        # [batch_size, hidden_dim]
        
        # 输出投影和dropout
        graph_representation = self.output_projection(graph_representation)
        graph_representation = self.dropout(graph_representation)
        
        return graph_representation
    
    def get_attention_weights(self, x, edge_index, batch=None):
        """获取注意力权重用于可视化"""
        # 这里可以实现注意力权重的提取
        # 为了简化，暂时返回None
        return None


class TransformerOnlyWithEdges(TransformerOnly):
    """
    考虑边信息的Transformer-only架构
    通过边信息构建注意力偏置
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 边嵌入（可选）
        self.edge_embedding = nn.Linear(1, self.num_heads)  # 简单的边权重嵌入
        
    def create_attention_bias(self, edge_index, batch, max_atoms):
        """
        基于图的边信息创建注意力偏置
        
        Args:
            edge_index: [2, num_edges]
            batch: [num_nodes]
            max_atoms: int
            
        Returns:
            attention_bias: [batch_size, num_heads, max_atoms, max_atoms]
        """
        batch_size = batch.max().item() + 1
        device = edge_index.device
        
        # 初始化注意力偏置矩阵
        attention_bias = torch.zeros(
            batch_size, self.num_heads, max_atoms, max_atoms,
            device=device
        )
        
        # 为每个批次构建邻接矩阵
        for b in range(batch_size):
            # 获取当前批次的节点
            node_mask = (batch == b)
            node_indices = torch.where(node_mask)[0]
            
            if len(node_indices) == 0:
                continue
                
            # 重新映射节点索引到局部索引
            global_to_local = {global_idx.item(): local_idx 
                             for local_idx, global_idx in enumerate(node_indices)}
            
            # 找到当前批次的边
            edge_mask = torch.isin(edge_index[0], node_indices) & \
                       torch.isin(edge_index[1], node_indices)
            batch_edges = edge_index[:, edge_mask]
            
            # 转换为局部索引
            for edge_idx in range(batch_edges.size(1)):
                src_global = batch_edges[0, edge_idx].item()
                dst_global = batch_edges[1, edge_idx].item()
                
                if src_global in global_to_local and dst_global in global_to_local:
                    src_local = global_to_local[src_global]
                    dst_local = global_to_local[dst_global]
                    
                    # 设置注意力偏置（连接的节点之间注意力权重更高）
                    attention_bias[b, :, src_local, dst_local] = 1.0
                    attention_bias[b, :, dst_local, src_local] = 1.0  # 无向图
        
        return attention_bias
    
    def forward(self, x, edge_index, batch=None):
        """
        带边信息的前向传播
        """
        # 基础的Transformer处理
        return super().forward(x, edge_index, batch)
        
        # TODO: 如果需要使用边信息，可以在这里实现
        # 目前为了简化，直接使用父类的实现
