"""
几何编码器 (Geometric Encoder)
用于处理分子的3D坐标信息

作者: AI助手
日期: 2024-06-11
"""

import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from typing import Optional, Tuple


class GaussianSmearing(nn.Module):
    """
    高斯模糊，用于将距离转换为特征向量
    """
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class GeometricEncoder(MessagePassing):
    """
    几何编码器，使用类SchNet的架构来处理3D坐标
    通过连续滤波器卷积（CFConv）来更新原子表示
    """
    def __init__(self,
                 embed_dim: int,
                 hidden_dim: int,
                 num_rbf: int,
                 max_distance: float):
        """
        初始化几何编码器

        Args:
            embed_dim (int): 输入/输出的嵌入维度
            hidden_dim (int): 滤波器网络的隐藏维度
            num_rbf (int): 径向基函数的数量
            max_distance (float): 考虑的最大距离
        """
        super(GeometricEncoder, self).__init__(aggr='add')
        
        self.embed_dim = embed_dim
        
        # 距离编码
        self.distance_expansion = GaussianSmearing(
            start=0.0, stop=max_distance, num_gaussians=num_rbf
        )
        
        # 滤波器网络 (用于生成卷积权重)
        self.filter_net = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim * embed_dim)
        )
        
        # 原子表示更新网络
        self.update_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        """重置网络参数"""
        for layer in self.filter_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)
        for layer in self.update_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)

    def forward(self, 
                x: torch.Tensor, 
                pos: torch.Tensor, 
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x (torch.Tensor): 原子特征 (num_atoms, embed_dim)
            pos (torch.Tensor): 原子坐标 (num_atoms, 3)
            edge_index (torch.Tensor): 边索引 (2, num_bonds)

        Returns:
            torch.Tensor: 更新后的原子表示 (num_atoms, embed_dim)
        """
        # 计算距离
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)
        
        # 编码距离
        dist_emb = self.distance_expansion(dist)
        
        # 消息传递
        updated_x = self.propagate(edge_index, x=x, dist_emb=dist_emb)
        
        # 更新原子表示
        x = x + self.update_net(updated_x)
        return x

    def message(self, x_j: torch.Tensor, dist_emb: torch.Tensor) -> torch.Tensor:
        """
        消息计算

        Args:
            x_j (torch.Tensor): 邻居原子特征
            dist_emb (torch.Tensor): 距离编码

        Returns:
            torch.Tensor: 消息
        """
        # 生成滤波器权重
        filter_weights = self.filter_net(dist_emb).view(
            -1, self.embed_dim, self.embed_dim
        )
        
        # 计算消息
        # (num_bonds, embed_dim) -> (num_bonds, 1, embed_dim)
        x_j_reshaped = x_j.unsqueeze(1)
        
        # (num_bonds, 1, embed_dim) x (num_bonds, embed_dim, embed_dim) -> (num_bonds, 1, embed_dim)
        message = torch.bmm(x_j_reshaped, filter_weights)
        
        # (num_bonds, 1, embed_dim) -> (num_bonds, embed_dim)
        return message.squeeze(1)


if __name__ == '__main__':
    # --- 测试几何编码器 ---
    
    # 模拟输入
    num_atoms = 50
    embed_dim = 128
    hidden_dim = 256
    num_rbf = 50
    max_distance = 10.0
    
    atom_features = torch.randn(num_atoms, embed_dim)
    positions = torch.randn(num_atoms, 3)
    edge_index = torch.randint(0, num_atoms, (2, 200))
    
    # 创建编码器
    geo_encoder = GeometricEncoder(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_rbf=num_rbf,
        max_distance=max_distance
    )
    
    print("几何编码器测试:")
    print(f"输入原子特征形状: {atom_features.shape}")
    print(f"输入坐标形状: {positions.shape}")
    
    # 前向传播
    output = geo_encoder(atom_features, positions, edge_index)
    
    print(f"输出原子特征形状: {output.shape}")
    
    # 验证输出形状
    assert output.shape == (num_atoms, embed_dim)
    print("✓ 测试通过!") 