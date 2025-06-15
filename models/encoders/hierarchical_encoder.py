"""
分层化学语义编码器 (Hierarchical Chemical Encoder)
基于ToxD4C原有数据格式，实现多层次分子表示学习

作者: AI助手
日期: 2024-06-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Fragments
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.nn.conv import GCNConv


class FunctionalGroupEncoder(nn.Module):
    """官能团编码器 - 识别和编码常见化学官能团"""
    
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 定义常见官能团的SMARTS模式
        self.functional_groups = {
            'benzene': '[c1ccccc1]',
            'carboxyl': '[CX3](=O)[OX2H1]',
            'hydroxyl': '[OX2H]',
            'amino': '[NX3;H2,H1;!$(NC=O)]',
            'carbonyl': '[CX3]=[OX1]',
            'ester': '[#6][CX3](=O)[OX2H0][#6]',
            'ether': '[OD2]([#6])[#6]',
            'amide': '[CX3](=[OX1])[NX3H2]',
            'nitro': '[NX3+](=O)[O-]',
            'sulfhydryl': '[SH]',
            'phosphate': '[PX4](=[OX1])[OX2H,OX1-]',
            'halogen': '[F,Cl,Br,I]',
        }
        
        # 为每个官能团创建嵌入
        self.fg_embeddings = nn.ModuleDict({
            name: nn.Linear(1, embed_dim // len(self.functional_groups))
            for name in self.functional_groups.keys()
        })
        
        # 官能团聚合层
        self.fg_aggregator = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.1)
        
    def identify_functional_groups(self, mol) -> Dict[str, int]:
        """识别分子中的官能团"""
        if mol is None:
            return {name: 0 for name in self.functional_groups.keys()}
        
        fg_counts = {}
        for name, smarts in self.functional_groups.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern:
                matches = mol.GetSubstructMatches(pattern)
                fg_counts[name] = len(matches)
            else:
                fg_counts[name] = 0
        
        return fg_counts
    
    def forward(self, mol_batch: List) -> torch.Tensor:
        """
        前向传播
        Args:
            mol_batch: RDKit分子对象列表
        Returns:
            官能团特征向量 [batch_size, embed_dim]
        """
        batch_size = len(mol_batch)
        fg_features = []
        
        for mol in mol_batch:
            fg_counts = self.identify_functional_groups(mol)
            
            # 编码每个官能团
            fg_embeds = []
            for name, count in fg_counts.items():
                count_tensor = torch.tensor([float(count)], device=next(self.parameters()).device)
                embed = self.fg_embeddings[name](count_tensor)
                fg_embeds.append(embed)
            
            # 连接所有官能团嵌入
            mol_fg_features = torch.cat(fg_embeds, dim=0)
            fg_features.append(mol_fg_features)
        
        # 批处理
        fg_batch = torch.stack(fg_features, dim=0)  # [batch_size, embed_dim]
        
        # 聚合和标准化
        output = self.fg_aggregator(fg_batch)
        output = self.dropout(output)
        
        return output


class ScaffoldEncoder(nn.Module):
    """分子骨架编码器 - 提取和编码分子骨架特征"""
    
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 骨架特征编码器
        self.scaffold_features = nn.Sequential(
            nn.Linear(10, embed_dim // 2),  # 骨架描述符维度
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
    def extract_scaffold_features(self, mol) -> np.ndarray:
        """提取分子骨架特征"""
        if mol is None:
            return np.zeros(10)
        
        features = []
        try:
            # 基本骨架特征
            features.append(mol.GetNumAtoms())  # 原子数
            features.append(mol.GetNumBonds())  # 键数
            features.append(mol.GetNumHeavyAtoms())  # 重原子数
            features.append(Descriptors.NumAromaticRings(mol))  # 芳香环数
            features.append(Descriptors.NumAliphaticRings(mol))  # 脂肪环数
            features.append(Descriptors.RingCount(mol))  # 总环数
            features.append(Descriptors.FractionCsp3(mol) or 0)  # sp3碳分数
            features.append(Descriptors.HeavyAtomCount(mol))  # 重原子计数
            features.append(Descriptors.NumRotatableBonds(mol))  # 可旋转键数
            features.append(Descriptors.TPSA(mol))  # 极性表面积
            
        except Exception as e:
            print(f"Error calculating scaffold features: {e}")
            features = [0.0] * 10
        
        return np.array(features, dtype=np.float32)
    
    def forward(self, mol_batch: List) -> torch.Tensor:
        """
        前向传播
        Args:
            mol_batch: RDKit分子对象列表
        Returns:
            骨架特征向量 [batch_size, embed_dim]
        """
        batch_features = []
        
        for mol in mol_batch:
            scaffold_feat = self.extract_scaffold_features(mol)
            batch_features.append(scaffold_feat)
        
        # 转换为张量
        batch_tensor = torch.tensor(np.stack(batch_features), 
                                  device=next(self.parameters()).device)
        
        # 特征编码
        output = self.scaffold_features(batch_tensor)
        
        return output


class AtomicLevelEncoder(nn.Module):
    """原子级编码器 - 增强原子特征表示"""
    
    def __init__(self, embed_dim: int = 256, max_atoms: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_atoms = max_atoms
        
        # 原子特征维度 (与ToxD4C原始特征保持一致)
        self.atom_feature_dim = 9  # 根据ToxD4C的原子特征维度
        
        # 原子特征编码器
        self.atom_encoder = nn.Sequential(
            nn.Linear(self.atom_feature_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # 位置编码
        self.pos_encoder = nn.Parameter(torch.randn(max_atoms, embed_dim))
        
        # 原子级自注意力
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, atom_features: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            atom_features: 原子特征 [batch_size, max_atoms, atom_feature_dim]
            atom_mask: 原子掩码 [batch_size, max_atoms]
        Returns:
            原子级表示 [batch_size, embed_dim]
        """
        batch_size, num_atoms, _ = atom_features.shape
        
        # 原子特征编码
        atom_embeds = self.atom_encoder(atom_features)  # [batch_size, max_atoms, embed_dim]
        
        # 添加位置编码
        pos_embeds = self.pos_encoder[:num_atoms].unsqueeze(0).expand(batch_size, -1, -1)
        atom_embeds = atom_embeds + pos_embeds
        
        # 自注意力
        attn_mask = ~atom_mask.bool()  # 转换掩码格式
        atom_embeds, _ = self.self_attention(
            atom_embeds, atom_embeds, atom_embeds,
            key_padding_mask=attn_mask
        )
        
        atom_embeds = self.norm(atom_embeds)
        
        # 全局池化 (考虑掩码)
        mask_expanded = atom_mask.unsqueeze(-1).float()
        masked_embeds = atom_embeds * mask_expanded
        
        # 平均池化
        atom_counts = atom_mask.sum(dim=1, keepdim=True).float()
        global_repr = masked_embeds.sum(dim=1) / (atom_counts + 1e-8)
        
        return global_repr


class HierarchicalEncoder(nn.Module):
    """
    分层编码器，用于在不同尺度上对分子图进行编码。
    通过在不同层次上进行图卷积和池化，捕捉局部和全局的结构信息。
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 hierarchy_levels: List[int],
                 dropout: float = 0.1):
        """
        初始化分层编码器

        Args:
            input_dim (int): 输入特征维度
            hidden_dim (int): 隐藏层维度
            hierarchy_levels (List[int]): 定义层次结构中每一层的GCN层数
            dropout (float): Dropout比率
        """
        super().__init__()
        
        self.hierarchy_levels = hierarchy_levels
        self.num_hierarchies = len(hierarchy_levels)
        
        # 创建多个GCN层级
        self.level_encoders = nn.ModuleList()
        current_dim = input_dim
        
        for num_layers in hierarchy_levels:
            level_encoder = self._create_gcn_block(current_dim, hidden_dim, num_layers, dropout)
            self.level_encoders.append(level_encoder)
            current_dim = hidden_dim  # 后续层级的输入维度是前一级的输出维度
            
        # 特征融合层
        self.fusion_layer = nn.Linear(hidden_dim * self.num_hierarchies, hidden_dim)
        self.fusion_activation = nn.ReLU()
        self.fusion_dropout = nn.Dropout(dropout)

    def _create_gcn_block(self, 
                          in_channels: int, 
                          out_channels: int, 
                          num_layers: int, 
                          dropout: float) -> nn.Module:
        """
        创建GCN编码块 (多层GCN + 激活 + 归一化)
        """
        layers = nn.ModuleList()
        
        # 第一层 (输入维度 -> 输出维度)
        layers.append(GCNConv(in_channels, out_channels))
        
        # 后续层 (输出维度 -> 输出维度)
        for _ in range(num_layers - 1):
            layers.append(GCNConv(out_channels, out_channels))
        
        # 激活和归一化
        activations = nn.ModuleList([nn.ReLU() for _ in range(num_layers)])
        batch_norms = nn.ModuleList([nn.BatchNorm1d(out_channels) for _ in range(num_layers)])
        
        return nn.ModuleDict({
            'convs': layers,
            'activations': activations,
            'norms': batch_norms
        })

    def forward(self, 
                x: torch.Tensor, 
                edge_index: torch.Tensor, 
                batch: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x (torch.Tensor): 原子特征 (num_atoms, input_dim)
            edge_index (torch.Tensor): 边索引 (2, num_bonds)
            batch (torch.Tensor): 批次索引 (num_atoms)

        Returns:
            torch.Tensor: 融合后的图级别表示 (batch_size, hidden_dim)
        """
        level_representations = []
        current_x = x
        
        # 遍历每个层级
        for i, level_encoder in enumerate(self.level_encoders):
            # GCN编码
            level_x = current_x
            for j, conv in enumerate(level_encoder['convs']):
                level_x = conv(level_x, edge_index)
                level_x = level_encoder['norms'][j](level_x)
                level_x = level_encoder['activations'][j](level_x)
            
            # 池化得到图级别表示
            graph_rep = global_mean_pool(level_x, batch)
            level_representations.append(graph_rep)
            
            # 更新下一层级的输入
            current_x = level_x
            
        # 融合所有层级的表示
        fused_rep = torch.cat(level_representations, dim=1)
        
        # 通过融合层进行处理
        fused_rep = self.fusion_layer(fused_rep)
        fused_rep = self.fusion_activation(fused_rep)
        fused_rep = self.fusion_dropout(fused_rep)
        
        return fused_rep


class HierarchicalChemicalEncoder(nn.Module):
    """
    分层化学语义编码器主类
    整合原子级、官能团级、骨架级和分子级的多层次表示
    """
    
    def __init__(self, embed_dim: int = 512, max_atoms: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 各层级编码器
        self.atomic_encoder = AtomicLevelEncoder(embed_dim // 4, max_atoms)
        self.fg_encoder = FunctionalGroupEncoder(embed_dim // 4)
        self.scaffold_encoder = ScaffoldEncoder(embed_dim // 4)
        
        # 分子级全局特征编码器
        self.molecular_encoder = nn.Sequential(
            nn.Linear(15, embed_dim // 4),  # 分子描述符维度
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 层次化融合注意力
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=embed_dim // 4,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # 最终融合层
        self.final_fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def extract_molecular_features(self, mol) -> np.ndarray:
        """提取分子级全局特征"""
        if mol is None:
            return np.zeros(15)
        
        features = []
        try:
            # 基本分子属性
            features.append(Descriptors.MolWt(mol))  # 分子量
            features.append(Descriptors.MolLogP(mol))  # LogP
            features.append(Descriptors.NumHDonors(mol))  # 氢键供体
            features.append(Descriptors.NumHAcceptors(mol))  # 氢键受体
            features.append(Descriptors.TPSA(mol))  # 极性表面积
            features.append(Descriptors.NumRotatableBonds(mol))  # 可旋转键
            features.append(Descriptors.FractionCsp3(mol) or 0)  # sp3分数
            features.append(Descriptors.HeavyAtomCount(mol))  # 重原子数
            features.append(Descriptors.RingCount(mol))  # 环数
            features.append(Descriptors.AromaticProportion(mol))  # 芳香性比例
            features.append(Descriptors.BalabanJ(mol))  # Balaban指数
            features.append(Descriptors.BertzCT(mol))  # Bertz复杂度
            features.append(Descriptors.Chi0(mol))  # 连接性指数
            features.append(Descriptors.Chi1(mol))
            features.append(Descriptors.Kappa1(mol))  # Kappa形状指数
            
        except Exception as e:
            print(f"Error calculating molecular features: {e}")
            features = [0.0] * 15
            
        return np.array(features, dtype=np.float32)
    
    def forward(self, 
                atom_features: torch.Tensor,
                atom_mask: torch.Tensor,
                mol_batch: List,
                fingerprints: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        分层化学语义编码前向传播
        
        Args:
            atom_features: 原子特征 [batch_size, max_atoms, atom_feat_dim]
            atom_mask: 原子掩码 [batch_size, max_atoms]  
            mol_batch: RDKit分子对象列表
            fingerprints: 分子指纹 (可选) [batch_size, fp_dim]
            
        Returns:
            分层化学语义表示 [batch_size, embed_dim]
        """
        batch_size = len(mol_batch)
        
        # 1. 原子级编码
        atomic_repr = self.atomic_encoder(atom_features, atom_mask)
        
        # 2. 官能团级编码
        fg_repr = self.fg_encoder(mol_batch)
        
        # 3. 骨架级编码  
        scaffold_repr = self.scaffold_encoder(mol_batch)
        
        # 4. 分子级编码
        mol_features = []
        for mol in mol_batch:
            mol_feat = self.extract_molecular_features(mol)
            mol_features.append(mol_feat)
        
        mol_feat_tensor = torch.tensor(np.stack(mol_features),
                                     device=atom_features.device)
        molecular_repr = self.molecular_encoder(mol_feat_tensor)
        
        # 5. 层次化融合
        # 将所有层级表示堆叠
        level_reprs = torch.stack([
            atomic_repr,
            fg_repr, 
            scaffold_repr,
            molecular_repr
        ], dim=1)  # [batch_size, 4, embed_dim//4]
        
        # 层次化注意力融合
        fused_reprs, attention_weights = self.fusion_attention(
            level_reprs, level_reprs, level_reprs
        )
        
        # 展平并最终融合
        final_repr = fused_reprs.view(batch_size, -1)  # [batch_size, embed_dim]
        final_repr = self.final_fusion(final_repr)
        final_repr = self.layer_norm(final_repr)
        
        return final_repr


# 使用示例
if __name__ == "__main__":
    # 创建编码器
    encoder = HierarchicalChemicalEncoder(embed_dim=512, max_atoms=256)
    
    # 模拟输入数据
    batch_size = 4
    max_atoms = 256
    atom_feat_dim = 9
    
    atom_features = torch.randn(batch_size, max_atoms, atom_feat_dim)
    atom_mask = torch.ones(batch_size, max_atoms)
    
    # 模拟分子对象 (实际使用中会是RDKit分子对象)
    mol_batch = [None] * batch_size
    
    # 前向传播
    try:
        output = encoder(atom_features, atom_mask, mol_batch)
        print(f"输出形状: {output.shape}")
        print(f"分层化学语义编码器创建成功!")
    except Exception as e:
        print(f"错误: {e}") 