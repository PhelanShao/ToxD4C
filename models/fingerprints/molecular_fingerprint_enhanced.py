"""
分子指纹增强模块 (Molecular Fingerprint Enhanced Module)
融合多种分子指纹特征，包括ECFP、MACCS、RDKit描述符等

作者: AI助手
日期: 2024-06-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, Fragments
from rdkit.Chem import rdFingerprintGenerator
import warnings

# 抑制RDKit警告
warnings.filterwarnings('ignore')


class FingerprintEncoder(nn.Module):
    """分子指纹编码器"""
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 fingerprint_type: str = 'ECFP',
                 hidden_dims: List[int] = [512, 256],
                 dropout: float = 0.1,
                 use_batch_norm: bool = True):
        super().__init__()
        self.fingerprint_type = fingerprint_type
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class MolecularDescriptorCalculator:
    """分子描述符计算器"""
    
    def __init__(self):
        # 常用的分子描述符
        self.descriptor_functions = {
            'MW': Descriptors.MolWt,  # 分子量
            'LogP': Descriptors.MolLogP,  # 脂水分配系数
            'HBD': Descriptors.NumHDonors,  # 氢键供体数
            'HBA': Descriptors.NumHAcceptors,  # 氢键受体数
            'TPSA': Descriptors.TPSA,  # 拓扑极性表面积
            'NumRotatableBonds': Descriptors.NumRotatableBonds,  # 可旋转键数
            'NumAromaticRings': Descriptors.NumAromaticRings,  # 芳香环数
            'FractionCSP3': Descriptors.FractionCSP3,  # sp3碳比例
            'NumAliphaticCarbocycles': Descriptors.NumAliphaticCarbocycles,
            'NumAliphaticHeterocycles': Descriptors.NumAliphaticHeterocycles,
            'NumAliphaticRings': Descriptors.NumAliphaticRings,
            'NumAromaticCarbocycles': Descriptors.NumAromaticCarbocycles,
            'NumAromaticHeterocycles': Descriptors.NumAromaticHeterocycles,
            'RingCount': Descriptors.RingCount,
            'BertzCT': Descriptors.BertzCT,  # 分子复杂度
        }
    
    def calculate_descriptors(self, mol: Chem.Mol) -> np.ndarray:
        """计算分子描述符"""
        descriptors = []
        for desc_name, desc_func in self.descriptor_functions.items():
            try:
                value = desc_func(mol)
                if np.isnan(value) or np.isinf(value):
                    value = 0.0
                descriptors.append(value)
            except:
                descriptors.append(0.0)
        
        return np.array(descriptors)
    
    def get_descriptor_names(self) -> List[str]:
        """获取描述符名称"""
        return list(self.descriptor_functions.keys())


class FingerprintCalculator:
    """分子指纹计算器"""
    
    def __init__(self):
        self.calculator = MolecularDescriptorCalculator()
    
    def calculate_ecfp(self, mol: Chem.Mol, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
        """计算ECFP指纹"""
        try:
            fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
            fp = fp_gen.GetFingerprintAsBitVect(mol)
            return np.array(fp)
        except:
            return np.zeros(n_bits)
    
    def calculate_maccs(self, mol: Chem.Mol) -> np.ndarray:
        """计算MACCS指纹"""
        try:
            fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
            return np.array(fp)
        except:
            return np.zeros(167)  # MACCS有167位
    
    def calculate_rdkit_fp(self, mol: Chem.Mol, fp_size: int = 2048) -> np.ndarray:
        """计算RDKit指纹"""
        try:
            fp = Chem.RDKFingerprint(mol, fpSize=fp_size)
            return np.array(fp)
        except:
            return np.zeros(fp_size)
    
    def calculate_avalon_fp(self, mol: Chem.Mol, n_bits: int = 512) -> np.ndarray:
        """计算Avalon指纹"""
        try:
            from rdkit.Avalon import pyAvalonTools
            fp = pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)
            return np.array(fp)
        except:
            return np.zeros(n_bits)
    
    def calculate_atom_pair_fp(self, mol: Chem.Mol, n_bits: int = 2048) -> np.ndarray:
        """计算原子对指纹"""
        try:
            fp = rdMolDescriptors.GetAtomPairFingerprintAsBitVect(mol, nBits=n_bits)
            return np.array(fp)
        except:
            return np.zeros(n_bits)


class MolecularFingerprintModule(nn.Module):
    """分子指纹增强模块"""
    
    def __init__(self, 
                 output_dim: int = 512,
                 fingerprint_configs: Optional[Dict] = None,
                 use_attention_fusion: bool = True,
                 dropout: float = 0.1):
        """
        Args:
            output_dim: 输出特征维度
            fingerprint_configs: 指纹配置字典
            use_attention_fusion: 是否使用注意力融合
            dropout: Dropout比率
        """
        super().__init__()
        self.output_dim = output_dim
        self.use_attention_fusion = use_attention_fusion
        
        # 默认指纹配置
        if fingerprint_configs is None:
            fingerprint_configs = {
                'ecfp': {'n_bits': 2048, 'radius': 2},
                'maccs': {'n_bits': 167},
                'rdkit_fp': {'n_bits': 2048},
                'avalon': {'n_bits': 512},
                'atom_pair': {'n_bits': 2048},
                'descriptors': {'n_features': 15}  # 分子描述符数量
            }
        
        self.fingerprint_configs = fingerprint_configs
        
        # 指纹编码器
        self.fingerprint_encoders = nn.ModuleDict()
        for fp_name, config in fingerprint_configs.items():
            input_dim = config.get('n_bits', config.get('n_features', 512))
            self.fingerprint_encoders[fp_name] = FingerprintEncoder(
                input_dim=input_dim,
                output_dim=output_dim // len(fingerprint_configs),
                fingerprint_type=fp_name,
                dropout=dropout
            )
        
        # 特征融合
        total_fp_dim = output_dim
        
        if use_attention_fusion:
            # 注意力融合
            encoded_fp_dim = output_dim // len(fingerprint_configs)
            self.attention_weights = nn.Sequential(
                nn.Linear(total_fp_dim, total_fp_dim // 4),
                nn.ReLU(),
                nn.Linear(total_fp_dim // 4, len(fingerprint_configs)),
                nn.Softmax(dim=-1)
            )
            
            self.feature_transform = nn.Sequential(
                nn.Linear(encoded_fp_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            # 简单连接
            self.feature_fusion = nn.Sequential(
                nn.Linear(total_fp_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        
        # 分子指纹计算器（仅在需要时初始化）
        self.fp_calculator = None
    
    def _get_fp_calculator(self):
        """延迟初始化指纹计算器"""
        if self.fp_calculator is None:
            self.fp_calculator = FingerprintCalculator()
        return self.fp_calculator
    
    def calculate_fingerprints_from_smiles(self, smiles: List[str]) -> Dict[str, torch.Tensor]:
        """从SMILES计算分子指纹"""
        calculator = self._get_fp_calculator()
        
        batch_fingerprints = {fp_name: [] for fp_name in self.fingerprint_configs.keys()}
        
        for smi in smiles:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    # 如果SMILES无效，使用零向量
                    for fp_name, config in self.fingerprint_configs.items():
                        input_dim = config.get('n_bits', config.get('n_features', 512))
                        batch_fingerprints[fp_name].append(np.zeros(input_dim))
                    continue
                
                # 计算各种指纹
                for fp_name, config in self.fingerprint_configs.items():
                    if fp_name == 'ecfp':
                        fp = calculator.calculate_ecfp(mol, 
                                                     radius=config.get('radius', 2),
                                                     n_bits=config.get('n_bits', 2048))
                    elif fp_name == 'maccs':
                        fp = calculator.calculate_maccs(mol)
                    elif fp_name == 'rdkit_fp':
                        fp = calculator.calculate_rdkit_fp(mol, fp_size=config.get('n_bits', 2048))
                    elif fp_name == 'avalon':
                        fp = calculator.calculate_avalon_fp(mol, n_bits=config.get('n_bits', 512))
                    elif fp_name == 'atom_pair':
                        fp = calculator.calculate_atom_pair_fp(mol, n_bits=config.get('n_bits', 2048))
                    elif fp_name == 'descriptors':
                        fp = calculator.calculator.calculate_descriptors(mol)
                    else:
                        input_dim = config.get('n_bits', config.get('n_features', 512))
                        fp = np.zeros(input_dim)
                    
                    batch_fingerprints[fp_name].append(fp)
                    
            except Exception as e:
                # 处理异常情况
                for fp_name, config in self.fingerprint_configs.items():
                    input_dim = config.get('n_bits', config.get('n_features', 512))
                    batch_fingerprints[fp_name].append(np.zeros(input_dim))
        
        # 转换为张量
        tensor_fingerprints = {}
        for fp_name, fp_list in batch_fingerprints.items():
            tensor_fingerprints[fp_name] = torch.FloatTensor(np.array(fp_list))
        
        return tensor_fingerprints
    
    def forward(self, 
                fingerprints: Optional[Dict[str, torch.Tensor]] = None,
                smiles: Optional[List[str]] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            fingerprints: 预计算的指纹特征字典
            smiles: SMILES字符串列表（如果fingerprints为None）
            
        Returns:
            fused_features: 融合后的特征 [batch_size, output_dim]
        """
        if fingerprints is None and smiles is not None:
            fingerprints = self.calculate_fingerprints_from_smiles(smiles)
        elif fingerprints is None:
            raise ValueError("必须提供fingerprints或smiles参数之一")
        
        # 确保所有指纹都在同一设备上
        device = next(self.parameters()).device
        for fp_name in fingerprints:
            fingerprints[fp_name] = fingerprints[fp_name].to(device)
        
        # 编码各种指纹
        encoded_fingerprints = []
        fp_names = []
        
        for fp_name, fp_tensor in fingerprints.items():
            if fp_name in self.fingerprint_encoders:
                encoded_fp = self.fingerprint_encoders[fp_name](fp_tensor)
                encoded_fingerprints.append(encoded_fp)
                fp_names.append(fp_name)
        
        if not encoded_fingerprints:
            raise ValueError("没有有效的指纹特征")
        
        # 融合特征
        if self.use_attention_fusion and len(encoded_fingerprints) > 1:
            # 连接所有编码后的指纹
            concatenated_fps = torch.cat(encoded_fingerprints, dim=-1)
            
            # 计算注意力权重
            attention_weights = self.attention_weights(concatenated_fps)
            
            # 加权融合
            weighted_fps = []
            start_idx = 0
            for i, encoded_fp in enumerate(encoded_fingerprints):
                end_idx = start_idx + encoded_fp.size(-1)
                weight = attention_weights[:, i:i+1]
                weighted_fp = encoded_fp * weight
                weighted_fps.append(weighted_fp)
                start_idx = end_idx
            
            # 求和并变换
            summed_fps = torch.stack(weighted_fps, dim=0).sum(dim=0)
            final_features = self.feature_transform(summed_fps)
        else:
            # 简单连接和融合
            concatenated_fps = torch.cat(encoded_fingerprints, dim=-1)
            final_features = self.feature_fusion(concatenated_fps)
        
        return final_features
    
    def get_fingerprint_importance(self, 
                                 fingerprints: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """获取各指纹的重要性权重"""
        if not self.use_attention_fusion:
            # 如果不使用注意力融合，返回均等权重
            num_fps = len(fingerprints)
            return {fp_name: 1.0/num_fps for fp_name in fingerprints.keys()}
        
        device = next(self.parameters()).device
        for fp_name in fingerprints:
            fingerprints[fp_name] = fingerprints[fp_name].to(device)
        
        # 编码指纹
        encoded_fingerprints = []
        fp_names = []
        
        for fp_name, fp_tensor in fingerprints.items():
            if fp_name in self.fingerprint_encoders:
                encoded_fp = self.fingerprint_encoders[fp_name](fp_tensor)
                encoded_fingerprints.append(encoded_fp)
                fp_names.append(fp_name)
        
        # 计算注意力权重
        concatenated_fps = torch.cat(encoded_fingerprints, dim=-1)
        attention_weights = self.attention_weights(concatenated_fps)
        
        # 平均注意力权重
        avg_weights = attention_weights.mean(dim=0).cpu().detach().numpy()
        
        return {fp_names[i]: float(avg_weights[i]) for i in range(len(fp_names))}


# 使用示例
if __name__ == "__main__":
    # 创建分子指纹增强模块
    fp_module = MolecularFingerprintModule(
        output_dim=512,
        use_attention_fusion=True
    )
    
    # 测试SMILES
    test_smiles = [
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # 布洛芬
        "CC1=CC=C(C=C1)C(=O)C2=CC=C(C=C2)C",  # 苯乙酮
        "CCO"  # 乙醇
    ]
    
    # 从SMILES计算指纹并前向传播
    features = fp_module(smiles=test_smiles)
    print(f"输出特征形状: {features.shape}")
    
    # 获取指纹重要性
    fingerprints = fp_module.calculate_fingerprints_from_smiles(test_smiles)
    importance = fp_module.get_fingerprint_importance(fingerprints)
    print("指纹重要性:")
    for fp_name, weight in importance.items():
        print(f"{fp_name}: {weight:.4f}") 