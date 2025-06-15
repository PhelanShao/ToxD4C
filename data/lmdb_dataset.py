"""
真实LMDB数据集加载器
基于原始ToxD4C项目的数据加载逻辑，支持真实分子数据训练

作者: AI助手
日期: 2024-06-11
"""

import os
import sys
import lmdb
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem

# 添加配置路径
sys.path.append(str(Path(__file__).parent.parent))
from configs.toxd4c_config import get_enhanced_toxd4c_config

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LMDBToxD4CDataset(Dataset):
    """基于LMDB的ToxD4C数据集加载器"""
    
    def __init__(self, lmdb_path: str, max_atoms: int = 64):
        """
        初始化LMDB数据集
        
        Args:
            lmdb_path: LMDB文件路径
            max_atoms: 最大原子数量限制
        """
        self.lmdb_path = lmdb_path
        self.max_atoms = max_atoms
        
        # 打开LMDB环境
        self.env = lmdb.open(
            lmdb_path,
            subdir=False,  # LMDB文件是单个文件，不是目录
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256
        )
        
        # 收集有效的SMILES键
        self.smiles_keys = []
        logger.info("开始收集SMILES键...")
        count = 0
        with self.env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                count += 1
                if count % 5000 == 0:
                    logger.info(f"已处理 {count} 个条目...")
                
                try:
                    key_str = key.decode('ascii')
                    if not key_str.isdigit() and key_str != 'length':
                        self.smiles_keys.append(key_str)
                except:
                    continue
                
        
        logger.info(f"加载LMDB数据集: {lmdb_path}")
        logger.info(f"找到 {len(self.smiles_keys)} 个分子")
        
        # 原子符号到原子序数的映射
        self.atom_to_num = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
            'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
            'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
            'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
            'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
            'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54
        }
        
        # 特征提取器
        self.feature_extractor = MolecularFeatureExtractor()
    
    def __len__(self):
        return len(self.smiles_keys)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        smiles = self.smiles_keys[idx]
        
        with self.env.begin() as txn:
            try:
                data_bytes = txn.get(smiles.encode('ascii'))
                if data_bytes is None:
                    return None

                data = pickle.loads(data_bytes)
                
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return None

                # 化学合理性修正
                try:
                    Chem.SanitizeMol(mol)
                except Exception as e:
                    logger.warning(f"分子化学合理性检查失败，将跳过该样本。SMILES: {smiles}, 错误: {e}")
                    return None

                # 从SMILES重新生成分子图特征和3D构象
                graph_data = self.feature_extractor.mol_to_graph(mol)
                if graph_data is None:
                    return None
                
                atom_features = graph_data['atom_features']
                bond_features = graph_data['bond_features']
                edge_index = graph_data['edge_index']
                coords = graph_data['coordinates']

                # 限制原子数量
                if len(atom_features) > self.max_atoms:
                    atom_features = atom_features[:self.max_atoms]
                    coords = coords[:self.max_atoms]
                    valid_edges = (edge_index[0] < self.max_atoms) & (edge_index[1] < self.max_atoms)
                    edge_index = edge_index[:, valid_edges]
                    bond_features = bond_features[valid_edges]

                # 提取标签
                classification_labels = np.array(data.get('classification_target', [0]*26), dtype=np.float32)
                regression_labels = np.array(data.get('regression_target', [0]*5), dtype=np.float32)
                
                # 确保标签长度正确
                if len(classification_labels) < 26:
                    classification_labels = np.pad(classification_labels, (0, 26 - len(classification_labels)))
                elif len(classification_labels) > 26:
                    classification_labels = classification_labels[:26]
                
                if len(regression_labels) < 5:
                    regression_labels = np.pad(regression_labels, (0, 5 - len(regression_labels)))
                elif len(regression_labels) > 5:
                    regression_labels = regression_labels[:5]
                
                # 处理缺失值的掩码
                cls_mask = (classification_labels != -10000)
                reg_mask = (regression_labels != -10000.0)
                
                # 将缺失值替换为0
                classification_labels = np.where(cls_mask, classification_labels, 0.0)
                regression_labels = np.where(reg_mask, regression_labels, 0.0)
                
                return {
                    'atom_features': torch.tensor(atom_features, dtype=torch.float32),
                    'bond_features': torch.tensor(bond_features, dtype=torch.float32),
                    'edge_index': torch.tensor(edge_index, dtype=torch.long),
                    'coordinates': torch.tensor(coords, dtype=torch.float32),
                    'classification_labels': torch.tensor(classification_labels, dtype=torch.float32),
                    'regression_labels': torch.tensor(regression_labels, dtype=torch.float32),
                    'classification_mask': torch.tensor(cls_mask.astype(np.float32), dtype=torch.bool),
                    'regression_mask': torch.tensor(reg_mask.astype(np.float32), dtype=torch.bool),
                    'smiles': smiles
                }
                if isinstance(coords_list, list) and len(coords_list) > 0:
                    coords = np.array(coords_list[0], dtype=np.float32)
                else:
                    # 如果没有坐标，生成随机坐标
                    coords = np.random.randn(len(atoms), 3).astype(np.float32)
                
                # 确保原子和坐标的一致性
                min_len = min(len(atoms), len(coords))
                atoms = atoms[:min_len]
                coords = coords[:min_len]
                
                # 如果原子太多则截断
                if len(atoms) > self.max_atoms:
                    atoms = atoms[:self.max_atoms]
                    coords = coords[:self.max_atoms]
                
                # 从SMILES重新生成分子图特征
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    graph_data = self.feature_extractor.mol_to_graph(mol)
                    if graph_data is not None:
                        # 使用真实的分子图特征
                        atom_features = graph_data['atom_features']
                        bond_features = graph_data['bond_features']
                        edge_index = graph_data['edge_index']
                        
                        # 限制原子数量
                        if len(atom_features) > self.max_atoms:
                            atom_features = atom_features[:self.max_atoms]
                            coords = coords[:self.max_atoms]
                            # 过滤边索引
                            valid_edges = (edge_index[0] < self.max_atoms) & (edge_index[1] < self.max_atoms)
                            edge_index = edge_index[:, valid_edges]
                            bond_features = bond_features[valid_edges]
                    else:
                        # 回退到简单特征
                        atom_features, bond_features, edge_index = self._create_simple_features(atoms)
                else:
                    # 回退到简单特征
                    atom_features, bond_features, edge_index = self._create_simple_features(atoms)
                
                # 提取标签
                classification_labels = np.array(data.get('classification_target', [0]*26), dtype=np.float32)
                regression_labels = np.array(data.get('regression_target', [0]*5), dtype=np.float32)
                
                # 确保标签长度正确
                if len(classification_labels) < 26:
                    classification_labels = np.pad(classification_labels, (0, 26 - len(classification_labels)))
                elif len(classification_labels) > 26:
                    classification_labels = classification_labels[:26]
                
                if len(regression_labels) < 5:
                    regression_labels = np.pad(regression_labels, (0, 5 - len(regression_labels)))
                elif len(regression_labels) > 5:
                    regression_labels = regression_labels[:5]
                
                # 处理缺失值的掩码
                cls_mask = (classification_labels != -10000)
                reg_mask = (regression_labels != -10000.0)
                
                # 将缺失值替换为0
                classification_labels = np.where(cls_mask, classification_labels, 0.0)
                regression_labels = np.where(reg_mask, regression_labels, 0.0)
                
                return {
                    'atom_features': torch.tensor(atom_features, dtype=torch.float32),
                    'bond_features': torch.tensor(bond_features, dtype=torch.float32),
                    'edge_index': torch.tensor(edge_index, dtype=torch.long),
                    'coordinates': torch.tensor(coords, dtype=torch.float32),
                    'classification_labels': torch.tensor(classification_labels, dtype=torch.float32),
                    'regression_labels': torch.tensor(regression_labels, dtype=torch.float32),
                    'classification_mask': torch.tensor(cls_mask.astype(np.float32), dtype=torch.bool),
                    'regression_mask': torch.tensor(reg_mask.astype(np.float32), dtype=torch.bool),
                    'smiles': smiles
                }
                
            except Exception as e:
                logger.warning(f"加载样本 {idx} ({smiles}) 时出错: {e}")
                return None
    
    def _create_simple_features(self, atoms):
        """创建简单的原子特征"""
        num_atoms = len(atoms)
        
        # 简单的原子特征：one-hot编码原子类型
        atom_features = np.zeros((num_atoms, 119), dtype=np.float32)
        for i, atom_num in enumerate(atoms):
            if 1 <= atom_num <= 118:
                atom_features[i, atom_num-1] = 1.0
        
        # 简单的键特征：全连接图
        edge_index = []
        bond_features = []
        for i in range(num_atoms):
            for j in range(i+1, num_atoms):
                edge_index.extend([[i, j], [j, i]])
                bond_features.extend([np.ones(12, dtype=np.float32), np.ones(12, dtype=np.float32)])
        
        if len(edge_index) == 0:
            edge_index = [[0, 0]]
            bond_features = [np.zeros(12, dtype=np.float32)]
        
        edge_index = np.array(edge_index).T
        bond_features = np.array(bond_features)
        
        return atom_features, bond_features, edge_index
    


class MolecularFeatureExtractor:
    """分子特征提取器"""
    
    def __init__(self):
        self.atom_features_dim = 119
        self.bond_features_dim = 12
        
    def get_atom_features(self, atom) -> np.ndarray:
        """提取原子特征"""
        features = []
        
        # 原子类型 (一共118种元素，加上未知)
        atomic_num = atom.GetAtomicNum()
        features.extend(self._one_hot(atomic_num, list(range(1, 119))))
        
        # 度 (连接的原子数)
        degree = atom.GetDegree()
        features.extend(self._one_hot(degree, [0, 1, 2, 3, 4, 5]))
        
        # 正式电荷
        formal_charge = atom.GetFormalCharge()
        features.extend(self._one_hot(formal_charge, [-1, 0, 1, 2]))
        
        # 杂化类型
        hybridization = atom.GetHybridization()
        features.extend(self._one_hot(hybridization, [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2
        ]))
        
        # 是否在环中
        features.append(int(atom.IsInRing()))
        
        # 是否为芳香原子
        features.append(int(atom.GetIsAromatic()))
        
        # 氢原子数量
        num_hs = atom.GetTotalNumHs()
        features.extend(self._one_hot(num_hs, [0, 1, 2, 3, 4]))
        
        # 补充到119维
        while len(features) < self.atom_features_dim:
            features.append(0)
        
        return np.array(features[:self.atom_features_dim], dtype=np.float32)
    
    def get_bond_features(self, bond) -> np.ndarray:
        """提取键特征"""
        features = []
        
        # 键类型
        bond_type = bond.GetBondType()
        features.extend(self._one_hot(bond_type, [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ]))
        
        # 是否共轭
        features.append(int(bond.GetIsConjugated()))
        
        # 是否在环中
        features.append(int(bond.IsInRing()))
        
        # 立体化学
        stereo = bond.GetStereo()
        features.extend(self._one_hot(stereo, [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE
        ]))
        
        # 补充到12维
        while len(features) < self.bond_features_dim:
            features.append(0)
        
        return np.array(features[:self.bond_features_dim], dtype=np.float32)
    
    def _one_hot(self, value, choices):
        """One-hot编码"""
        encoding = [0] * len(choices)
        if value in choices:
            encoding[choices.index(value)] = 1
        return encoding
    
    def mol_to_graph(self, mol) -> Dict[str, np.ndarray]:
        """将分子转换为图数据，并生成3D构象"""
        if mol is None:
            return None

        try:
            # 1. 加氢
            mol = Chem.AddHs(mol)
            
            # 2. 生成3D构象
            if AllChem.EmbedMolecule(mol, randomSeed=42) == -1:
                return None # 如果构象生成失败，则跳过该分子
            
            # 3. (可选) 力场优化 (增强鲁棒性)
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception as e:
                logger.warning(f"力场优化失败，将使用未优化的3D构象。错误: {e}")

            conformer = mol.GetConformer()
            coordinates = conformer.GetPositions()

        except Exception:
            return None

        # 原子特征
        atom_features = []
        for atom in mol.GetAtoms():
            atom_features.append(self.get_atom_features(atom))
        atom_features = np.array(atom_features)
        
        # 键特征和边索引
        bond_features = []
        edge_indices = []
        
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            bond_feat = self.get_bond_features(bond)
            
            edge_indices.extend([[begin_idx, end_idx], [end_idx, begin_idx]])
            bond_features.extend([bond_feat, bond_feat])
        
        if not edge_indices:
            edge_index = np.empty((2, 0), dtype=np.int64)
            bond_features = np.empty((0, self.bond_features_dim), dtype=np.float32)
        else:
            edge_index = np.array(edge_indices, dtype=np.int64).T
            bond_features = np.array(bond_features, dtype=np.float32)

        return {
            'atom_features': atom_features,
            'bond_features': bond_features,
            'edge_index': edge_index,
            'coordinates': coordinates,
            'num_atoms': mol.GetNumAtoms(),
            'num_bonds': len(bond_features)
        }


def collate_lmdb_batch(batch):
    """LMDB批次整理函数，会过滤掉None的样本"""
    # 过滤掉加载失败的样本 (返回None的)
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    # 收集所有数据
    atom_features_list = []
    bond_features_list = []
    edge_indices_list = []
    coordinates_list = []
    batch_indices = []
    
    classification_labels_list = []
    regression_labels_list = []
    classification_masks_list = []
    regression_masks_list = []
    smiles_list = []
    
    atom_offset = 0
    
    for i, sample in enumerate(batch):
        # 图数据
        atom_features_list.append(sample['atom_features'])
        bond_features_list.append(sample['bond_features'])
        coordinates_list.append(sample['coordinates'])
        
        # 调整边索引偏移
        edge_index = sample['edge_index'] + atom_offset
        edge_indices_list.append(edge_index)
        
        # 批次索引
        num_atoms = sample['atom_features'].shape[0]
        batch_indices.extend([i] * num_atoms)
        atom_offset += num_atoms
        
        # 标签
        classification_labels_list.append(sample['classification_labels'])
        regression_labels_list.append(sample['regression_labels'])
        classification_masks_list.append(sample['classification_mask'])
        regression_masks_list.append(sample['regression_mask'])
        smiles_list.append(sample['smiles'])
    
    # 合并数据
    batch_data = {
        'atom_features': torch.cat(atom_features_list, dim=0),
        'bond_features': torch.cat(bond_features_list, dim=0),
        'edge_index': torch.cat(edge_indices_list, dim=1),
        'coordinates': torch.cat(coordinates_list, dim=0),
        'batch': torch.tensor(batch_indices, dtype=torch.long),
        'classification_labels': torch.stack(classification_labels_list),
        'regression_labels': torch.stack(regression_labels_list),
        'classification_mask': torch.stack(classification_masks_list),
        'regression_mask': torch.stack(regression_masks_list),
        'smiles': smiles_list
    }
    
    return batch_data


def create_lmdb_dataloaders(data_dir: str, 
                           batch_size: int = 4,
                           max_atoms: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建LMDB数据加载器"""
    data_path = Path(data_dir)
    
    # 查找LMDB文件
    train_file = data_path / "train.lmdb"
    valid_file = data_path / "valid.lmdb"
    test_file = data_path / "test.lmdb"
    
    # 检查文件是否存在
    if not train_file.exists():
        raise FileNotFoundError(f"训练文件不存在: {train_file}")
    if not valid_file.exists():
        raise FileNotFoundError(f"验证文件不存在: {valid_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"测试文件不存在: {test_file}")
    
    logger.info(f"使用LMDB文件:")
    logger.info(f"  训练集: {train_file}")
    logger.info(f"  验证集: {valid_file}")
    logger.info(f"  测试集: {test_file}")
    
    # 创建数据集
    train_dataset = LMDBToxD4CDataset(str(train_file), max_atoms)
    valid_dataset = LMDBToxD4CDataset(str(valid_file), max_atoms)
    test_dataset = LMDBToxD4CDataset(str(test_file), max_atoms)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 避免多进程问题
        pin_memory=True,
        collate_fn=collate_lmdb_batch
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_lmdb_batch
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_lmdb_batch
    )
    
    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    # 测试LMDB数据加载器
    data_dir = "/mnt/backup3/toxscan/ToxScan/Toxd4c/Toxd4c/data/dataset"
    
    try:
        train_loader, valid_loader, test_loader = create_lmdb_dataloaders(data_dir, batch_size=2)
        
        print(f"训练集批次数: {len(train_loader)}")
        print(f"验证集批次数: {len(valid_loader)}")
        print(f"测试集批次数: {len(test_loader)}")
        
        # 测试一个批次
        for batch in train_loader:
            print("\n批次数据形状:")
            print(f"原子特征: {batch['atom_features'].shape}")
            print(f"键特征: {batch['bond_features'].shape}")
            print(f"边索引: {batch['edge_index'].shape}")
            print(f"坐标: {batch['coordinates'].shape}")
            print(f"批次索引: {batch['batch'].shape}")
            print(f"分类标签: {batch['classification_labels'].shape}")
            print(f"回归标签: {batch['regression_labels'].shape}")
            print(f"SMILES数量: {len(batch['smiles'])}")
            break
        
        print("\nLMDB数据加载器测试成功!")
        
    except Exception as e:
        print(f"LMDB数据加载器测试失败: {e}")