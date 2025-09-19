"""
ToxD4C 推理脚本
支持31个毒性预测任务的分子毒性预测

作者: AI助手
日期: 2024-06-11
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import sys
from pathlib import Path
from rdkit import Chem
import warnings
from torch.utils.data import Dataset

# 添加项目路径
sys.path.append(str(Path(__file__).parent))
import argparse
from configs.toxd4c_config import (
    get_enhanced_toxd4c_config, CLASSIFICATION_TASKS, REGRESSION_TASKS
)
from data.lmdb_dataset import create_lmdb_dataloaders, MolecularFeatureExtractor, collate_lmdb_batch
from models.toxd4c import ToxD4C

warnings.filterwarnings('ignore')


class ToxD4CPredictor:
    """ToxD4C 预测器"""
    
    def __init__(self, model_path: str, config: Dict, device: str = 'cpu'):
        self.device = device
        self.config = config
        
        # 创建模型
        self.model = ToxD4C(config=self.config, device=device).to(device)
        
        # 加载模型权重
        self.load_model(model_path)
        self.model.eval()
        
        print(f"ToxD4C 预测器已加载")
        print(f"设备: {device}")
        print(f"模型: {model_path}")
    
    def load_model(self, model_path: str):
        """加载模型权重"""
        if not Path(model_path).exists():
            print(f"错误: 模型文件不存在 {model_path}")
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"成功加载模型: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {str(e)}")
            raise e
    
    def predict_on_loader(self, dataloader: torch.utils.data.DataLoader) -> pd.DataFrame:
        """在数据加载器上进行批量预测"""
        all_smiles = []
        all_cls_preds = []
        all_reg_preds = []

        with torch.no_grad():
            for batch in dataloader:
                if batch is None:
                    continue
                # 移动数据到设备
                data = {
                    'atom_features': batch['atom_features'].to(self.device),
                    'edge_index': batch['edge_index'].to(self.device),
                    'coordinates': batch['coordinates'].to(self.device),
                    'batch': batch['batch'].to(self.device)
                }
                smiles_list = batch['smiles']
                
                # 模型预测
                outputs = self.model(data, smiles_list)
                cls_preds = outputs['predictions']['classification']
                reg_preds = outputs['predictions']['regression']
                
                # 收集结果
                all_smiles.extend(smiles_list)
                all_cls_preds.append(torch.sigmoid(cls_preds).cpu().numpy())
                all_reg_preds.append(reg_preds.cpu().numpy())

        # 合并结果
        if not all_smiles:
            return pd.DataFrame()

        cls_preds_np = np.concatenate(all_cls_preds, axis=0)
        reg_preds_np = np.concatenate(all_reg_preds, axis=0)
        
        # 创建DataFrame
        results_df = pd.DataFrame({'SMILES': all_smiles})

        # 添加分类概率与标签 (prob 概率 + 二值化标签)
        for i, task_name in enumerate(CLASSIFICATION_TASKS):
            results_df[f"{task_name}_prob"] = cls_preds_np[:, i]
            results_df[task_name] = (cls_preds_np[:, i] > 0.5).astype(int)
            
        # 添加回归结果
        for i, task_name in enumerate(REGRESSION_TASKS):
            results_df[task_name] = reg_preds_np[:, i]
            
        return results_df
    
class SmilesDataset(Dataset):
    """从SMILES列表创建数据集
    兼容多列输入：默认取首列为 SMILES（制表符或空白分隔）。
    跳过空行与疑似表头（包含 'smiles'）。
    """
    def __init__(self, smiles_list: List[str]):
        cleaned = []
        for raw in smiles_list:
            line = raw.strip()
            if not line:
                continue
            low = line.lower()
            if 'smiles' in low and (low.startswith('smiles') or low.split()[0] == 'smiles'):
                continue
            parts = line.split('\t') if ('\t' in line) else line.split()
            if not parts:
                continue
            smiles = parts[0].strip()
            if smiles:
                cleaned.append(smiles)

        self.smiles_list = cleaned
        self.feature_extractor = MolecularFeatureExtractor()

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"警告: 无法解析SMILES: {smiles}")
            return None
        
        graph_data = self.feature_extractor.mol_to_graph(mol)
        if graph_data is None:
            print(f"警告: 无法为SMILES生成图特征: {smiles}")
            return None
            

        return {
            'atom_features': torch.tensor(graph_data['atom_features'], dtype=torch.float32),
            'bond_features': torch.tensor(graph_data['bond_features'], dtype=torch.float32),
            'edge_index': torch.tensor(graph_data['edge_index'], dtype=torch.long),
            'coordinates': torch.tensor(graph_data['coordinates'], dtype=torch.float32),
            'classification_labels': torch.zeros(len(CLASSIFICATION_TASKS)),
            'regression_labels': torch.zeros(len(REGRESSION_TASKS)),
            'classification_mask': torch.ones(len(CLASSIFICATION_TASKS), dtype=torch.bool),
            'regression_mask': torch.ones(len(REGRESSION_TASKS), dtype=torch.bool),
            'smiles': smiles
        }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='ToxD4C 推理脚本')
    parser.add_argument('--model_path', type=str,
                        default='checkpoints_real/toxd4c_real_best.pth',
                        help='训练好的模型路径')
    parser.add_argument('--smiles_file', type=str, default=None, help='包含SMILES字符串的输入文件路径')
    parser.add_argument('--data_dir', type=str, default='data/dataset', help='LMDB数据目录 (如果smiles_file未提供)')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--device', type=str, default=None, choices=['cpu', 'cuda'], help='推理设备(cpu/cuda)')
    parser.add_argument('--output_file', type=str, default='inference_results.csv', help='输出CSV文件路径')
    
    args = parser.parse_args()

    print("=== ToxD4C 推理 ===")
    
    # 设置设备
    if args.device is not None:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 使用安全配置进行推理
    config = get_enhanced_toxd4c_config()
    
    # 创建数据加载器
    if args.smiles_file:
        print(f"从SMILES文件加载数据: {args.smiles_file}")
        try:
            with open(args.smiles_file, 'r') as f:
                smiles_list = f.readlines()
            
            dataset = SmilesDataset(smiles_list)
            test_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_lmdb_batch
            )
            print(f"成功加载 {len(dataset)} 个SMILES")
        except Exception as e:
            print(f"从SMILES文件加载数据失败: {e}")
            return
    else:
        print(f"从LMDB目录加载数据: {args.data_dir}")
        try:
            _, _, test_loader = create_lmdb_dataloaders(
                args.data_dir,
                batch_size=args.batch_size
            )
            print(f"成功加载测试数据: {args.data_dir}")
            print(f"测试集批次数: {len(test_loader)}")
        except Exception as e:
            print(f"加载测试数据失败: {e}")
            return
        
    # 创建预测器
    try:
        predictor = ToxD4CPredictor(
            model_path=args.model_path,
            config=config,
            device=device
        )
    except Exception as e:
        print(f"创建预测器失败: {e}")
        return

    # 进行预测
    print("开始预测...")
    results_df = predictor.predict_on_loader(test_loader)
    
    # 保存结果
    if not results_df.empty:
        results_df.to_csv(args.output_file, index=False)
        print(f"预测结果已保存到: {args.output_file}")
        
        # 打印前几行结果
        print("\n预测结果预览:")
        print(results_df.head())
    else:
        print("没有生成任何预测结果。")

if __name__ == "__main__":
    main()
