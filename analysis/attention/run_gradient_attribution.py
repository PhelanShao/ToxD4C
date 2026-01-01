#!/usr/bin/env python3
"""
ToxD4C 梯度归因可视化 - 使用梯度方法获取原子重要性
比注意力可视化更有效地解释模型预测

支持的方法:
1. 输入梯度 (Input Gradients)
2. 集成梯度 (Integrated Gradients)
3. 原子消融分析 (Atom Ablation)

Usage:
    python run_gradient_attribution.py
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

from rdkit import Chem
from rdkit.Chem import AllChem, Draw

sys.path.insert(0, str(Path(__file__).parent))

# ============ 配置 ============
XYZ_DIR = Path("outputs/testattention")
OUTPUT_BASE = Path("outputs/testattention")
CHECKPOINT = Path("outputs/random9_conformer_100_v2/best_model.pt")

# 关注的任务（用于可解释性分析）
FOCUS_TASKS = [
    'Carcinogenicity',
    'Ames Mutagenicity', 
    'Acute oral toxicity (LD50)',
    'NR-ER',  # 雌激素受体 - BPA相关
    'NR-AhR',  # 芳香烃受体 - 环境毒物相关
]

CID_SMILES_MAP = {
    "61972": "CC(C)(C1=CC(=C(C(=C1)Br)OCC=C)Br)C2=CC(=C(C(=C2)Br)OCC=C)Br",
    "6623": "CC(C)(C1=CC=C(C=C1)O)C2=CC=C(C=C2)O",
    "8268": "C1=CC(=CC=C1C(C2=CC=C(C=C2)Cl)(C(Cl)(Cl)Cl)O)Cl",
}

CID_NAMES = {
    "61972": "TBBPA-DAE (Brominated flame retardant)",
    "6623": "BPA (Bisphenol A, endocrine disruptor)",
    "8268": "Dicofol (Organochlorine pesticide)",
}

# ============ 工具函数 ============

def parse_xyz_file(xyz_path: Path) -> tuple:
    """解析XYZ文件"""
    with open(xyz_path, 'r') as f:
        lines = f.readlines()
    
    n_atoms = int(lines[0].strip())
    comment = lines[1].strip()
    
    atoms = []
    coords = []
    
    for i in range(2, 2 + n_atoms):
        parts = lines[i].split()
        atom_symbol = parts[0]
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        atoms.append(atom_symbol)
        coords.append([x, y, z])
    
    return atoms, np.array(coords), comment


def prepare_model_input(smiles: str, coordinates: np.ndarray, config: dict, device: str):
    """准备模型输入"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    
    num_atoms = mol.GetNumAtoms()
    
    # 原子特征
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum() / 100.0,
            atom.GetDegree() / 6.0,
            atom.GetFormalCharge() / 3.0,
            atom.GetNumRadicalElectrons() / 3.0,
            float(atom.GetHybridization()) / 6.0,
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            atom.GetTotalNumHs() / 4.0,
        ]
        while len(features) < config['atom_features_dim']:
            features.append(0.0)
        atom_features.append(features[:config['atom_features_dim']])
    
    atom_features = torch.tensor(atom_features, dtype=torch.float32, device=device)
    atom_features.requires_grad_(True)
    
    # 边信息
    edge_list = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_list.extend([[i, j], [j, i]])
    
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    
    # 坐标
    coords_tensor = torch.tensor(coordinates[:num_atoms], dtype=torch.float32, device=device)
    coords_tensor.requires_grad_(True)
    
    batch = torch.zeros(num_atoms, dtype=torch.long, device=device)
    
    data = {
        'atom_features': atom_features,
        'edge_index': edge_index,
        'coordinates': coords_tensor,
        'batch': batch
    }
    
    return data, mol


def compute_input_gradients(model, data, smiles: str, task_names: list, device: str, config: dict):
    """
    计算输入梯度 - 哪些原子特征对预测最重要
    """
    model.train()  # 需要train模式才能计算梯度

    # 确保需要梯度
    atom_features = data['atom_features'].clone().detach().requires_grad_(True)
    coords = data['coordinates'].clone().detach().requires_grad_(True)

    data_copy = {
        'atom_features': atom_features,
        'edge_index': data['edge_index'],
        'coordinates': coords,
        'batch': data['batch']
    }

    # 前向传播
    outputs = model.forward(data_copy, [smiles])

    # 获取预测
    predictions = outputs.get('predictions', {})
    cls_preds = predictions.get('classification', None)  # [1, 26]
    reg_preds = predictions.get('regression', None)  # [1, 5]

    # 任务列表
    cls_tasks = config.get('classification_tasks_list', [])
    reg_tasks = config.get('regression_tasks_list', [])

    # 收集各任务的梯度
    task_gradients = {}

    # 分类任务
    if cls_preds is not None:
        for idx, task_name in enumerate(cls_tasks):
            if task_name in task_names and idx < cls_preds.shape[1]:
                pred = cls_preds[0, idx]

                # 计算梯度
                model.zero_grad()
                if atom_features.grad is not None:
                    atom_features.grad.zero_()

                pred.backward(retain_graph=True)

                if atom_features.grad is not None:
                    grad = atom_features.grad.detach().cpu().numpy()
                    atom_importance = np.linalg.norm(grad, axis=1)
                    # 将logit转换为概率
                    prob = torch.sigmoid(pred).item()
                    task_gradients[task_name] = {
                        'importance': atom_importance,
                        'prediction': prob,
                        'logit': pred.item()
                    }

    # 回归任务
    if reg_preds is not None:
        for idx, task_name in enumerate(reg_tasks):
            if task_name in task_names and idx < reg_preds.shape[1]:
                pred = reg_preds[0, idx]

                model.zero_grad()
                if atom_features.grad is not None:
                    atom_features.grad.zero_()

                pred.backward(retain_graph=True)

                if atom_features.grad is not None:
                    grad = atom_features.grad.detach().cpu().numpy()
                    atom_importance = np.linalg.norm(grad, axis=1)
                    task_gradients[task_name] = {
                        'importance': atom_importance,
                        'prediction': pred.item(),
                        'logit': pred.item()
                    }

    model.eval()
    return task_gradients, outputs


def compute_integrated_gradients(model, data, smiles: str, task_name: str, 
                                  device: str, steps: int = 50):
    """
    集成梯度 - 更稳定的归因方法
    """
    model.eval()
    
    # 基线 (零输入)
    baseline_features = torch.zeros_like(data['atom_features'])
    
    # 插值路径
    scaled_inputs = []
    for alpha in np.linspace(0, 1, steps):
        scaled = baseline_features + alpha * (data['atom_features'] - baseline_features)
        scaled.requires_grad_(True)
        scaled_inputs.append(scaled)
    
    # 收集梯度
    gradients = []
    for scaled in scaled_inputs:
        data_copy = data.copy()
        data_copy['atom_features'] = scaled
        
        outputs = model.forward(data_copy, [smiles])
        
        if task_name in outputs:
            pred = outputs[task_name]
            if pred.numel() > 0:
                model.zero_grad()
                pred.sum().backward(retain_graph=True)
                
                if scaled.grad is not None:
                    gradients.append(scaled.grad.detach().cpu().numpy())
    
    if gradients:
        # 积分近似
        avg_gradients = np.mean(gradients, axis=0)
        # 乘以输入差异
        input_diff = (data['atom_features'] - baseline_features).detach().cpu().numpy()
        attributions = avg_gradients * input_diff
        # 原子重要性
        atom_importance = np.linalg.norm(attributions, axis=1)
        return atom_importance
    
    return None


def plot_atom_importance(mol: Chem.Mol, atom_importance: np.ndarray, 
                         task_name: str, prediction: float,
                         save_path: Path, molecule_name: str):
    """绘制原子重要性可视化"""
    
    # 归一化重要性
    if atom_importance.max() > 0:
        importance_norm = atom_importance / atom_importance.max()
    else:
        importance_norm = atom_importance
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图: 2D分子结构带原子着色
    ax = axes[0]
    try:
        # 使用RDKit绘制
        from rdkit.Chem import Draw
        from rdkit.Chem.Draw import rdMolDraw2D
        
        # 设置原子颜色
        atom_colors = {}
        cmap = cm.get_cmap('Reds')
        for i, imp in enumerate(importance_norm):
            rgba = cmap(imp)
            atom_colors[i] = rgba[:3]  # RGB
        
        # 绘制分子
        drawer = rdMolDraw2D.MolDraw2DCairo(500, 400)
        drawer.drawOptions().addAtomIndices = True
        drawer.DrawMolecule(mol, highlightAtoms=list(range(mol.GetNumAtoms())),
                           highlightAtomColors=atom_colors)
        drawer.FinishDrawing()
        
        # 转换为图像
        import io
        from PIL import Image
        png = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(png))
        ax.imshow(img)
        ax.axis('off')
    except Exception as e:
        # 备选方案
        ax.text(0.5, 0.5, f"Molecule structure\n(error: {e})", 
                ha='center', va='center', fontsize=12)
        ax.axis('off')
    
    pred_label = f"Pred: {prediction:.3f}"
    if prediction > 0.5:
        pred_label += " (Toxic/Positive)"
    else:
        pred_label += " (Non-toxic/Negative)"
    ax.set_title(f'{task_name}\n{pred_label}', fontsize=12, fontweight='bold')
    
    # 右图: 原子重要性柱状图
    ax = axes[1]
    
    # 获取原子标签
    atom_labels = []
    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        atom_labels.append(f"{symbol}{i}")
    
    # 按重要性排序
    sorted_idx = np.argsort(importance_norm)[::-1]
    top_n = min(15, len(sorted_idx))  # 显示前15个
    
    colors = [cmap(importance_norm[i]) for i in sorted_idx[:top_n]]
    bars = ax.barh(range(top_n), importance_norm[sorted_idx[:top_n]], color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([atom_labels[i] for i in sorted_idx[:top_n]])
    ax.set_xlabel('Normalized Importance', fontweight='bold')
    ax.set_title('Top Atoms by Importance', fontweight='bold')
    ax.invert_yaxis()
    
    # 添加色条
    sm = cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Importance', fontweight='bold')
    
    plt.suptitle(f'{molecule_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_multi_task_comparison(mol: Chem.Mol, task_gradients: dict,
                               save_path: Path, molecule_name: str, atoms: list):
    """绘制多任务原子重要性对比"""
    
    n_tasks = len(task_gradients)
    if n_tasks == 0:
        return
    
    fig, axes = plt.subplots(2, (n_tasks + 1) // 2, figsize=(6 * ((n_tasks + 1) // 2), 10))
    if n_tasks == 1:
        axes = np.array([[axes]])
    axes = axes.flatten()
    
    cmap = cm.get_cmap('RdYlBu_r')
    
    for idx, (task_name, data) in enumerate(task_gradients.items()):
        ax = axes[idx]
        importance = data['importance']
        pred = data['prediction']
        
        # 归一化
        if importance.max() > 0:
            imp_norm = importance / importance.max()
        else:
            imp_norm = importance
        
        # 获取原子标签 (使用重原子)
        num_atoms = len(importance)
        atom_labels = atoms[:num_atoms]
        
        # 热图
        imp_2d = imp_norm.reshape(-1, 1)
        im = ax.imshow(imp_2d, cmap=cmap, aspect='auto')
        
        ax.set_yticks(range(num_atoms))
        ax.set_yticklabels(atom_labels, fontsize=8)
        ax.set_xticks([])
        
        pred_str = f"Pred: {pred:.3f}"
        ax.set_title(f'{task_name}\n{pred_str}', fontsize=10, fontweight='bold')
        
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # 隐藏多余的子图
    for idx in range(n_tasks, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'{molecule_name}\nAtom Importance Across Tasks', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    print("=" * 60)
    print("ToxD4C 梯度归因可视化")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 加载模型
    print("\n加载模型...")
    from models.toxd4c import ToxD4C
    
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    model = ToxD4C(config, device=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()
    
    # 查找XYZ文件
    xyz_files = list(XYZ_DIR.glob("*.xyz"))
    print(f"\n找到 {len(xyz_files)} 个XYZ文件")
    
    for xyz_file in xyz_files:
        print(f"\n{'='*50}")
        print(f"处理: {xyz_file.name}")
        
        cid = xyz_file.stem.split("_")[-1]
        smiles = CID_SMILES_MAP.get(cid)
        molecule_name = CID_NAMES.get(cid, f"CID {cid}")
        
        if not smiles:
            print(f"  跳过: 无SMILES")
            continue
        
        output_dir = OUTPUT_BASE / f"CID_{cid}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 解析XYZ
        atoms, coords, _ = parse_xyz_file(xyz_file)
        print(f"  原子数: {len(atoms)}")
        print(f"  SMILES: {smiles}")
        
        # 准备输入
        data, mol = prepare_model_input(smiles, coords, config, device)
        if data is None:
            print(f"  跳过: 无法解析分子")
            continue
        
        # 计算梯度归因
        print(f"  计算梯度归因...")
        task_gradients, outputs = compute_input_gradients(
            model, data, smiles, FOCUS_TASKS, device, config
        )
        
        print(f"  找到 {len(task_gradients)} 个任务的预测")
        
        # 保存预测结果
        predictions = {}
        for task_name, task_data in task_gradients.items():
            predictions[task_name] = {
                'prediction': float(task_data['prediction']),
                'top_atoms': []
            }
            # 找出最重要的原子
            imp = task_data['importance']
            top_idx = np.argsort(imp)[::-1][:5]
            for i in top_idx:
                i_int = int(i)  # 转换为Python int
                if i_int < mol.GetNumAtoms():
                    atom = mol.GetAtomWithIdx(i_int)
                    predictions[task_name]['top_atoms'].append({
                        'index': i_int,
                        'symbol': atom.GetSymbol(),
                        'importance': float(imp[i])
                    })
        
        with open(output_dir / 'predictions.json', 'w') as f:
            json.dump(predictions, f, indent=2)
        
        # 绘制各任务的原子重要性
        for task_name, task_data in task_gradients.items():
            safe_name = task_name.replace(' ', '_').replace('(', '').replace(')', '')
            save_path = output_dir / f'importance_{safe_name}.png'
            plot_atom_importance(
                mol, task_data['importance'], task_name, task_data['prediction'],
                save_path, molecule_name
            )
            print(f"  保存: importance_{safe_name}.png")
        
        # 绘制多任务对比
        plot_multi_task_comparison(
            mol, task_gradients,
            output_dir / 'multi_task_importance.png',
            molecule_name, atoms
        )
        print(f"  保存: multi_task_importance.png")
        
        print(f"  完成!")
    
    print(f"\n{'='*60}")
    print("所有分子处理完成!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

