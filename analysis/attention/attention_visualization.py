"""
ToxD4C Attention Visualization Module

This module provides tools to extract and visualize attention weights from ToxD4C model,
similar to the Uni-Mol style attention visualization showing:
1. Atom pair distance matrix (log1p transformed)
2. Average attention weights across all heads
3. Individual attention head weights

Reference: Uni-Mol attention visualization approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import math


class AttentionExtractor:
    """Extract attention weights from ToxD4C model components."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.attention_weights = {}
        self.hooks = []
        
    def register_hooks(self):
        """Register forward hooks to capture attention weights."""
        
        def get_attention_hook(name):
            def hook(module, input, output):
                # For GraphAttentionLayer - we need to modify the forward to return attention
                if hasattr(module, 'attention_weights_cache'):
                    self.attention_weights[name] = module.attention_weights_cache.detach().cpu()
            return hook
        
        # Find and register hooks for attention layers
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                hook = module.register_forward_hook(get_attention_hook(name))
                self.hooks.append(hook)
                
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def clear_cache(self):
        """Clear cached attention weights."""
        self.attention_weights = {}


class GraphAttentionLayerWithVisualization(nn.Module):
    """
    Modified GraphAttentionLayer that stores attention weights for visualization.
    Drop-in replacement for the original GraphAttentionLayer.
    """
    
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        
        assert output_dim % num_heads == 0
        
        self.q_linear = nn.Linear(input_dim, output_dim)
        self.k_linear = nn.Linear(input_dim, output_dim)
        self.v_linear = nn.Linear(input_dim, output_dim)
        self.out_linear = nn.Linear(output_dim, output_dim)
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)
        
        # Cache for attention weights (for visualization)
        self.attention_weights_cache = None
        self.store_attention = False
        
    def forward(self,
                node_features: torch.Tensor,
                adjacency: torch.Tensor,
                edge_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, num_atoms, _ = node_features.shape
        
        Q = self.q_linear(node_features)
        K = self.k_linear(node_features)
        V = self.v_linear(node_features)
        
        Q = Q.view(batch_size, num_atoms, self.num_heads, self.head_dim)
        K = K.view(batch_size, num_atoms, self.num_heads, self.head_dim)
        V = V.view(batch_size, num_atoms, self.num_heads, self.head_dim)
        
        scores = torch.einsum('bihd,bjhd->bhij', Q, K) / math.sqrt(self.head_dim)
        
        if edge_features is not None:
            edge_embeds = self.edge_encoder(edge_features)
            edge_embeds = edge_embeds.view(batch_size, num_atoms, num_atoms, self.num_heads, self.head_dim)
            edge_bias = torch.einsum('bihd,bijhd->bhij', Q, edge_embeds)
            scores = scores + edge_bias
        
        adjacency_mask = adjacency.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        scores = scores.masked_fill(~adjacency_mask.bool(), float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        
        # Store attention weights for visualization
        if self.store_attention:
            self.attention_weights_cache = attention_weights.detach()
        
        attention_weights = self.dropout(attention_weights)
        
        out = torch.einsum('bhij,bjhd->bihd', attention_weights, V)
        out = out.contiguous().view(batch_size, num_atoms, self.output_dim)
        
        if self.input_dim == self.output_dim:
            out = out + node_features
        out = self.norm(out)
        
        return out
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return cached attention weights [batch, heads, atoms, atoms]."""
        return self.attention_weights_cache


def compute_atom_pair_distances(coordinates: np.ndarray) -> np.ndarray:
    """
    Compute pairwise distances between all atoms.
    
    Args:
        coordinates: Atom coordinates [num_atoms, 3]
        
    Returns:
        Distance matrix [num_atoms, num_atoms]
    """
    num_atoms = coordinates.shape[0]
    distances = np.zeros((num_atoms, num_atoms))
    
    for i in range(num_atoms):
        for j in range(num_atoms):
            diff = coordinates[i] - coordinates[j]
            distances[i, j] = np.sqrt(np.sum(diff ** 2))
            
    return distances


def get_atom_labels(mol: Chem.Mol) -> List[str]:
    """Get atom labels (element symbols) for a molecule."""
    return [atom.GetSymbol() for atom in mol.GetAtoms()]


def plot_attention_visualization(
    smiles: str,
    coordinates: np.ndarray,
    attention_weights: torch.Tensor,
    heads_to_show: List[int] = [0, 4, 7],
    figsize: Tuple[int, int] = (18, 12),
    save_path: Optional[str] = None
):
    """
    Create Uni-Mol style attention visualization.
    
    Args:
        smiles: SMILES string of the molecule
        coordinates: Atom 3D coordinates [num_atoms, 3]
        attention_weights: Attention weights [heads, atoms, atoms] or [batch, heads, atoms, atoms]
        heads_to_show: List of head indices to visualize
        figsize: Figure size
        save_path: Path to save the figure
    """
    # Parse molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    atom_labels = get_atom_labels(mol)
    num_atoms = len(atom_labels)
    
    # Handle batch dimension
    if attention_weights.dim() == 4:
        attention_weights = attention_weights[0]  # Take first sample
    
    attention_weights = attention_weights.numpy()
    num_heads = attention_weights.shape[0]
    
    # Compute distance matrix
    distances = compute_atom_pair_distances(coordinates)
    distances_log = np.log1p(distances)  # log(1+x) transformation
    
    # Compute average attention across all heads
    avg_attention = attention_weights.mean(axis=0)
    
    # Create figure with 2 rows
    num_plots = 2 + len(heads_to_show)  # distance + avg + individual heads
    ncols = min(3, num_plots)
    nrows = (num_plots + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # 1. Plot atom pair distance (log1p)
    ax = axes[0]
    im = ax.imshow(distances_log, cmap='hot_r', aspect='auto')
    ax.set_title('Atoms pair distance (log1p)', fontsize=12, fontweight='bold')
    ax.set_xticks(range(num_atoms))
    ax.set_yticks(range(num_atoms))
    ax.set_xticklabels(atom_labels, fontsize=6, rotation=90)
    ax.set_yticklabels(atom_labels, fontsize=6)
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # 2. Plot average attention
    ax = axes[1]
    im = ax.imshow(avg_attention, cmap='hot_r', aspect='auto', vmin=0, vmax=0.5)
    ax.set_title('Average attention weight of all attention heads', fontsize=10, fontweight='bold')
    ax.set_xticks(range(num_atoms))
    ax.set_yticks(range(num_atoms))
    ax.set_xticklabels(atom_labels, fontsize=6, rotation=90)
    ax.set_yticklabels(atom_labels, fontsize=6)
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # 3. Plot individual attention heads
    for idx, head_idx in enumerate(heads_to_show):
        if head_idx >= num_heads:
            print(f"Warning: head {head_idx} does not exist (max: {num_heads-1})")
            continue
            
        ax = axes[2 + idx]
        head_attention = attention_weights[head_idx]
        im = ax.imshow(head_attention, cmap='hot_r', aspect='auto', vmin=0, vmax=0.5)
        ax.set_title(f'Attention weight of head {head_idx}', fontsize=12, fontweight='bold')
        ax.set_xticks(range(num_atoms))
        ax.set_yticks(range(num_atoms))
        ax.set_xticklabels(atom_labels, fontsize=6, rotation=90)
        ax.set_yticklabels(atom_labels, fontsize=6)
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Hide unused axes
    for idx in range(2 + len(heads_to_show), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()
    return fig


def compute_attention_distance_correlation(
    attention_weights: np.ndarray,
    distances: np.ndarray
) -> Dict[str, float]:
    """
    Compute correlation between attention weights and 3D distances.
    
    Args:
        attention_weights: [heads, atoms, atoms]
        distances: [atoms, atoms]
        
    Returns:
        Dictionary with correlation statistics per head
    """
    from scipy.stats import pearsonr, spearmanr
    
    results = {}
    num_heads = attention_weights.shape[0]
    
    # Flatten matrices for correlation
    dist_flat = distances.flatten()
    
    for head_idx in range(num_heads):
        attn_flat = attention_weights[head_idx].flatten()
        
        # Remove inf/nan values
        valid_mask = np.isfinite(attn_flat) & np.isfinite(dist_flat)
        attn_valid = attn_flat[valid_mask]
        dist_valid = dist_flat[valid_mask]
        
        if len(attn_valid) > 2:
            pearson_r, pearson_p = pearsonr(attn_valid, dist_valid)
            spearman_r, spearman_p = spearmanr(attn_valid, dist_valid)
            
            results[f'head_{head_idx}'] = {
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p
            }
    
    # Average correlation
    avg_pearson = np.mean([v['pearson_r'] for v in results.values()])
    results['average'] = {'pearson_r': avg_pearson}
    
    return results


if __name__ == "__main__":
    # Demo usage
    print("ToxD4C Attention Visualization Module")
    print("=" * 50)
    print("\nUsage:")
    print("1. Load model and enable attention storage")
    print("2. Run forward pass on a molecule")
    print("3. Extract attention weights")
    print("4. Call plot_attention_visualization()")
    print("\nSee visualize_toxd4c_attention.py for complete example")

