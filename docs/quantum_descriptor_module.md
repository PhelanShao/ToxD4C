# Quantum Descriptor Module

## Overview

The **Quantum Descriptor Module** is an optional component of ToxD4C that enables integration of pre-computed quantum chemical descriptors into the molecular toxicity prediction pipeline.

This module implements an **Adaptive Feature Gating with Hierarchical Decay** mechanism that:
- Automatically learns which quantum descriptors are most important for the prediction task
- Progressively suppresses irrelevant features through multiple decay layers
- Provides interpretable feature importance scores

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                Input Descriptors                         │
│              [batch, num_descriptors]                    │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Input Projection                            │
│         Linear → LayerNorm → GELU → Dropout             │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│           Feature Gating Layer                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │  gate = sigmoid(gate_net(x)/τ) × sigmoid(base)  │    │
│  │  output = x × gate                              │    │
│  │  sparsity_loss = λ × mean(gate)                 │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│        Hierarchical Decay Blocks (×N layers)            │
│  ┌─────────────────────────────────────────────────┐    │
│  │  decay_factor = max(1 - rate × layer_idx, 0.3)  │    │
│  │  h = transform(x)                               │    │
│  │  gate = sigmoid(gate_net(x)) × decay_factor     │    │
│  │  output = gate × h + residual_scale × x         │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Output Projection                           │
│              Linear → LayerNorm                          │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│         Fusion with Graph Representation (Optional)     │
│  ┌─────────────────────────────────────────────────┐    │
│  │  fusion_weight = sigmoid(W × [graph; desc])     │    │
│  │  fused = weight × desc + (1-weight) × graph     │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Feature Gating Layer

The feature gating mechanism learns importance weights for each feature dimension:

- **Dynamic Gate**: Input-dependent gating computed via a neural network
- **Static Gate**: Learned base gate values (task-specific prior)
- **Temperature Control**: Lower temperature → sharper gating decisions
- **Sparsity Regularization**: Encourages sparse feature selection

```python
# Gate computation
gate_logits = gate_net(x)                    # Dynamic gate
dynamic_gate = sigmoid(gate_logits / τ)      # Temperature-controlled
static_gate = sigmoid(base_gate)             # Learned prior
gate = dynamic_gate × static_gate            # Combined gate
output = x × gate                            # Gated features
```

### 2. Hierarchical Decay Blocks

Multiple decay layers progressively filter out noise:

- **Decay Factor**: Decreases with layer depth (deeper = more decay)
- **Minimum Retention**: At least 30% of information is always preserved
- **Residual Connection**: Ensures gradient flow and stable training

```python
# Decay computation
decay_factor = max(1.0 - decay_rate × layer_idx, 0.3)
h = transform(x)
gate = decay_gate(x) × decay_factor
output = gate × h + residual_scale × x
```

## Usage

### Configuration

Add these options to your ToxD4C config:

```python
config = {
    # Enable quantum descriptors
    'use_quantum_descriptors': True,
    
    # Descriptor settings
    'num_quantum_descriptors': 69,     # Number of input descriptors
    'quantum_descriptor_dim': 256,      # Output dimension
    
    # Decay settings
    'quantum_decay_layers': 4,          # Number of decay layers
    'quantum_decay_rate': 0.1,          # Decay rate per layer
    
    # Gating settings
    'quantum_use_gating': True,         # Enable feature gating
    'quantum_sparsity_weight': 0.01,    # Sparsity loss weight
}
```

### Data Preparation

Include quantum descriptors in your data dict:

```python
data = {
    'atom_features': atom_features,
    'edge_index': edge_index,
    'coordinates': coordinates,
    'quantum_descriptors': torch.tensor([
        # HOMO, LUMO, gap, dipole, Mulliken charges, etc.
        [homo, lumo, gap, dipole_x, dipole_y, dipole_z, ...],
        # ... for each molecule in batch
    ]),  # Shape: [batch_size, num_descriptors]
}
```

### Standalone Usage

```python
from models.encoders.quantum_descriptor_module import QuantumDescriptorModule

# Initialize
module = QuantumDescriptorModule(
    num_descriptors=69,
    hidden_dim=256,
    output_dim=256,
    graph_repr_dim=512,  # Dimension of graph representation (for fusion)
    num_decay_layers=4,
    use_gating=True,
)

# Forward pass
descriptors = torch.randn(32, 69)  # Batch of 32
graph_repr = torch.randn(32, 256)  # Optional

result = module(descriptors, graph_repr)

# Access outputs
encoded = result['descriptor_repr']      # [32, 256]
fused = result['fused_repr']             # [32, 256] (if graph_repr provided)
gates = result['gate_values']            # [32, 256] (feature importance)
sparsity = result['sparsity_loss']       # Scalar (add to training loss)
```

### Interpretability

After training, analyze which features are most important:

```python
# Get learned gate values
gate_values = module.get_gate_values_mean()  # [hidden_dim]

# Visualize feature importance
import matplotlib.pyplot as plt
plt.bar(range(len(gate_values)), gate_values.detach().cpu().numpy())
plt.xlabel('Feature Index')
plt.ylabel('Importance (Gate Value)')
plt.title('Learned Feature Importance')
```

## Supported Descriptors

The module supports any numerical descriptors. Common quantum descriptors include:

| Category | Descriptors |
|----------|-------------|
| Orbital Energies | HOMO, LUMO, HOMO-1, LUMO+1, gap |
| Thermodynamics | Gibbs free energy, electronic energy, zero-point energy |
| Dipole Moment | μx, μy, μz, |μ| |
| Atomic Charges | Mulliken charges (per atom) |
| Reactivity | Fukui functions, electrophilicity index |

## Training Tips

1. **Start with gating enabled**: The gating mechanism helps identify irrelevant features
2. **Tune sparsity weight**: Higher values → sparser gates → more feature selection
3. **Monitor gate values**: If all gates converge to 0 or 1, adjust temperature
4. **Add sparsity loss**: Include `quantum_sparsity_loss` in your total loss

