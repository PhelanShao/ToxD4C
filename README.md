# ToxD4C

ToxD4C (Dual‑Driven Dynamic Deep Chemistry) is a unified deep learning framework for multi‑task molecular toxicity prediction. It combines graph neural networks, transformers, geometric reasoning, and handcrafted fingerprint features to deliver accurate and interpretable toxicity assessments across diverse endpoints.

This README highlights the design of the framework, how the components interact, and the main configuration levers that enable reproduction and ablation studies.

## Architecture at a Glance

ToxD4C follows a modular encoder–fusion–head design so that researchers can experiment with different molecular representations without rewriting the training loop.

### Inputs & shared preprocessing

- **Atom/bond graphs** – `MolecularFeatureExtractor` builds 119-d atom and 12-d bond descriptors from SMILES and packs them into LMDB entries as tensors (`preprocess_data.py`).
- **3D geometry (optional)** – RDKit generates conformers during preprocessing; coordinates are stored so the geometric encoder can run without on-the-fly embedding.
- **Hierarchical context (optional)** – multi-scale features are computed inside the hierarchical encoder rather than persisted, so no extra preprocessing step is required.
- **Fingerprints** – SMILES strings are kept alongside tensors; attention-pooled ECFP/MACCS/RDKit descriptors are calculated at runtime when the fingerprint branch is enabled.

### Encoder stack (implemented in `models/`)

- **GNN branch** – `GraphAttentionNetwork` is the default (`models/architectures/gnn_transformer_hybrid.py`); switching `--gnn_backbone pyg_gcn_stack` swaps in the residual PyG `GCNStack`.
- **Transformer branch** – `MolecularTransformer` consumes dense node features with positional encodings to capture longer-range dependencies.
- **Geometric enrichment (optional)** – `GeometricEncoder` refines atom embeddings with distance-aware message passing when coordinates are provided.
- **Hierarchical branch (optional)** – `HierarchicalEncoder` runs staged GCN blocks and pools fragment-level signals back into a graph descriptor.
- **Fingerprint encoder** – `MolecularFingerprintModule` computes multiple fingerprints and fuses them with attention weighting.

### Fusion & heads

- **Hybrid fusion** – When `use_dynamic_fusion` is true, `DynamicFusionModule` applies cross-attention between GNN and Transformer streams; disabling it or setting `--fusion_method concatenation` falls back to linear fusion.
- **Auxiliary aggregation** – Hierarchical and fingerprint representations (when enabled) are concatenated with the fused hybrid output and projected through a shared fusion MLP.
- **Prediction heads** – `MultiScalePredictionHead` exposes both classification and regression branches with configurable task masks, and a supervised contrastive projection can be activated via `use_contrastive_learning`.

### Component-to-code reference

| Subsystem | Main implementation | Key toggles |
|-----------|---------------------|-------------|
| Atom embed + hybrid encoder | `models/toxd4c.py`, `models/architectures/gnn_transformer_hybrid.py` | `--disable_gnn`, `--disable_transformer`, `--gnn_backbone`, `--disable_dynamic_fusion`, `--fusion_method` |
| Geometric encoder | `models/encoders/geometric_encoder.py` | `--disable_geometric` |
| Hierarchical encoder | `models/encoders/hierarchical_encoder.py` | `--disable_hierarchical` |
| Fingerprint module | `models/fingerprints/molecular_fingerprint_enhanced.py` | `--disable_fingerprint` |
| Prediction head & contrastive loss | `models/heads/multi_scale_prediction_head.py`, `models/losses/contrastive_loss.py` | `--disable_classification`, `--disable_regression`, `--disable_contrastive` |

### High-level data flow

```
SMILES / Graph / 3D coords / Fingerprints
           │           │           │
           │           │           └──► Fingerprint Module (ECFP/MACCS/etc., attention fusion)
           │           │
           │           └──► Geometric Encoder (optional)
           │
           └──► Hybrid Main Encoder
                 ├─ GNN Branch (GraphAttention or PyG GCN stack)
                 ├─ Transformer Branch
                 └─ Dynamic Fusion (cross-attention) or Concatenation

          ──► Fused Graph Representation ──► Multi-task Head (Cls + Reg)
                                         └─► (optional) SupCon representation for contrastive learning
```

### Component configuration

- Swap the GNN backbone via `--gnn_backbone {default,pyg_gcn_stack}` and control depth with `--gcn_stack_layers`.
- Enable or disable auxiliary branches independently: `--disable_geometric`, `--disable_hierarchical`, `--disable_fingerprint`.
- Choose the fusion strategy (`--disable_dynamic_fusion`, `--fusion_method concatenation`).
- Route tasks by turning classification or regression heads on/off, or selecting specific task indices for per-endpoint studies.

## Why This Architecture?

- **Complementary representations** – Pairing topology-aware GNNs with sequence-aware transformers captures both local chemical environments and global substructure motifs.
- **Multi-view enrichment** – Optional geometric and hierarchical branches incorporate 3D conformational cues and fragment-level insights when available, yet degrade gracefully when disabled.
- **Descriptor alignment** – Attention-based fingerprint pooling integrates handcrafted descriptors without overwhelming the learned representation.
- **Contrastive regularisation** – The SupCon objective encourages discriminative, well-separated embeddings that benefit both classification and regression heads.
- **Reproducible experimentation** – Clear toggles for each component make it straightforward to run ablations, compare modalities, and reproduce published baselines.

## Quick Start
1) Prepare LMDB data with splits under a data directory (default below):
   - `train.lmdb`, `valid.lmdb`, `test.lmdb`
2) Train the full model (with contrastive learning):

```bash
python ToxD4C/train.py \
  --experiment_name "toxd4c_full_model" \
  --data_dir data/data/processed \
  --seed 42 --num_epochs 50 --batch_size 16 --deterministic
```

Outputs for each run are saved under `experiments/<name>_<timestamp>/`:
- `train.log`: run‑specific logs
- `checkpoints/<name>_best.pth`: best checkpoint
- `checkpoints/<name>_results.json`: summary metrics + config snapshot

## Key Flags
- `--disable_contrastive`: disable supervised contrastive learning (default is enabled)
- `--disable_gnn | --disable_transformer | --disable_geometric | --disable_hierarchical | --disable_fingerprint`: ablate components
- `--disable_dynamic_fusion` or `--fusion_method concatenation`: use concatenation instead of cross‑attention fusion
- `--gnn_backbone {default,pyg_gcn_stack}` and `--gcn_stack_layers N` (2–4): choose GNN backbone
- `--use_preprocessed` is enabled by default; preprocessed LMDB under `--preprocessed_dir` is used when present

### GNN backbone variants
- default (GraphAttentionNetwork): multi‑head attention message passing with configurable depth.
- pyg_gcn_stack (GCNStack): residual stack of PyG `GCNConv` layers with LayerNorm and dropout.
  - Recommended `--gcn_stack_layers` in [2, 4].
  - Works both in hybrid (with Transformer) and in GNN‑only ablations.

Example (GCN stack, hybrid):

```bash
python ToxD4C/train.py \
  --experiment_name "toxd4c_hybrid_gcnstack" \
  --gnn_backbone pyg_gcn_stack --gcn_stack_layers 3 \
  --seed 42 --num_epochs 50 --batch_size 16 --deterministic
```

## Common Ablations
All runs share the same base args as the full model; only the ablation flags are shown below.

- GNN only (baseline):
  `--disable_transformer --disable_geometric --disable_hierarchical --disable_fingerprint`
- GNN + Transformer (core):
  `--disable_geometric --disable_hierarchical --disable_fingerprint`
- GNN + Transformer + 3D:
  `--disable_hierarchical --disable_fingerprint`
- GNN + Transformer + Fingerprint:
  `--disable_geometric --disable_hierarchical`
- Full − Fingerprint:
  `--disable_fingerprint`
- Full − Contrastive:
  `--disable_contrastive`
- Concatenation Fusion:
  `--disable_dynamic_fusion`
- Classification only / Regression only:
  `--disable_regression` / `--disable_classification`

## Data & Preprocessing

### Preparing LMDB datasets

1. **Raw LMDB** – Each key corresponds to a SMILES string with classification/regression targets.
2. **Preprocessing (`preprocess_data.py`)**
   - Builds atom/bond feature tensors and RDKit 3D conformers via `MolecularFeatureExtractor`.
   - Stores tensors, label arrays, and the originating SMILES string directly in a new LMDB so training never has to regenerate conformers.
   - Enforces `--max_atoms` by skipping over-length molecules to keep tensor shapes consistent.
3. **Output layout** – The processed LMDB retains the original keys and adds:
   - `atom_features` `[num_atoms, 119]`
   - `bond_features` `[num_edges, 12]`
   - `edge_index` `[2, num_edges]`
   - `coordinates` `[num_atoms, 3]`
   - `classification_target`, `regression_target`, and SMILES.

### Runtime data loading (`data/lmdb_dataset.py`)

- `LMDBToxD4CDataset` reads the tensors, truncates to `--max_atoms`, builds boolean masks for missing labels (value `-10000`), and returns SMILES for fingerprinting.
- `collate_lmdb_batch` concatenates node-level tensors, constructs a global `batch` index vector for PyG pooling, and stacks per-task labels/masks.
- `create_lmdb_dataloaders` wires the datasets into PyTorch dataloaders with optional shuffling and exposes all three splits.

### Training-time controls

- `--use_preprocessed` (default) loads from `--preprocessed_dir`; `--force_preprocess` triggers preprocessing if cached tensors are absent.
- `--max_atoms` must match the value used during preprocessing to avoid silently truncating nodes at load time.
- The preprocessing script logs counts of processed vs. skipped molecules, which is useful when verifying data coverage.

## Reproducibility & Splits
- Determinism: `--seed` and `--deterministic` set consistent training behavior and snapshot full run metadata.
- Splits: random / scaffold / cluster do not overlap by design; see `utils/splitter.py` for diagnostics.

## Notes on Recent Changes
- Contrastive learning is now part of the training objective, weighted by `config['contrastive_weight']` (default 0.3). To disable use `--disable_contrastive`.
- Logs are no longer written to a global file. Each run now logs to `experiments/<name>_<timestamp>/train.log`.

## Re‑running Old Experiments
Experiments trained before this change did not include the contrastive loss in the objective even if enabled in the config. For fair comparisons, re‑run experiments where `use_contrastive_learning: true` appears in the saved config.

Typical experiments to re‑run:
- `toxd4c_ablation_gnn_only`
- `toxd4c_ablation_gnn_transformer`
- `toxd4c_ablation_gnn_trans_3d`
- `toxd4c_ablation_gnn_trans_fp`
- `toxd4c_ablation_full_model` (ensure NOT passing `--disable_contrastive`)
- `toxd4c_ablation_full_no_fp`
- `toxd4c_ablation_concat_fusion`
- `toxd4c_ablation_classification_only`
- `toxd4c_ablation_regression_only`

Additionally, earlier `full_hybrid_gcnstack` runs that failed during forward have been fixed and should be re‑run.

## Inference
See `ToxD4C/inference_toxd4c.py` for batch inference on SMILES files using a trained checkpoint.

## License and Contributions
MIT‑licensed. Issues and PRs are welcome.
