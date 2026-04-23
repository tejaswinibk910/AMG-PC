# AMG-PC: Adaptive Modality-Gated Point Cloud Completion


---

## Overview

AMG-PC extends [IAET (IJCAI 2025)](https://github.com/doldolOuO/IAET) with per-stage adaptive modality gating for multimodal point cloud completion. Given a partial point cloud, an RGB image, and a BLIP-2 generated text description, AMG-PC learns to weight each modality differently at each decoder stage.

**Key contribution:** A lightweight ModalityGate (~200 params) at each of 3 decoder stages outputs softmax weights [α_pc, α_img, α_txt] trained purely from Chamfer Distance loss — no auxiliary supervision. The model discovers on its own to use image for global structure and text for fine detail.

---

## Results

| Method | Venue | Avg CD ↓ |
|--------|-------|----------|
| XMFNet | NeurIPS 2022 | 1.443 |
| EGIINet | ECCV 2024 | 1.211 |
| IAET | IJCAI 2025 | 1.090 |
| HGACNet | arXiv 2025 | ~1.006 |
| PGNet | arXiv 2025 | 0.927 |
| **AMG-PC (ours)** | CSC 449 2026 | **0.976** |

> AMG-PC trained on 4 categories (airplane, car, chair, watercraft). Published methods report 8-category average.

### Ablation (Car, Batch 32, 100 epochs)

| Model | CD ↓ | vs IAET |
|-------|------|---------|
| w/o image | 4.730 | +95% |
| IAET baseline | 2.428 | — |
| w/o text | 2.360 | −2.8% |
| uniform gate | 2.035 | −16.2% |
| **AMG-PC adaptive** | **2.007** | **−17.3%** |

### Zero-Shot (5 unseen categories)

| Category | CD ↓ |
|----------|------|
| Bench | 0.799 |
| Firearm | 1.451 |
| Cellphone | 1.529 |
| Speaker | 1.990 |
| Monitor | 2.471 |

---

## Architecture

```
Partial PC  → PointNet++ SA  → pc_token  (B, 256, 256) ─┐
RGB Image   → ResNet-18      → im_token  (B, 256, 49)  ──┤→ InterlacedTransformer → GatedDecoder → 2048 pts
Text (CLIP) → Linear+LN      → txt_token (B, 256, 1)  ──┘
```

**GatedDecoder:** CoarseDecoder → 3× ViewGuidedUpLayer, each preceded by a ModalityGate:
```
α = softmax(MLP(mean_pool(fused_pc_features)))  →  [α_pc, α_img, α_txt]
gated_im = α_img × fused_im_token + α_txt × txt_token
```

**Observed gate behavior (epoch 99):**
- Stage 1 (256→512): IMG=0.94, PC=0.06, TXT≈0.00
- Stage 2 (512→1024): TXT=0.39, PC=0.31, IMG=0.30
- Stage 3 (1024→2048): TXT=0.42, PC=0.29, IMG=0.29

---

## Setup

```bash
# Clone and install dependencies
git clone https://github.com/tejaswinibk910/AMG-PC.git
cd AMG-PC

pip install torch torchvision numpy Pillow transformers

# Build CUDA extensions
cd cuda/ChamferDistance && python setup.py install
cd ../pointnet2_ops_lib && python setup.py install

# Generate text embeddings (one-time, requires BLIP-2)
bash preprocess_text.sh
```

---

## Training

```bash
# Full AMG-PC adaptive
python train_amgpc.py

# Ablations
python train_amgpc_uniform.py   # fixed 1/3 gate weights
python train_amgpc_notext.py    # PC + image only
python train_amgpc_noimage.py   # PC + text only
```

Edit `config_vipc.py` to set `data_root`, `cat`, `batch_size`, and `n_epochs`.

---

## Visualization

```bash
# 4-category interactive viz (best 2 per category by CD)
python viz_amgpc_4cat.py
# output: /tmp/amgpc_viz.html

# Zero-shot generalization viz
python viz_zeroshot.py
# output: /tmp/amgpc_zeroshot.html
```

Open the HTML files in any browser. Drag to rotate, scroll to zoom. All point clouds rotate synchronously.

---

## Acknowledgements

Built on [IAET](https://github.com/doldolOuO/IAET) (IJCAI 2025). Dataset: [ShapeNet-ViPC](https://github.com/Hydrogenion/ViPC). Text embeddings generated with [BLIP-2](https://github.com/salesforce/LAVIS) and [CLIP](https://github.com/openai/CLIP).
