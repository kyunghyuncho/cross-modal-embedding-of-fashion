# Pedagogical Cross-Modal Fashion Retrieval

This repository presents a streamlined, pedagogically sound implementation of a cross-modal retrieval system, serving as a functional demonstration of mapping images and short text descriptions into a shared latent semantic space $\mathbb{R}^d$.

## Architectural Overview

We leverage pre-trained foundation models to extract representations from both modalities:
* **Vision Encoder:** `facebook/dinov2-small`, yielding $d_{image} = 384$ dimensional vectors.
* **Text Encoder:** `nomic-ai/nomic-embed-text-v1.5`, yielding $d_{text} = 768$ dimensional vectors. 

The embeddings are subsequently projected via two learnable linear transformations parameterizing $W_{image}$ and $W_{text}$ onto a shared joint embedding dimensionality $d_{joint} = 512$:

$$z_{image} = \frac{W_{image} x_{image} + b_{image}}{\|W_{image} x_{image} + b_{image}\|_2}$$
$$z_{text} = \frac{W_{text} x_{text} + b_{text}}{\|W_{text} x_{text} + b_{text}\|_2}$$

The objective formulation utilizes a symmetric contrastive loss driven by the pairwise cosine similarity scaled by a learnable temperature scalar $\tau$:
$$L = \frac{1}{2}\left(L_{image \to text} + L_{text \to image}\right)$$

## Data Engineering and Hardware Agnosticism

The codebase aggressively separates representation extraction from model training. 
1. **Precomputation (`precompute.py`)**: Extracted embeddings are saved as dense PyTorch tensors (`.pt`). This mitigates the memory footprint and computation required during iterative training, permitting execution on constrained local hardware including Apple Silicon GPUs (`mps`).
2. **Training (`train.py`)**: A `LightningModule` strictly orchestrates gradient flow through the final projection matrices.

## Execution and Usage

### 1. Environment Instantiation 
This repository relies strictly on `uv` for reproducible and fast environment instantiation.
```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt --exclude-newer "$EXCLUDE_DATE"
```

### 2. Feature Extraction
For rapid qualitative iteration or debugging, prefix with `DEV_MODE=1` to enforce subsampling ($N=1000$).
```bash
python precompute.py
```

### 3. Contrastive Training 
The underlying PyTorch Lightning `Trainer` will automatically utilize the available accelerators (e.g., Apple `mps` or NVIDIA `cuda`).
```bash
python train.py
```

### 4. Interactive Diagnostics 
To engage with the multi-dimensional feature space interactively, we support an Exploratory Data Analysis module equipped with `FAISS` capabilities.
```bash
streamlit run app.py
```