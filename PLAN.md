# PLAN.md: Cross-Modal Retrieval Web Application

## 1. Project Objectives and Pedagogical Goals
The objective is to implement a lightweight, hardware-agnostic cross-modal retrieval application (Text-to-Image and Image-to-Text) using PyTorch, PyTorch Lightning, and Streamlit. 

Pedagogical emphasis is placed on:
* Rigorous data partitioning (Train / Validation / Test).
* Representation learning via a symmetric contrastive objective.
* Hardware abstraction leveraging PyTorch Lightning (seamless transition between Apple Silicon MPS and NVIDIA CUDA).
* Hyperparameter optimization guided by validation metrics.
* Interactive Exploratory Data Analysis (EDA) of multi-modal datasets.

## 2. Architecture Specification
* **Image Encoder:** `facebook/dinov2-vits14` (frozen).
* **Text Encoder:** `nomic-ai/nomic-embed-text-v1.5` (frozen). Note: This model requires the prefix `search_document:` for database descriptions and `search_query:` for user inputs.
* **Projection Layers:** Two un-frozen linear layers mapping encoder outputs to a shared latent space dimension $d_{joint}$.
* **Objective Function:** Symmetric cross-entropy with a learnable temperature scalar $\tau$.

## 3. Phased Implementation Plan

### Phase 1: Environment and Dependency Configuration
* Define a `requirements.txt` or `pyproject.toml` utilizing `uv` for dependency management.
* Core dependencies: `torch`, `torchvision`, `pytorch-lightning`, `streamlit`, `transformers`, `datasets`, `pandas`, `scikit-learn`, `faiss-cpu`, `altair` or `plotly` (for EDA visualizations).
* Configure the local environment for Apple Silicon (`device="mps"`).

### Phase 2: Data Ingestion and Feature Precomputation
This phase strictly decouples heavy foundation model inference from the iterative training loop.

* **Dataset Acquisition:** Load `Marqo/fashion200k` via the Hugging Face `datasets` library.
* **Subsampling Strategy:** Implement an environment variable (e.g., `DEV_MODE=1`). When active, truncate the dataset to a small subset (e.g., $N=1000$ samples) to enable rapid local debugging.
* **Train/Val/Test Split:** Utilize `scikit-learn`'s `train_test_split` to partition the dataset (e.g., 80% Train, 10% Validation, 10% Test). Ensure strict isolation between splits.
* **Precomputation Pipeline:**
    1.  Initialize `dinov2` and `nomic-embed` in evaluation mode.
    2.  Iterate over all partitions.
    3.  Extract $d=384$ visual features and $d=768$ textual features.
    4.  Save the resulting unprojected embeddings alongside textual descriptions and image references to disk as PyTorch tensor files (`.pt`).

### Phase 3: Model Definition and Training Loop (PyTorch Lightning)
* **DataModule:** Create a `LightningDataModule` loading the precomputed `.pt` tensors. Implement `DataLoader` instances for train, val, and test splits.
* **LightningModule:**
    * Define $W_{image}$, $b_{image}$, $W_{text}$, $b_{text}$, and $\tau$.
    * Implement the `training_step`. Apply linear projections to input batches and normalize.
    * **Loss Formulation:** Compute the pairwise cosine similarity matrix scaled by $\exp(\tau)$. Calculate symmetric cross-entropy.
* **Validation Metrics:**
    * Log the symmetric cross-entropy loss as `val_loss`.
    * Implement and track Recall@K (e.g., R@1, R@5, R@10) as the primary monitor for hyperparameter tuning.
* **Hyperparameter Tuning:** Implement a grid search or integrate Ray Tune/Optuna over learning rate, weight decay, and $d_{joint}$. Monitor `val_recall@5` to trigger Early Stopping and Model Checkpointing.

### Phase 4: Remote Execution and Artifact Management
* Push the verified local codebase to a remote GPU server.
* Execute Phase 2 (Precomputation) on the remote server with `DEV_MODE=0` using CUDA to process the full dataset.
* Execute Phase 3 (Training) on the remote server utilizing CUDA.
* **Artifact Transfer:** Establish a clear protocol (e.g., `rsync`, `scp`, or scripted cloud storage transfer) to download the trained projection weights (the Lightning checkpoint), the raw dataset metadata (for EDA), and the precomputed, projected test-set database (for the retrieval index) back to the local machine.

### Phase 5: Streamlit Web Application
* **Initialization:** Load the trained projection layers ($W_{text}$, $W_{image}$) from the downloaded checkpoint. Load the precomputed, projected test-set database into a FAISS index. Load the raw dataset metadata into memory.
* **UI Layout:**
    * Sidebar: Controls for application mode selection (EDA, Text-to-Image, Image-to-Text) and hyperparameter adjustment (e.g., $K$ retrieved items, EDA sample size).
    * Main Panel: Context-dependent rendering based on the active mode.
* **Exploratory Data Analysis (EDA) Module:**
    * Implement a paginated or sampled view of the raw dataset partitions.
    * Render images alongside their corresponding textual descriptions.
    * Integrate lightweight distribution visualizations (e.g., histograms of description word counts, bar charts of garment categories) utilizing `altair` or Streamlit native charts to assess dataset variance and potential biases.
* **Inference Pipeline (Text-to-Image):**
    1.  Accept textual input.
    2.  Compute `nomic-embed` feature dynamically (prefixed with `search_query:`).
    3.  Apply $W_{text}$.
    4.  Query the image FAISS index.
    5.  Render the top-$K$ images.
* **Inference Pipeline (Image-to-Text):**
    1.  Accept uploaded image.
    2.  Compute `dinov2` feature dynamically.
    3.  Apply $W_{image}$.
    4.  Query the text FAISS index.
    5.  Render the top-$K$ textual descriptions.

### Phase 6: Testing and Refinement
* Verify inference and rendering latency on the local environment.
* Confirm that empirical evaluation metrics align with qualitative performance in the retrieval application.