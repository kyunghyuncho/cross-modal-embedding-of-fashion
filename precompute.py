import os
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from tqdm import tqdm

def main():
    print("Beginning Phase 2: Data Ingestion and Feature Precomputation")
    
    # Check execution device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    # Load dataset
    print("Loading dataset 'Marqo/fashion200k'...")
    dataset = load_dataset("Marqo/fashion200k", split="data")

    # Dev Mode Subsampling
    dev_mode = os.environ.get("DEV_MODE", "0") == "1"
    if dev_mode:
        print("DEV_MODE=1 detected, subsampling 1000 samples for rapid debugging.")
        dataset = dataset.select(range(1000))

    # Identify text column
    text_col = next((c for c in dataset.column_names if c.lower() in ["text", "caption", "title", "description"]), None)
    if not text_col:
        raise ValueError(f"Could not automatically determine the text column from {dataset.column_names}")
    print(f"Using '{text_col}' for textual descriptions and 'image' for visual data.")

    # Convert to pandas for easier train/test splits, while keeping track of indices 
    # to access the raw PIL images from the hf dataset
    print("Partitioning data into Train (80%), Val (10%), Test (10%)...")
    indices = np.arange(len(dataset))
    
    # 80% train, 20% validation+test
    train_idx, val_test_idx = train_test_split(indices, test_size=0.20, random_state=42)
    # 10% validation, 10% test (from original)
    val_idx, test_idx = train_test_split(val_test_idx, test_size=0.50, random_state=42)

    partitions = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx
    }

    # Initialize Encoders
    print("Initializing Vision Encoder (facebook/dinov2-small)...")
    try:
        vis_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        vis_model = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
        vis_model.eval()
    except Exception as e:
        print(f"Error loading vision encoder: {e}")
        return

    print("Initializing Text Encoder (nomic-ai/nomic-embed-text-v1.5)...")
    try:
        # nomic-embed-text requires trust_remote_code=True for its specific RoPE scaling
        txt_tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
        txt_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True).to(device)
        txt_model.eval()
    except Exception as e:
        print(f"Error loading text encoder: {e}")
        return

    os.makedirs("data", exist_ok=True)

    batch_size = 32 if not dev_mode else 8

    # Extraction Loop
    for split_name, idx_array in partitions.items():
        print(f"Processing partition: {split_name} ({len(idx_array)} samples)")
        
        subset = dataset.select(idx_array)
        
        all_vis_embeddings = []
        all_txt_embeddings = []
        all_texts = []
        
        # Batch inference
        with torch.no_grad():
            for i in tqdm(range(0, len(subset), batch_size), desc=f"Extracting features for {split_name}"):
                batch = subset[i:i+batch_size]
                
                # Visual Processing
                # Filter out None images or corrupt modes if any exist
                images = []
                for img in batch["image"]:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    images.append(img)
                    
                vis_inputs = vis_processor(images=images, return_tensors="pt").to(device)
                vis_outputs = vis_model(**vis_inputs)
                vis_embeds = vis_outputs.last_hidden_state[:, 0, :] # CLS token (or standard output pooler)
                vis_embeds = torch.nn.functional.normalize(vis_embeds, p=2, dim=1)
                
                # Text Processing
                # As per nomic-embed instructions, prefix database entries with search_document:
                texts = [f"search_document: {t}" for t in batch[text_col]]
                
                txt_inputs = txt_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
                txt_outputs = txt_model(**txt_inputs)
                
                # mean pooling over target tokens
                attention_mask = txt_inputs['attention_mask']
                token_embeddings = txt_outputs[0]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                txt_embeds = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                txt_embeds = torch.nn.functional.normalize(txt_embeds, p=2, dim=1)

                # Move to cpu and append
                all_vis_embeddings.append(vis_embeds.cpu())
                all_txt_embeddings.append(txt_embeds.cpu())
                all_texts.extend(batch[text_col]) # Store original un-prefixed text
        
        # Concatenate tensors
        vis_tensor = torch.cat(all_vis_embeddings, dim=0)
        txt_tensor = torch.cat(all_txt_embeddings, dim=0)

        assert vis_tensor.shape[0] == len(subset)
        assert txt_tensor.shape[0] == len(subset)

        out_path = f"data/{split_name}_embeddings.pt"
        torch.save({
            "image_embeddings": vis_tensor,
            "text_embeddings": txt_tensor,
            "texts": all_texts,
            "indices": idx_array
        }, out_path)
        print(f"Saved {out_path} (Images: {vis_tensor.shape}, Texts: {txt_tensor.shape})")

    print("\nPrecomputation complete.")

if __name__ == "__main__":
    main()
