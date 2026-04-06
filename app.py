import streamlit as st
import torch
import torch.nn.functional as F
import faiss
import numpy as np
import altair as alt
import os
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from models import CrossModalRetrievalModel
from datasets import load_dataset
from PIL import Image

@st.cache_resource
def load_encoders():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    vis_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    vis_model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    
    txt_tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    txt_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True).to(device).eval()
    return vis_processor, vis_model, txt_tokenizer, txt_model, device

@st.cache_resource
def load_projection_model(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        st.warning(f"Checkpoint not found at {checkpoint_path}. Using uninitialized projection weights.")
        return CrossModalRetrievalModel(d_image=384, d_text=768, d_joint=512)
    return CrossModalRetrievalModel.load_from_checkpoint(checkpoint_path)

@st.cache_resource
def load_dataset_metadata():
    ds = load_dataset("Marqo/fashion200k", split="data")
    dev_mode = os.environ.get("DEV_MODE", "0") == "1"
    if dev_mode:
        ds = ds.select(range(1000))
    return ds

@st.cache_resource
def load_faiss_indices(data_dir="data"):
    test_pt = os.path.join(data_dir, "test_embeddings.pt")
    if not os.path.exists(test_pt):
        st.error(f"{test_pt} not found. Please run precompute.py and train.py first.")
        return None, None, None, None
        
    data = torch.load(test_pt, map_location="cpu", weights_only=False)
    image_embeddings = data["image_embeddings"].numpy()
    text_embeddings = data["text_embeddings"].numpy()
    texts = data["texts"]
    indices = data["indices"]

    return image_embeddings, text_embeddings, texts, indices

def _get_best_checkpoint():
    ckpt_dir = "lightning_logs/checkpoints"
    if os.path.exists(ckpt_dir):
        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".ckpt")]
        if ckpts:
            return os.path.join(ckpt_dir, ckpts[0]) # For production, parse val_recall to find max
    return "dummy.ckpt"

def main():
    st.set_page_config(page_title="Cross-Modal Fashion Retrieval", layout="wide")
    st.sidebar.title("Navigation")
    
    mode = st.sidebar.radio("Mode", ["Exploratory Data Analysis", "Text-to-Image", "Image-to-Text"])
    
    st.title("Fashion Retrieval Application")
    dataset = load_dataset_metadata()
    
    if mode == "Exploratory Data Analysis":
        st.header("Exploratory Data Analysis")
        st.write("Browse a random sample of the Fashion200k dataset.")
        sample_size = st.sidebar.slider("Sample Size", 10, 500, 50)
        
        # Subsample for EDA
        import random
        random_indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
        subset = dataset.select(random_indices)
        
        df = subset.to_pandas()
        text_col = next((c for c in df.columns if c.lower() in ["text", "caption", "title", "description"]), None)
        if text_col:
            df["word_count"] = df[text_col].apply(lambda x: len(str(x).split()))
            
            # Simple histogram using Altair
            chart = alt.Chart(df).mark_bar().encode(
                alt.X("word_count:Q", bin=True, title="Word Count in Description"),
                alt.Y("count()", title="Frequency")
            ).properties(title="Distribution of Description Lengths")
            st.altair_chart(chart, use_container_width=True)
            
        st.subheader("Data Samples")
        cols = st.columns(4)
        for i, idx in enumerate(random_indices[:12]):
            with cols[i % 4]:
                item = dataset[idx]
                st.image(item["image"], use_container_width=True)
                st.caption(item.get(text_col, "") if text_col else "")

    elif mode == "Text-to-Image":
        st.header("Text-to-Image Retrieval")
        query = st.text_input("Enter a query (e.g., 'white t-shirt with print')")
        k = st.sidebar.slider("Number of results", 1, 20, 5)
        
        if query:
            vis_processor, vis_model, txt_tokenizer, txt_model, device = load_encoders()
            checkpoint_path = _get_best_checkpoint()
            model = load_projection_model(checkpoint_path)
            model.to(device).eval()
            
            image_embeddings, _, _, indices = load_faiss_indices()
            if image_embeddings is not None:
                with torch.no_grad():
                    txt_inputs = txt_tokenizer([f"search_query: {query}"], padding=True, truncation=True, return_tensors="pt").to(device)
                    txt_outputs = txt_model(**txt_inputs)
                    
                    attention_mask = txt_inputs['attention_mask']
                    token_embeddings = txt_outputs[0]
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    txt_embeds = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    txt_embeds = F.normalize(txt_embeds, p=2, dim=1)
                    
                    joint_query = model.forward_text(txt_embeds).cpu().numpy()
                    
                    # Project image precomputed base embeddings through the W_image projection layer
                    # Batch processing to prevent OOM
                    projected_img_list = []
                    for i in range(0, len(image_embeddings), 5000):
                        batch = torch.tensor(image_embeddings[i:i+5000]).to(device)
                        proj = model.forward_image(batch).cpu().numpy()
                        projected_img_list.append(proj)
                    projected_img = np.concatenate(projected_img_list, axis=0)
                    
                    index = faiss.IndexFlatIP(projected_img.shape[1])
                    index.add(projected_img)
                    
                    distances, I = index.search(joint_query, k)
                    
                    st.subheader(f"Top {k} matches:")
                    cols = st.columns(min(k, 5))
                    for i, (dist, idx) in enumerate(zip(distances[0], I[0])):
                        col_idx = i % 5
                        orig_idx = indices[idx]
                        if i > 0 and col_idx == 0:
                            cols = st.columns(5)
                        with cols[col_idx]:
                            item = dataset[int(orig_idx)]
                            st.image(item["image"], use_container_width=True)
                            text_col = next((c for c in item.keys() if c.lower() in ["text", "caption", "title", "description"]), None)
                            st.caption(item.get(text_col, ""))
                            st.write(f"Similarity: {dist:.3f}")

    elif mode == "Image-to-Text":
        st.header("Image-to-Text Retrieval")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        k = st.sidebar.slider("Number of results", 1, 20, 5)
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, width=200)
            
            vis_processor, vis_model, txt_tokenizer, txt_model, device = load_encoders()
            checkpoint_path = _get_best_checkpoint()
            model = load_projection_model(checkpoint_path)
            model.to(device).eval()
            
            _, text_embeddings, texts, _ = load_faiss_indices()
            if text_embeddings is not None:
                with torch.no_grad():
                    vis_inputs = vis_processor(images=[image], return_tensors="pt").to(device)
                    vis_outputs = vis_model(**vis_inputs)
                    vis_embeds = vis_outputs.last_hidden_state[:, 0, :]
                    vis_embeds = F.normalize(vis_embeds, p=2, dim=1)
                    
                    joint_query = model.forward_image(vis_embeds).cpu().numpy()
                    
                    # Project text precomputed embeddings through the W_text projection layer
                    projected_txt_list = []
                    for i in range(0, len(text_embeddings), 5000):
                        batch = torch.tensor(text_embeddings[i:i+5000]).to(device)
                        proj = model.forward_text(batch).cpu().numpy()
                        projected_txt_list.append(proj)
                    projected_txt = np.concatenate(projected_txt_list, axis=0)
                    
                    index = faiss.IndexFlatIP(projected_txt.shape[1])
                    index.add(projected_txt)
                    
                    distances, I = index.search(joint_query, k)
                    
                    st.subheader(f"Top {k} textual matches:")
                    for idx, dist in zip(I[0], distances[0]):
                        st.write(f"**Similarity: {dist:.3f}** -> {texts[idx]}")

if __name__ == "__main__":
    main()
