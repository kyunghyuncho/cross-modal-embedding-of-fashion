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

# ---------------------------------------------------------
# Custom CSS for Premium, Academic UI
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* App Background / Global styles */
.stApp {
    background: linear-gradient(180deg, #F9FAFB 0%, #FFFFFF 100%);
}

/* Dramatic Title Gradient */
h1 {
    background: linear-gradient(120deg, #4F46E5, #9333EA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    letter-spacing: -1px;
    margin-bottom: 0.5rem;
}

h2, h3 {
    color: #1F2937;
    font-weight: 600 !important;
    letter-spacing: -0.5px;
}

/* Metric / Subtitle text styling */
.stMarkdown p {
    color: #4B5563;
    font-size: 1.05rem;
    line-height: 1.6;
}

/* Image Card Styling */
img {
    border-radius: 12px;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
img:hover {
    transform: scale(1.03) translateY(-4px);
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #F3F4F6 !important;
    border-right: 1px solid #E5E7EB;
}
[data-testid="stSidebar"] h1 {
    -webkit-text-fill-color: #1F2937;
    font-size: 1.5rem !important;
}

/* Similarity Badge */
.sim-badge {
    display: inline-block;
    padding: 0.25em 0.75em;
    font-size: 0.85em;
    font-weight: 600;
    line-height: 1;
    text-align: center;
    white-space: nowrap;
    vertical-align: baseline;
    border-radius: 9999px;
    background-color: #E0E7FF;
    color: #3730A3;
    margin-top: 8px;
}
</style>
"""

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
            return os.path.join(ckpt_dir, ckpts[0])
    return "dummy.ckpt"

def main():
    st.set_page_config(page_title="Semantic Fashion Retrieval", layout="wide", page_icon="🛍️")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Navigation Sidebar
    with st.sidebar:
        st.title("Navigation")
        st.write("Explore pedagogical cross-modal embedding spaces.")
        st.divider()
        mode = st.radio("Select View", ["📊 Exploratory Data Analysis", "🕵️‍♂️ Text-to-Image", "📸 Image-to-Text", "🚀 Model Training"])
        st.divider()
        st.caption("powered by PyTorch Lightning & FAISS")

    # Main App Header
    st.title("Semantic Fashion Retrieval")
    
    dataset = load_dataset_metadata()
    
    if mode == "📊 Exploratory Data Analysis":
        st.subheader("Dataset Topography")
        st.write("Browse a randomized sample representing the multi-modal distribution of the `Fashion200k` dataset, visualizing both visual assets and corresponding semantic annotations.")
        
        sample_size = st.sidebar.slider("Sample Size", 10, 500, 50)
        
        import random
        if "eda_indices" not in st.session_state or st.session_state.get("eda_sample_size") != sample_size:
            st.session_state.eda_indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))
            st.session_state.eda_sample_size = sample_size
            
        random_indices = st.session_state.eda_indices
        subset = dataset.select(random_indices)
        df = subset.to_pandas()
        
        text_col = next((c for c in df.columns if c.lower() in ["text", "caption", "title", "description"]), None)
        if text_col:
            df["word_count"] = df[text_col].apply(lambda x: len(str(x).split()))
            
            with st.container():
                st.markdown("### Description Length Distribution")
                chart = alt.Chart(df).mark_area(
                    line={'color':'#4F46E5'},
                    color=alt.Gradient(
                        gradient='linear',
                        stops=[alt.GradientStop(color='#4F46E5', offset=0),
                               alt.GradientStop(color='#9333EA', offset=1)],
                        x1=1, x2=1, y1=1, y2=0
                    )
                ).encode(
                    alt.X("word_count:Q", bin=alt.Bin(maxbins=20), title="Word Count"),
                    alt.Y("count()", title="Frequency Density"),
                    tooltip=["count()"]
                ).properties(height=250)
                st.altair_chart(chart, use_container_width=True)

        st.markdown("### Sample Embeddings View")
        cols = st.columns(4, gap="large")
        for i, idx in enumerate(random_indices[:12]):
            with cols[i % 4]:
                item = dataset[idx]
                st.image(item["image"], use_container_width=True)
                cap_text = item.get(text_col, "") if text_col else ""
                st.markdown(f"<div style='text-align: center; font-style: italic; color: #6B7280; font-size: 0.9em; margin-top: 8px;'>\"{cap_text}\"</div>", unsafe_allow_html=True)

    elif mode == "🕵️‍♂️ Text-to-Image":
        st.subheader("Text-to-Image Search")
        st.write("Enter a semantic query to search the joint index. The DINOv2 visual space has been aligned with the Nomic text boundaries.")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("Semantic Query", placeholder="e.g., 'a casual blue denim jacket'", label_visibility="collapsed")
        with col2:
            k = st.slider("Retrievals ($K$)", 1, 20, 5, label_visibility="collapsed")
            
        if query:
            with st.spinner("Embedding query and traversing FAISS index..."):
                vis_processor, vis_model, txt_tokenizer, txt_model, device = load_encoders()
                checkpoint_path = _get_best_checkpoint()
                model = load_projection_model(checkpoint_path)
                model.to(device).eval()
                
                image_embeddings, _, _, indices = load_faiss_indices()
                if image_embeddings is None:
                    st.error("No built FAISS index. Did you run the precomputation and training loops?")
                    return

                with torch.no_grad():
                    txt_inputs = txt_tokenizer([f"search_query: {query}"], padding=True, truncation=True, return_tensors="pt").to(device)
                    txt_outputs = txt_model(**txt_inputs)
                    
                    attention_mask = txt_inputs['attention_mask']
                    token_embeddings = txt_outputs[0]
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    txt_embeds = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    txt_embeds = F.normalize(txt_embeds, p=2, dim=1)
                    
                    joint_query = model.forward_text(txt_embeds).cpu().numpy()
                    
                    projected_img_list = []
                    for i in range(0, len(image_embeddings), 5000):
                        batch = torch.tensor(image_embeddings[i:i+5000]).to(device)
                        proj = model.forward_image(batch).cpu().numpy()
                        projected_img_list.append(proj)
                    projected_img = np.concatenate(projected_img_list, axis=0)
                    
                    index = faiss.IndexFlatIP(projected_img.shape[1])
                    index.add(projected_img)
                    
                    distances, I = index.search(joint_query, k)
                    
            st.markdown(f"**Top {k} Ranked Visual Matches:**")
            cols = st.columns(min(k, 5), gap="large")
            for i, (dist, idx) in enumerate(zip(distances[0], I[0])):
                col_idx = i % 5
                orig_idx = indices[idx]
                if i > 0 and col_idx == 0:
                    cols = st.columns(5, gap="large")
                with cols[col_idx]:
                    item = dataset[int(orig_idx)]
                    st.image(item["image"], use_container_width=True)
                    text_col = next((c for c in item.keys() if c.lower() in ["text", "caption", "title", "description"]), None)
                    cap = item.get(text_col, "")
                    st.markdown(f"<p style='text-align:center; font-style:italic; font-size:0.85em;'>{cap}</p>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align:center;'><span class='sim-badge'>cos sim: {dist:.3f}</span></div>", unsafe_allow_html=True)

    elif mode == "📸 Image-to-Text":
        st.subheader("Image-to-Text Reverse Search")
        st.write("Upload a target graphic. The aligned joint space will construct a localized semantic neighborhood.")
        
        k = st.sidebar.slider("Number of results", 1, 15, 5)
        
        uploaded_file = st.file_uploader("Drop Image File", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            
            # Display uploaded image beautifully
            col1, col2 = st.columns([1, 2], gap="large")
            with col1:
                st.image(image, caption="Query Image", use_container_width=True)
            
            with col2:
                with st.spinner("Iterating Image Graph Traversal..."):
                    vis_processor, vis_model, txt_tokenizer, txt_model, device = load_encoders()
                    checkpoint_path = _get_best_checkpoint()
                    model = load_projection_model(checkpoint_path)
                    model.to(device).eval()
                    
                    _, text_embeddings, texts, _ = load_faiss_indices()
                    if text_embeddings is None:
                        st.error("No built FAISS index.")
                        return

                    with torch.no_grad():
                        vis_inputs = vis_processor(images=[image], return_tensors="pt").to(device)
                        vis_outputs = vis_model(**vis_inputs)
                        vis_embeds = vis_outputs.last_hidden_state[:, 0, :]
                        vis_embeds = F.normalize(vis_embeds, p=2, dim=1)
                        
                        joint_query = model.forward_image(vis_embeds).cpu().numpy()
                        
                        projected_txt_list = []
                        for i in range(0, len(text_embeddings), 5000):
                            batch = torch.tensor(text_embeddings[i:i+5000]).to(device)
                            proj = model.forward_text(batch).cpu().numpy()
                            projected_txt_list.append(proj)
                        projected_txt = np.concatenate(projected_txt_list, axis=0)
                        
                        index = faiss.IndexFlatIP(projected_txt.shape[1])
                        index.add(projected_txt)
                        
                        distances, I = index.search(joint_query, k)
                    
                    st.markdown("#### Predicted Alignments:")
                    for idx, dist in zip(I[0], distances[0]):
                        st.markdown(f"""
                        <div style="background: white; padding: 12px 20px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #E5E7EB; display: flex; justify-content: space-between; align-items: center;">
                            <span style="font-weight: 500; font-size: 1.05em; color: #111827;">{texts[idx].replace('search_document:', '').strip()}</span>
                            <span class='sim-badge'>cos sim: {dist:.3f}</span>
                        </div>
                        """, unsafe_allow_html=True)
                        
    elif mode == "🚀 Model Training":
        st.subheader("Interactive Model Training")
        st.write("Orchestrate the cross-modal projection layer tuning natively. Metrics will stream live below.")
        
        is_running = os.path.exists("lightning_logs/running.flag")
        
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.markdown("### Hyperparameters")
            if not is_running:
                epochs = st.number_input("Max Epochs", min_value=1, max_value=100, value=20)
                batch_size = st.number_input("Batch Size", min_value=8, max_value=1024, value=256)
                lr = st.number_input("Learning Rate", min_value=1e-5, max_value=1e-2, value=1e-3, format="%.5f")
                
                if st.button("▶️ Start Training", type="primary", use_container_width=True):
                    import subprocess
                    if os.path.exists("lightning_logs/stop.flag"):
                        os.remove("lightning_logs/stop.flag")
                    if os.path.exists("lightning_logs/live_metrics.json"):
                        os.remove("lightning_logs/live_metrics.json")
                    
                    env = os.environ.copy()
                    subprocess.Popen([
                        "python", "train.py", 
                        "--epochs", str(epochs),
                        "--batch_size", str(batch_size),
                        "--lr", str(lr)
                    ], env=env)
                    st.rerun()
            else:
                st.info("Training is currently active in the background.")
                if st.button("⏹️ Stop Training", type="primary", use_container_width=True):
                    with open("lightning_logs/stop.flag", "w") as f:
                        f.write("stop")
                    st.warning("Stop signal sent. Wait a moment for the process to exit.")
                    st.rerun()
                    
        with col2:
            st.markdown("### Live Optimization Metrics")
            
            run_interval = "2s" if is_running else None
            
            @st.fragment(run_every=run_interval)
            def render_live_metrics():
                if os.path.exists("lightning_logs/live_metrics.json"):
                    import json
                    try:
                        with open("lightning_logs/live_metrics.json", "r") as f:
                            metrics = json.load(f)
                        
                        if metrics.get("train_loss"):
                            import pandas as pd
                            df_loss = pd.DataFrame(metrics["train_loss"]).set_index("step")
                            st.markdown("**Train Loss**")
                            st.line_chart(df_loss, y="value", height=200)
                        else:
                            st.write("Waiting for first training batch...")
                            
                        if metrics.get("val_recall@5"):
                            import pandas as pd
                            df_val = pd.DataFrame(metrics["val_recall@5"]).set_index("step")
                            st.markdown("**Validation Recall@5**")
                            st.line_chart(df_val, y="value", color="#9333EA", height=200)
                            
                    except Exception as e:
                        pass
            
            render_live_metrics()

if __name__ == "__main__":
    main()
