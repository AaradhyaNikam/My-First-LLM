import streamlit as st
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests

# ------------------- Caching -------------------
@st.cache_data
def load_embeddings():
    return joblib.load('embeddings.joblib')

@st.cache_data
def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    return r.json()["embeddings"]

# ------------------- LLM Inference -------------------
def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })
    return r.json()["response"]

# ------------------- App UI -------------------
st.title("🎥 Video Tutorial Search Assistant")
st.write("Ask a question about the tutorial videos, and I will guide you to the relevant video and timestamp.")

# Load embeddings once
df = load_embeddings()

# User input
query = st.text_input("Enter your question related to the videos:")

if st.button("Search") and query.strip() != "":
    # 1️⃣ Generate query embedding
    query_embedding = create_embedding([query])[0]

    # 2️⃣ Compute similarities and select top 5 chunks
    similarities = cosine_similarity(np.vstack(df['embedding']), [query_embedding]).flatten()
    top_indices = np.argsort(-similarities)[:5]  # Descending order
    new_df = df.iloc[top_indices]

    # 3️⃣ Prepare clean prompt text
    chunk_text = "\n".join([
        f"Video #{row['number']} ({row['title']}) | Start: {row['start']}s, End: {row['end']}s\n{row['text']}"
        for _, row in new_df.iterrows()
    ])

    prompt = f"""I am providing you with chunks of tutorial video content:

{chunk_text}

User asked this question: "{query}"

Answer where and how much content is taught in which video and at what timestamp. If the question is unrelated, say that you can only answer based on the videos provided.
"""

    # 4️⃣ LLM inference
    response = inference(prompt)

    # 5️⃣ Display answer
    st.subheader("Answer from Video Content")
    st.write(response)

    # 6️⃣ Display clickable top chunks for user reference
    st.subheader("Top Relevant Video Chunks")
    for _, row in new_df.iterrows():
        st.markdown(f"- **Video #{row['number']} ({row['title']})**: [{row['start']}s → {row['end']}s](#)")
        st.write(row['text'])
