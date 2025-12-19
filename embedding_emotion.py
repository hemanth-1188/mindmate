from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

df = pd.read_csv("datasets/goemotions.csv")
texts = df["text"].astype(str).tolist()
labels = df["emotion"].astype(str).tolist()

dataset_embeddings = model.encode(texts, show_progress_bar=True)

def detect_emotion_embedding(user_text: str):
    user_embedding = model.encode([user_text])
    similarities = np.dot(dataset_embeddings, user_embedding.T).flatten()

    best_idx = similarities.argmax()

    return {
        "emotion": labels[best_idx],
        "confidence": round(float(similarities[best_idx]), 2),
        "method": "minilm"
    }
