"""
Emotion Detection – MindMate (FINAL)
✔ Dataset-aware (GoEmotions text only)
✔ Rule-based + similarity scoring
✔ No column assumptions
✔ API optional
"""

import pandas as pd
import os
import re
from typing import Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# LOAD GOEMOTIONS DATASET
# =========================

DATASET_PATH = "datasets/goemotions.csv"

def load_goemotions():
    if not os.path.exists(DATASET_PATH):
        print("⚠️ GoEmotions dataset not found")
        return None

    try:
        df = pd.read_csv(DATASET_PATH)
        if "text" not in df.columns:
            print("❌ GoEmotions CSV has no 'text' column")
            return None

        df = df[["text"]].dropna()
        print(f"✅ GoEmotions loaded: {len(df)} rows")
        return df

    except Exception as e:
        print("❌ Failed loading GoEmotions:", e)
        return None


GOEMOTIONS_DF = load_goemotions()

# =========================
# TF-IDF MODEL
# =========================

VECTORIZER = None
TFIDF_MATRIX = None

if GOEMOTIONS_DF is not None:
    VECTORIZER = TfidfVectorizer(
        stop_words="english",
        max_features=6000
    )
    TFIDF_MATRIX = VECTORIZER.fit_transform(GOEMOTIONS_DF["text"])


# =========================
# KEYWORD EMOTION MAP
# =========================

EMOTION_KEYWORDS = {
    "sadness": ["sad", "lonely", "depressed", "empty", "hopeless", "cry"],
    "anxiety": ["anxious", "worried", "panic", "nervous", "stress"],
    "anger": ["angry", "furious", "hate", "irritated"],
    "joy": ["happy", "excited", "great", "amazing", "love"],
    "fear": ["scared", "afraid", "terrified"],
    "severe_distress": [
        "suicide", "kill myself", "want to die",
        "end my life", "better off dead"
    ]
}


# =========================
# SIMILARITY-BASED SCORING
# =========================

def similarity_emotion(text: str) -> str:
    if VECTORIZER is None or TFIDF_MATRIX is None:
        return "neutral"

    vec = VECTORIZER.transform([text])
    sims = cosine_similarity(vec, TFIDF_MATRIX)[0]
    top_indices = sims.argsort()[-20:]

    text_blob = " ".join(
        GOEMOTIONS_DF.iloc[top_indices]["text"].astype(str)
    ).lower()

    scores = {}
    for emo, words in EMOTION_KEYWORDS.items():
        scores[emo] = sum(text_blob.count(w) for w in words)

    return max(scores, key=scores.get) if max(scores.values()) > 0 else "neutral"


# =========================
# MAIN FUNCTION
# =========================

def detect_emotion(text: str) -> Dict:
    text = text.lower()

    # First: direct keyword detection
    for emo, words in EMOTION_KEYWORDS.items():
        if any(w in text for w in words):
            return {
                "emotion": emo,
                "confidence": 0.85,
                "method": "keyword"
            }

    # Second: dataset similarity
    emotion = similarity_emotion(text)

    return {
        "emotion": emotion,
        "confidence": 0.65 if emotion != "neutral" else 0.5,
        "method": "tfidf"
    }
