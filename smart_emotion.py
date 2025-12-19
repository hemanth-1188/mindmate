import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load GoEmotions dataset
df = pd.read_csv("datasets/goemotions.csv")

texts = df["text"].astype(str).tolist()
labels = df["emotion"].astype(str).tolist()

# Build TF-IDF model
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = vectorizer.fit_transform(texts)

def detect_emotion_tfidf(user_text: str):
    user_vec = vectorizer.transform([user_text])
    similarities = cosine_similarity(user_vec, tfidf_matrix)

    best_match = similarities.argmax()
    emotion = labels[best_match]
    confidence = similarities[0][best_match]

    return {
        "emotion": emotion,
        "confidence": round(float(confidence), 2),
        "method": "tfidf"
    }
