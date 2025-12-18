"""
Emotion Detection Module
This module handles:
- Connecting to Hugging Face API for emotion detection
- Rule-based emotion detection as fallback
- Reading and using the GoEmotions dataset
"""

import requests
import pandas as pd
import os
from typing import Dict, List

# Hugging Face API Configuration
# Get your free API key from: https://huggingface.co/settings/tokens
HUGGING_FACE_API_KEY = os.environ.get('HUGGING_FACE_API_KEY', '')
EMOTION_MODEL_URL = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"

# Load GoEmotions dataset for emotion mapping
# This dataset contains text examples labeled with 27 different emotions
def load_goemotions_dataset():
    """
    Load the GoEmotions dataset
    Dataset structure:
    - text: the text content
    - emotion: the labeled emotion (joy, sadness, anger, etc.)
    
    Returns a pandas DataFrame
    """
    try:
        dataset_path = 'datasets/goemotions.csv'
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            print(f"✅ Loaded GoEmotions dataset: {len(df)} rows")
            return df
        else:
            print("⚠️ GoEmotions dataset not found. Using rule-based detection only.")
            return None
    except Exception as e:
        print(f"❌ Error loading GoEmotions dataset: {e}")
        return None


# Load dataset once when module is imported
GOEMOTIONS_DATA = load_goemotions_dataset()


def detect_emotion_huggingface(text: str) -> Dict:
    """
    Detect emotion using Hugging Face API
    
    Args:
        text: User's message
        
    Returns:
        Dictionary with emotion and confidence score
        Example: {'emotion': 'sadness', 'confidence': 0.89}
    """
    # Check if API key is available
    if not HUGGING_FACE_API_KEY:
        print("⚠️ No Hugging Face API key found. Using rule-based detection.")
        return None
    
    try:
        # Prepare API request
        headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}
        payload = {"inputs": text}
        
        # Send POST request to Hugging Face API
        response = requests.post(
            EMOTION_MODEL_URL, 
            headers=headers, 
            json=payload,
            timeout=10  # 10 second timeout
        )
        
        # Check if request was successful
        if response.status_code == 200:
            results = response.json()
            
            # API returns a list of emotions with confidence scores
            # Format: [{'label': 'joy', 'score': 0.95}, {'label': 'sadness', 'score': 0.03}, ...]
            if results and len(results) > 0:
                # Get the emotion with highest confidence
                top_emotion = max(results[0], key=lambda x: x['score'])
                
                return {
                    'emotion': top_emotion['label'],
                    'confidence': top_emotion['score'],
                    'all_emotions': results[0]  # All detected emotions
                }
        else:
            print(f"⚠️ Hugging Face API error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error calling Hugging Face API: {e}")
        return None


def detect_emotion_rule_based(text: str) -> Dict:
    """
    Rule-based emotion detection (fallback method)
    Uses keyword matching to detect emotions
    
    Args:
        text: User's message
        
    Returns:
        Dictionary with emotion and confidence
    """
    # Convert text to lowercase for matching
    text_lower = text.lower()
    
    # Define emotion keywords
    # These are common words associated with each emotion
    emotion_keywords = {
        'severe_distress': [
            'suicide', 'kill myself', 'end it all', 'want to die', 
            'hopeless', 'worthless', 'no reason to live', 'give up on life'
        ],
        'sadness': [
            'sad', 'depressed', 'down', 'unhappy', 'crying', 'tears',
            'lonely', 'empty', 'heartbroken', 'miserable', 'gloomy'
        ],
        'anxiety': [
            'anxious', 'worried', 'nervous', 'scared', 'afraid', 'panic',
            'stress', 'overwhelmed', 'terrified', 'fear', 'uneasy'
        ],
        'anger': [
            'angry', 'mad', 'furious', 'hate', 'irritated', 'frustrated',
            'annoyed', 'rage', 'pissed', 'bitter', 'resentful'
        ],
        'joy': [
            'happy', 'great', 'excellent', 'wonderful', 'amazing', 'fantastic',
            'joyful', 'excited', 'thrilled', 'delighted', 'cheerful', 'good'
        ],
        'neutral': [
            'okay', 'fine', 'alright', 'normal', 'average'
        ]
    }
    
    # Count keyword matches for each emotion
    emotion_scores = {}
    for emotion, keywords in emotion_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            emotion_scores[emotion] = score
    
    # If no keywords matched, return neutral
    if not emotion_scores:
        return {
            'emotion': 'neutral',
            'confidence': 0.5,
            'method': 'rule_based'
        }
    
    # Get emotion with highest score
    detected_emotion = max(emotion_scores, key=emotion_scores.get)
    
    # Calculate confidence (normalized score)
    max_score = emotion_scores[detected_emotion]
    total_words = len(text_lower.split())
    confidence = min(max_score / max(total_words * 0.3, 1), 0.95)
    
    return {
        'emotion': detected_emotion,
        'confidence': confidence,
        'method': 'rule_based',
        'scores': emotion_scores
    }


def map_emotion_with_dataset(text: str, detected_emotion: str) -> Dict:
    """
    Enhance emotion detection using GoEmotions dataset
    Finds similar examples in the dataset to improve accuracy
    
    Args:
        text: User's message
        detected_emotion: Emotion detected by API or rules
        
    Returns:
        Enhanced emotion result
    """
    # If dataset not loaded, return original emotion
    if GOEMOTIONS_DATA is None:
        return {'emotion': detected_emotion, 'enhanced': False}
    
    try:
        # Find similar texts in dataset (simple matching)
        text_lower = text.lower()
        words = set(text_lower.split())
        
        # Calculate similarity scores for dataset entries
        similar_entries = []
        for idx, row in GOEMOTIONS_DATA.head(1000).iterrows():  # Check first 1000 rows
            if 'text' in row and 'emotion' in row:
                dataset_words = set(str(row['text']).lower().split())
                # Calculate word overlap
                overlap = len(words.intersection(dataset_words))
                if overlap > 0:
                    similar_entries.append({
                        'emotion': row['emotion'],
                        'overlap': overlap
                    })
        
        # If we found similar entries, consider their emotions
        if similar_entries:
            # Count emotion frequencies in similar entries
            emotion_counts = {}
            for entry in similar_entries:
                emotion = entry['emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + entry['overlap']
            
            # Get most common emotion from dataset
            dataset_emotion = max(emotion_counts, key=emotion_counts.get)
            
            return {
                'emotion': dataset_emotion,
                'original_emotion': detected_emotion,
                'enhanced': True,
                'dataset_confidence': emotion_counts[dataset_emotion] / sum(emotion_counts.values())
            }
    
    except Exception as e:
        print(f"Error using dataset: {e}")
    
    # Return original if enhancement failed
    return {'emotion': detected_emotion, 'enhanced': False}


def detect_emotion(text: str) -> Dict:
    """
    Main emotion detection function
    Tries multiple methods in order:
    1. Hugging Face API (most accurate)
    2. Rule-based detection (fallback)
    3. Dataset enhancement (improves accuracy)
    
    Args:
        text: User's message
        
    Returns:
        Dictionary with emotion detection results
    """
    print(f"🔍 Detecting emotion for: '{text[:50]}...'")
    
    # Method 1: Try Hugging Face API first
    result = detect_emotion_huggingface(text)
    
    # Method 2: If API fails, use rule-based detection
    if result is None:
        result = detect_emotion_rule_based(text)
    
    # Method 3: Enhance with dataset if available
    if GOEMOTIONS_DATA is not None:
        enhanced_result = map_emotion_with_dataset(text, result['emotion'])
        if enhanced_result.get('enhanced'):
            result.update(enhanced_result)
    
    print(f"✅ Detected emotion: {result['emotion']} (confidence: {result.get('confidence', 0):.2f})")
    
    return result


# Test function (for development)
if __name__ == "__main__":
    # Test the emotion detection
    test_messages = [
        "I'm feeling really happy today!",
        "I'm so sad and lonely",
        "This makes me so angry",
        "I'm worried about everything",
        "I don't know what to feel"
    ]
    
    print("🧪 Testing Emotion Detection Module\n")
    for message in test_messages:
        result = detect_emotion(message)
        print(f"Message: {message}")
        print(f"Emotion: {result['emotion']}")
        print(f"Confidence: {result.get('confidence', 0):.2f}")
        print("-" * 50)