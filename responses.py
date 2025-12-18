"""
Response Generation Module
This module handles:
- Generating empathetic responses based on detected emotions
- Calculating depression risk scores
- Using control.csv and condition.csv datasets for risk assessment
- Providing appropriate support resources
"""

import pandas as pd
import os
import random
from typing import Dict, List

# Load depression risk datasets
def load_depression_datasets():
    """
    Load control and condition datasets
    
    control.csv: Contains text from people without depression
    condition.csv: Contains text from people with depression
    scores.csv: Contains depression severity scores
    
    These datasets help identify patterns associated with depression risk
    """
    datasets = {}
    
    try:
        # Load control group (non-depressed individuals)
        control_path = 'datasets/control.csv'
        if os.path.exists(control_path):
            datasets['control'] = pd.read_csv(control_path)
            print(f"✅ Loaded control dataset: {len(datasets['control'])} rows")
    except Exception as e:
        print(f"⚠️ Could not load control.csv: {e}")
        datasets['control'] = None
    
    try:
        # Load condition group (individuals with depression)
        condition_path = 'datasets/condition.csv'
        if os.path.exists(condition_path):
            datasets['condition'] = pd.read_csv(condition_path)
            print(f"✅ Loaded condition dataset: {len(datasets['condition'])} rows")
    except Exception as e:
        print(f"⚠️ Could not load condition.csv: {e}")
        datasets['condition'] = None
    
    try:
        # Load severity scores dataset
        scores_path = 'datasets/scores.csv'
        if os.path.exists(scores_path):
            datasets['scores'] = pd.read_csv(scores_path)
            print(f"✅ Loaded scores dataset: {len(datasets['scores'])} rows")
    except Exception as e:
        print(f"⚠️ Could not load scores.csv: {e}")
        datasets['scores'] = None
    
    return datasets

# Load datasets when module is imported
DEPRESSION_DATASETS = load_depression_datasets()


def calculate_risk_score(text: str, emotion: str) -> int:
    """
    Calculate depression risk score based on message content and emotion
    
    Args:
        text: User's message
        emotion: Detected emotion
        
    Returns:
        Risk score increment (0-3)
        0 = No risk
        1 = Low risk
        2 = Medium risk
        3 = High risk
    """
    text_lower = text.lower()
    
    # High-risk keywords (immediate concern)
    high_risk_keywords = [
        'suicide', 'kill myself', 'end it all', 'want to die', 
        'no reason to live', 'better off dead', 'harm myself',
        'can\'t go on', 'give up on life'
    ]
    
    # Medium-risk keywords (depression indicators)
    medium_risk_keywords = [
        'hopeless', 'worthless', 'nobody cares', 'hate myself',
        'failure', 'useless', 'burden', 'can\'t do anything right',
        'everything is wrong', 'nothing matters'
    ]
    
    # Low-risk keywords (sadness/anxiety)
    low_risk_keywords = [
        'sad', 'depressed', 'anxious', 'worried', 'lonely',
        'empty', 'tired', 'exhausted', 'stressed'
    ]
    
    # Check for high-risk keywords first
    if any(keyword in text_lower for keyword in high_risk_keywords):
        return 3  # High risk - immediate concern
    
    # Check for medium-risk keywords
    if any(keyword in text_lower for keyword in medium_risk_keywords):
        return 2  # Medium risk
    
    # Check emotion-based risk
    if emotion == 'severe_distress':
        return 3
    elif emotion in ['sadness', 'anxiety']:
        # Check if it's just mild sadness or more concerning
        if any(keyword in text_lower for keyword in low_risk_keywords):
            return 1  # Low risk
    
    # Use dataset comparison if available
    if DEPRESSION_DATASETS.get('condition') is not None:
        risk_from_dataset = compare_with_depression_dataset(text)
        return max(risk_from_dataset, 0)
    
    return 0  # No significant risk detected


def compare_with_depression_dataset(text: str) -> int:
    """
    Compare user's message with depression dataset patterns
    
    Args:
        text: User's message
        
    Returns:
        Risk score based on dataset similarity
    """
    try:
        condition_df = DEPRESSION_DATASETS.get('condition')
        control_df = DEPRESSION_DATASETS.get('control')
        
        if condition_df is None or control_df is None:
            return 0
        
        text_lower = text.lower()
        words = set(text_lower.split())
        
        # Calculate similarity with condition (depressed) texts
        condition_similarity = 0
        for idx, row in condition_df.head(100).iterrows():  # Sample first 100
            if 'text' in row:
                condition_words = set(str(row['text']).lower().split())
                overlap = len(words.intersection(condition_words))
                condition_similarity += overlap
        
        # Calculate similarity with control (non-depressed) texts
        control_similarity = 0
        for idx, row in control_df.head(100).iterrows():  # Sample first 100
            if 'text' in row:
                control_words = set(str(row['text']).lower().split())
                overlap = len(words.intersection(control_words))
                control_similarity += overlap
        
        # If more similar to condition dataset, increase risk
        if condition_similarity > control_similarity * 1.5:
            return 2  # Medium risk
        elif condition_similarity > control_similarity:
            return 1  # Low risk
        
        return 0
        
    except Exception as e:
        print(f"Error in dataset comparison: {e}")
        return 0


def generate_response(emotion: str, risk_score: int, conversation_history: List[str] = None) -> str:
    """
    Generate empathetic response based on emotion and risk level
    
    Args:
        emotion: Detected emotion
        risk_score: Current cumulative risk score (0-10)
        conversation_history: List of previous emotions in conversation
        
    Returns:
        Empathetic response string
    """
    # Define response templates for each emotion
    responses = {
        'severe_distress': [
            "I hear that you're going through an incredibly difficult time. Your feelings are valid, and you don't have to face this alone. 💙",
            "What you're experiencing sounds overwhelming, and I'm genuinely concerned. Please know that you deserve support and care.",
            "I want you to know that your life matters, even when it doesn't feel that way. These feelings can change, and help is available.",
        ],
        'sadness': [
            "I'm sorry you're feeling this way. It's okay to feel sad - your emotions are completely valid. 💙",
            "It sounds like you're going through a tough time. Remember, it's okay not to be okay sometimes.",
            "I hear you, and I want you to know that these feelings won't last forever. Take things one moment at a time.",
            "Sadness is a natural human emotion. Be gentle with yourself during this difficult time.",
        ],
        'anxiety': [
            "It sounds like you're feeling anxious. Take a deep breath with me - in for 4, hold for 4, out for 4. 🌊",
            "Anxiety can feel overwhelming, but you're not alone in this. Let's take things one step at a time.",
            "I understand you're worried. Sometimes it helps to focus on what you can control right now.",
            "Your anxiety is valid. Would it help to talk about what's worrying you?",
        ],
        'anger': [
            "I can sense your frustration. It's okay to feel angry - it's a valid emotion. 🌟",
            "It sounds like something has really upset you. Would it help to talk about what happened?",
            "Your feelings of anger are understandable. Let's work through this together.",
            "Anger often tells us that something important to us has been affected. I'm here to listen.",
        ],
        'joy': [
            "That's wonderful to hear! I'm so glad you're feeling positive. ✨",
            "Your happiness is contagious! What's making you feel so good today?",
            "I love hearing this! It's great that you're in a good place right now.",
            "That's amazing! Celebrating the good moments with you! 🎉",
        ],
        'neutral': [
            "Thank you for sharing. I'm here to listen. How else are you feeling?",
            "I'm listening. Feel free to share whatever is on your mind.",
            "I'm here for you. What would you like to talk about?",
            "I appreciate you opening up. Tell me more about what's going on.",
        ]
    }
    
    # Get appropriate responses for the emotion
    emotion_responses = responses.get(emotion, responses['neutral'])
    base_response = random.choice(emotion_responses)
    
    # Add context-aware follow-ups based on conversation history
    if conversation_history and len(conversation_history) > 3:
        # Check for persistent sadness/anxiety
        recent_emotions = conversation_history[-5:]
        sad_count = recent_emotions.count('sadness')
        anxiety_count = recent_emotions.count('anxiety')
        
        if sad_count >= 3:
            base_response += "\n\nI've noticed you've been feeling down for a while. Have you considered talking to someone you trust about this?"
        elif anxiety_count >= 3:
            base_response += "\n\nI notice anxiety has been coming up often. Would it help to explore some coping strategies together?"
    
    # Add crisis resources if risk score is high
    if risk_score >= 7:
        base_response += "\n\n⚠️ I'm genuinely concerned about you. Please consider reaching out for professional support:"
        base_response += "\n• National Suicide Prevention Lifeline: 988 (24/7)"
        base_response += "\n• Crisis Text Line: Text HOME to 741741"
        base_response += "\n• Emergency Services: 911"
        base_response += "\n\nYou deserve help, and these resources are here for you anytime."
    elif risk_score >= 4:
        base_response += "\n\n💙 If you're struggling, please consider talking to a mental health professional. They can provide support tailored to your needs."
    
    # Add supportive closing for high-risk emotions
    if emotion in ['severe_distress', 'sadness']:
        closings = [
            "\n\nRemember: You are not alone in this. 💙",
            "\n\nI'm here with you. One step at a time. 🌟",
            "\n\nYour feelings matter, and so do you. 💙"
        ]
        base_response += random.choice(closings)
    
    return base_response


def get_coping_strategies(emotion: str) -> List[str]:
    """
    Provide coping strategies based on emotion
    
    Args:
        emotion: The detected emotion
        
    Returns:
        List of coping strategy suggestions
    """
    strategies = {
        'sadness': [
            "Practice self-compassion - talk to yourself like you would to a good friend",
            "Engage in small activities you enjoy, even if you don't feel like it",
            "Reach out to someone you trust",
            "Write down your thoughts in a journal",
            "Take a short walk outside if possible"
        ],
        'anxiety': [
            "Try the 5-4-3-2-1 grounding technique (5 things you see, 4 you hear, etc.)",
            "Practice deep breathing exercises",
            "Write down your worries to externalize them",
            "Focus on what you can control right now",
            "Progressive muscle relaxation"
        ],
        'anger': [
            "Take a break and step away from the situation if possible",
            "Physical exercise can help release tension",
            "Express your feelings through writing or art",
            "Practice counting to 10 before responding",
            "Identify the root cause of your anger"
        ]
    }
    
    return strategies.get(emotion, [])


# Test function
if __name__ == "__main__":
    print("🧪 Testing Response Generation Module\n")
    
    # Test different scenarios
    test_cases = [
        ("I want to end it all", "severe_distress", 8),
        ("I'm feeling really sad", "sadness", 2),
        ("I'm worried about my exam", "anxiety", 1),
        ("This is so frustrating!", "anger", 0),
        ("I'm having a great day!", "joy", 0),
    ]
    
    for text, emotion, risk in test_cases:
        print(f"Text: {text}")
        print(f"Emotion: {emotion}")
        print(f"Risk Score: {risk}")
        response = generate_response(emotion, risk)
        print(f"Response: {response}")
        print("-" * 70)