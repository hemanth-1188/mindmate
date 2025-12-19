"""
MindMate Backend - Flask Application
This is the main backend file that handles:
- Web routes (serving the HTML page)
- API endpoints (chat functionality)
- Session management (tracking user conversations)
"""

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import os
from datetime import datetime
import secrets

# Import our custom modules
from emotions import detect_emotion
from responses import generate_response, calculate_risk_score

# Initialize Flask app
app = Flask(__name__)
# Generate a secret key for session management
app.secret_key = secrets.token_hex(16)
# Enable CORS (Cross-Origin Resource Sharing) for API calls
CORS(app)

# Store user conversations in memory (in production, use a database)
# Dictionary structure: { session_id: { 'messages': [], 'emotions': [], 'risk_score': 0 } }
user_sessions = {}


@app.route('/')
def index():
    """
    Main route - serves the HTML page
    When user visits http://localhost:5000/, this function runs
    """
    # Generate a unique session ID for this user
    if 'session_id' not in session:
        session['session_id'] = secrets.token_hex(8)
        # Initialize user session data
        user_sessions[session['session_id']] = {
            'messages': [],
            'emotions': [],
            'risk_score': 0,
            'created_at': datetime.now()
        }
    
    # Render the HTML template
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Chat API endpoint
    Receives user message, detects emotion, calculates risk, returns response
    
    Expected input (JSON):
    {
        "message": "I'm feeling sad today"
    }
    
    Returns (JSON):
    {
        "response": "Bot's empathetic response",
        "emotion": "sadness",
        "risk_score": 2,
        "mood_history": ["sadness", "anxiety", ...]
    }
    """
    try:
        # Get the JSON data from the request
        data = request.get_json()
        
        # Extract user message
        user_message = data.get('message', '').strip()
        
        # Validate input
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Get or create session ID
        session_id = session.get('session_id', secrets.token_hex(8))
        if session_id not in user_sessions:
            user_sessions[session_id] = {
                'messages': [],
                'emotions': [],
                'risk_score': 0,
                'created_at': datetime.now()
            }
        
        # Get user session data
        user_data = user_sessions[session_id]
        
        # STEP 1: Detect emotion from user message
        # This calls the Hugging Face API (or uses rule-based detection)
        emotion_result = detect_emotion(user_message)
        detected_emotion = emotion_result['emotion']
        confidence = emotion_result.get('confidence', 0.0)
        
        # STEP 2: Calculate depression risk score
        # This analyzes the message and emotion to assess mental health risk
        risk_increment = calculate_risk_score(user_message, detected_emotion)
        user_data['risk_score'] = min(user_data['risk_score'] + risk_increment, 10)
        
        # STEP 3: Generate empathetic response
        # This creates a supportive response based on the detected emotion
        bot_response = generate_response(
            detected_emotion, 
            user_data['risk_score'],
            user_data['emotions']  # Pass conversation history for context
        )
        
        # STEP 4: Update user session data
        user_data['messages'].append({
            'user': user_message,
            'bot': bot_response,
            'emotion': detected_emotion,
            'timestamp': datetime.now().isoformat()
        })
        user_data['emotions'].append(detected_emotion)
        
        # Keep only last 20 emotions to prevent memory overflow
        if len(user_data['emotions']) > 20:
            user_data['emotions'] = user_data['emotions'][-20:]
        
        # STEP 5: Prepare response data
        response_data = {
            'response': bot_response,
            'emotion': detected_emotion,
            'confidence': confidence,
            'risk_score': user_data['risk_score'],
            'mood_history': user_data['emotions'][-10:],  # Last 10 emotions
            'timestamp': datetime.now().isoformat()
        }
        
        # Return JSON response
        return jsonify(response_data), 200
        
    except Exception as e:
        # If any error occurs, return error message
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': 'An error occurred processing your message'}), 500


@app.route('/api/mood-history', methods=['GET'])
def mood_history():
    """
    Get mood history for current user
    Returns the last 10 detected emotions
    """
    try:
        session_id = session.get('session_id')
        if session_id and session_id in user_sessions:
            emotions = user_sessions[session_id]['emotions'][-10:]
            return jsonify({'mood_history': emotions}), 200
        else:
            return jsonify({'mood_history': []}), 200
    except Exception as e:
        print(f"Error in mood-history endpoint: {str(e)}")
        return jsonify({'error': 'Could not retrieve mood history'}), 500


@app.route('/api/reset', methods=['POST'])
def reset_session():
    """
    Reset user session (clear conversation history)
    """
    try:
        session_id = session.get('session_id')
        if session_id and session_id in user_sessions:
            user_sessions[session_id] = {
                'messages': [],
                'emotions': [],
                'risk_score': 0,
                'created_at': datetime.now()
            }
        return jsonify({'message': 'Session reset successfully'}), 200
    except Exception as e:
        print(f"Error in reset endpoint: {str(e)}")
        return jsonify({'error': 'Could not reset session'}), 500


# Error handlers
@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    """
    Run the Flask development server
    - debug=True: Auto-reload on code changes (disable in production!)
    - host='0.0.0.0': Allow external connections
    - port=5000: Run on port 5000
    """
    print("🚀 Starting MindMate server...")
    print("📱 Visit: http://localhost:5000")
    print("⚠️  Remember: This is for educational purposes only")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
