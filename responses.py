"""
Response Generation – MindMate (FINAL)
✔ Extremely low repetition
✔ Component mixing
✔ Risk-aware
"""

import random
from typing import List

# =========================
# RISK SCORING
# =========================

def calculate_risk_score(text: str, emotion: str) -> int:
    text = text.lower()

    if any(k in text for k in [
        "suicide", "kill myself", "want to die", "end my life"
    ]):
        return 3

    if any(k in text for k in [
        "hopeless", "worthless", "empty", "nobody cares"
    ]):
        return 2

    if emotion in ["sadness", "anxiety"]:
        return 1

    return 0


# =========================
# RESPONSE COMPONENTS
# =========================

OPENERS = {
    "sadness": [
        "It sounds like you’ve been carrying a quiet weight.",
        "There’s a deep heaviness in what you’re saying.",
        "That emotional tiredness really comes through."
    ],
    "anxiety": [
        "Your thoughts seem to be racing a bit.",
        "That constant unease can be exhausting.",
        "It sounds mentally overwhelming."
    ],
    "anger": [
        "Something clearly crossed a boundary for you.",
        "That frustration feels intense.",
        "I hear a lot of bottled-up energy."
    ],
    "joy": [
        "That sounds genuinely uplifting!",
        "There’s a nice lightness in your words.",
        "That’s great to hear."
    ],
    "neutral": [
        "I’m here with you.",
        "Thanks for sharing.",
        "Go on — I’m listening."
    ],
    "severe_distress": [
        "I’m really concerned about your safety.",
        "What you’re describing sounds extremely painful.",
        "I’m glad you didn’t keep this to yourself."
    ]
}

VALIDATIONS = [
    "Your feelings are valid.",
    "You’re not weak for feeling this way.",
    "Anyone in your situation might feel similarly.",
    "What you’re experiencing makes sense."
]

FOLLOW_UPS = [
    "Do you want to share what led to this?",
    "What’s been weighing on you the most?",
    "Has this been building up for a while?",
    "Would talking more help right now?"
]

SUPPORTS = [
    "You don’t have to figure everything out today.",
    "Taking things one step at a time is enough.",
    "I’m here to support you.",
    "You’re not facing this alone."
]


# =========================
# RESPONSE MEMORY (ANTI-REPEAT)
# =========================

LAST_RESPONSES = []


def generate_response(emotion: str, risk_score: int, history: List[str] = None) -> str:
    for _ in range(15):  # try 15 combinations to avoid repeats
        response = " ".join([
            random.choice(OPENERS.get(emotion, OPENERS["neutral"])),
            random.choice(VALIDATIONS),
            random.choice(FOLLOW_UPS),
            random.choice(SUPPORTS)
        ])

        if response not in LAST_RESPONSES:
            LAST_RESPONSES.append(response)
            LAST_RESPONSES[:] = LAST_RESPONSES[-50:]  # memory limit
            break

    if history and history.count(emotion) >= 3:
        response += "\n\nI’ve noticed this feeling keeps returning. That can be really hard."

    if risk_score >= 3:
        response += (
            "\n\n🚨 Please reach out now:\n"
            "• Suicide Prevention Lifeline: 988\n"
            "• Crisis Text Line: Text HOME to 741741\n"
            "• Local emergency services"
        )

    return response

