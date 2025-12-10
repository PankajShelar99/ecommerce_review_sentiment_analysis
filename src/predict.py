# src/predict.py - FIXED VERSION

import joblib
import json
import re
import os
import numpy as np
from pathlib import Path


# ------------------------------
# Path Configuration
# ------------------------------
def get_project_root():
    return Path(__file__).resolve().parents[1]


def get_models_dir():
    return get_project_root() / "models"


def get_data_dir():
    return get_project_root() / "data"


MODEL_PATH = get_models_dir() / "sentiment_model.pkl"
VECT_PATH = get_models_dir() / "tfidf_vectorizer.pkl"
SUGGESTIONS_JSON = get_data_dir() / "suggestions_from_test.json"


# ------------------------------
# Import clean_text function
# ------------------------------
def clean_text(text):
    """Clean text for prediction (standalone version)"""
    if not text:
        return ""

    import string
    try:
        from nltk.corpus import stopwords
        STOPWORDS = set(stopwords.words("english"))
    except:
        import nltk
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        STOPWORDS = set(stopwords.words("english"))

    text = str(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(words)


# ------------------------------
# Load Model & Vectorizer
# ------------------------------
print("Loading sentiment analysis model...")

if not MODEL_PATH.exists():
    print(f"⚠️ Model not found at: {MODEL_PATH}")
    print("Please train the model first: python src/train_model_enhanced.py")
    model = None
    vectorizer = None
else:
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECT_PATH)
        print("✓ Model and vectorizer loaded successfully")
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        model = None
        vectorizer = None

# ------------------------------
# Load Suggestions
# ------------------------------
if SUGGESTIONS_JSON.exists():
    with open(SUGGESTIONS_JSON, "r", encoding="utf-8") as f:
        suggestions_map = json.load(f)
    print(f"✓ Loaded suggestions from: {SUGGESTIONS_JSON}")
else:
    print(f"⚠️ Suggestions file not found: {SUGGESTIONS_JSON}")
    print("Run: python src/extract_issues.py")
    suggestions_map = {}

# ------------------------------
# Category Patterns
# ------------------------------
CATEGORY_PATTERNS = {
    'battery': [
        'battery', 'charge', 'charging', 'power', 'drain', 'backup',
        'battery life', 'fast charging', 'charger', 'mah', 'battery drain'
    ],
    'camera': [
        'camera', 'photo', 'picture', 'image', 'selfie', 'video',
        'low light', 'blur', 'focus', 'autofocus', 'megapixel', 'lens'
    ],
    'delivery': [
        'delivery', 'delivered', 'shipping', 'package', 'courier',
        'late', 'delayed', 'shipment', 'arrived', 'packaging'
    ],
    'performance': [
        'slow', 'lag', 'freeze', 'hang', 'performance', 'speed',
        'fast', 'smooth', 'responsive', 'processor', 'ram', 'memory'
    ],
    'heating': [
        'heat', 'hot', 'warm', 'overheat', 'temperature', 'heating issue'
    ],
    'audio': [
        'sound', 'audio', 'speaker', 'volume', 'music', 'call quality',
        'mic', 'microphone', 'bass', 'treble', 'headphone', 'loud'
    ],
    'screen': [
        'screen', 'display', 'brightness', 'touch', 'touchscreen',
        'resolution', 'pixel', 'amoled', 'lcd', 'scratch', 'responsive'
    ],
    'build_quality': [
        'build', 'quality', 'material', 'durable', 'sturdy', 'premium',
        'broken', 'crack', 'damage', 'scratch', 'fragile', 'plastic'
    ],
    'price': [
        'price', 'expensive', 'cheap', 'costly', 'value', 'worth',
        'money', 'affordable', 'overpriced', 'budget', 'cost'
    ],
    'connectivity': [
        'wifi', 'wi-fi', 'bluetooth', 'network', 'signal', '4g', '5g',
        'connect', 'disconnect', 'internet', 'hotspot', 'gps'
    ],
    'software': [
        'software', 'ui', 'interface', 'app', 'update', 'bug',
        'android', 'ios', 'system', 'glitch', 'crash', 'smooth ui'
    ]
}


# ------------------------------
# Aspect Detection
# ------------------------------
def detect_aspects(text):
    """Detect product aspects mentioned in review"""
    text_lower = text.lower()
    aspect_scores = {}

    for aspect, keywords in CATEGORY_PATTERNS.items():
        score = 0
        matched_keywords = []

        for keyword in keywords:
            if ' ' in keyword:
                if keyword in text_lower:
                    score += 2
                    matched_keywords.append(keyword)
            else:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, text_lower):
                    score += 1
                    matched_keywords.append(keyword)

        if score > 0:
            aspect_scores[aspect] = {
                'score': score,
                'keywords': matched_keywords
            }

    return sorted(aspect_scores.items(), key=lambda x: x[1]['score'], reverse=True)


# ------------------------------
# Confidence Score
# ------------------------------
def get_sentiment_confidence(model, X):
    """Get prediction confidence"""
    try:
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)[0]
            return float(max(probs))
        return None
    except:
        return None


# ------------------------------
# Generate Suggestions
# ------------------------------
def generate_suggestions(sentiment, aspects, text):
    """Generate context-aware suggestions"""
    suggestions = []
    details = []

    if sentiment in ("negative", "neutral"):
        # Aspect-specific suggestions
        for aspect, info in aspects[:5]:
            aspect_info = suggestions_map.get(aspect, {})
            aspect_suggestions = aspect_info.get("suggestions", [])

            for sug in aspect_suggestions[:2]:
                suggestions.append(f"[{aspect.replace('_', ' ').title()}] {sug}")

            keywords = info['keywords'][:3]
            if keywords:
                details.append(
                    f"[{aspect.replace('_', ' ').title()}] "
                    f"Detected issues: {', '.join(keywords)}"
                )

        if not suggestions:
            suggestions = [
                "Improve overall product quality and reliability",
                "Enhance customer experience and satisfaction",
                "Conduct quality assurance testing before shipping",
                "Gather and act on customer feedback regularly"
            ]
    else:
        suggestions = [
            "✓ Excellent review! Customer is satisfied",
            "✓ Continue maintaining this quality standard",
            "✓ Consider featuring this review for marketing"
        ]

    return suggestions, details


# ------------------------------
# Main Prediction Function - FIXED
# ------------------------------
def predict_sentiment(text):
    """
    Predict sentiment for a given review text

    Args:
        text (str): Review text to analyze

    Returns:
        tuple: (sentiment, suggestions, details)
    """
    # Check if model is loaded
    if model is None or vectorizer is None:
        return (
            "error",
            ["Model not loaded. Please train the model first."],
            ["Run: python src/train_model_enhanced.py"]
        )

    # Clean text
    cleaned = clean_text(text)

    if not cleaned or len(cleaned.strip()) < 3:
        return "neutral", ["Review too short for analysis"], []

    # Vectorize
    X = vectorizer.transform([cleaned])

    # Predict
    pred = model.predict(X)[0]
    confidence = get_sentiment_confidence(model, X)

    # FIXED: Convert prediction to string and normalize
    sentiment = str(pred).lower().strip()

    # Validate sentiment - must be one of the three valid options
    valid_sentiments = ['positive', 'negative', 'neutral']

    if sentiment not in valid_sentiments:
        # Fallback: Use keyword analysis
        text_lower = text.lower()

        # Strong positive indicators
        positive_words = ['excellent', 'amazing', 'love', 'best', 'perfect', 'wonderful',
                          'great', 'fantastic', 'awesome', 'outstanding', 'superb',
                          'useful', 'helpful', 'good', 'nice', 'recommend']

        # Strong negative indicators
        negative_words = ['terrible', 'awful', 'poor', 'worst', 'hate', 'disappointed',
                          'waste', 'useless', 'broken', 'slow', 'bad', 'horrible',
                          'never', 'worst', 'pathetic', 'fraud']

        # Count occurrences
        pos_count = sum(1 for word in positive_words if
                        f' {word} ' in f' {text_lower} ' or text_lower.startswith(word) or text_lower.endswith(word))
        neg_count = sum(1 for word in negative_words if
                        f' {word} ' in f' {text_lower} ' or text_lower.startswith(word) or text_lower.endswith(word))

        # Decide based on counts
        if pos_count > neg_count and pos_count > 0:
            sentiment = 'positive'
        elif neg_count > pos_count and neg_count > 0:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

    # Detect aspects
    aspects = detect_aspects(text)

    # Generate suggestions
    suggestions, details = generate_suggestions(sentiment, aspects, text)

    # Add confidence
    if confidence:
        details.insert(0, f"Prediction confidence: {confidence * 100:.1f}%")

    return sentiment, suggestions, details


# ------------------------------
# Batch Prediction
# ------------------------------
def batch_predict(reviews):
    """Predict sentiments for multiple reviews"""
    results = []

    for review in reviews:
        try:
            sentiment, suggestions, details = predict_sentiment(review)
            results.append({
                'review': review,
                'sentiment': sentiment,
                'suggestions': suggestions,
                'details': details
            })
        except Exception as e:
            results.append({
                'review': review,
                'sentiment': 'error',
                'suggestions': [],
                'details': [f"Error: {str(e)}"]
            })

    return results


# ------------------------------
# CLI Interface
# ------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("SENTIMENT ANALYSIS - PREDICTION TEST")
    print("=" * 60)

    if model is None:
        print("\n❌ Model not loaded!")
        print("Please run: python src/train_model_enhanced.py")
        exit(1)

    # Test predictions
    test_reviews = [
        "This product is so useful",
        "Amazing phone! Great camera and battery life. Highly recommend!",
        "Terrible product. Broke after one week. Battery drains fast.",
        "It's okay. Nothing special but works fine.",
        "Love the display and performance. Fast charging is awesome!",
        "Disappointed with camera quality. Low light photos are blurry."
    ]

    print("\nTesting predictions:\n")

    for i, review in enumerate(test_reviews, 1):
        print(f"Review {i}: {review[:60]}...")
        sentiment, suggestions, details = predict_sentiment(review)

        print(f"  → Sentiment: {sentiment.upper()}")
        print(f"  → Suggestions: {len(suggestions)} generated")
        if details:
            print(f"  → Details: {details[0]}")
        print()

    print("=" * 60)
    print("\n✓ All tests completed successfully!")
    print("\nTo use interactively, run the web app: python app.py")