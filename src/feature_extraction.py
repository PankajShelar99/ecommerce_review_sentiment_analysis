# src/feature_extraction.py

from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def project_root():
    return Path(__file__).resolve().parents[1]


def models_dir():
    d = project_root() / "models"
    d.mkdir(exist_ok=True)
    return d


def extract_features(text_series, max_features=5000):
    """Extract TF-IDF features from text"""
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )
    X = vectorizer.fit_transform(text_series)
    return X, vectorizer


def save_vectorizer(vectorizer):
    """Save vectorizer to disk"""
    path = models_dir() / "tfidf_vectorizer.pkl"
    joblib.dump(vectorizer, path)
    print(f"✓ Vectorizer saved to: {path}")


def load_vectorizer():
    """Load vectorizer from disk"""
    path = models_dir() / "tfidf_vectorizer.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Vectorizer not found at {path}. Please train the model first.")
    return joblib.load(path)


if __name__ == "__main__":
    print("Testing feature_extraction.py...")
    try:
        # Test with sample data
        import pandas as pd

        sample_texts = pd.Series([
            "this is a great product",
            "terrible quality very bad",
            "okay product nothing special"
        ])

        X, vectorizer = extract_features(sample_texts)
        print(f"✓ Feature matrix shape: {X.shape}")
        print(f"✓ Number of features: {len(vectorizer.get_feature_names_out())}")

        # Test save/load
        save_vectorizer(vectorizer)
        loaded = load_vectorizer()
        print("✓ Vectorizer saved and loaded successfully")

    except Exception as e:
        print(f"✗ Error: {e}")