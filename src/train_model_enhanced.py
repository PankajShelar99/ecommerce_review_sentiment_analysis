# src/train_model_enhanced.py

import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import load_and_prepare
from feature_extraction import models_dir
import pandas as pd


def create_enhanced_vectorizer():
    """Create vectorizer with better parameters for detail capture"""
    return TfidfVectorizer(
        stop_words="english",
        max_features=8000,  # Increased for more detail
        ngram_range=(1, 3),  # Capture phrases up to 3 words
        min_df=2,  # Min document frequency
        max_df=0.8,  # Max document frequency
        sublinear_tf=True  # Apply sublinear tf scaling
    )


def train_enhanced_model(path=None):
    print("Loading and preparing dataset...")
    df = load_and_prepare(path)
    print(f"Dataset size: {len(df)} rows")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")

    # Enhanced vectorization
    print("\nCreating enhanced features...")
    vectorizer = create_enhanced_vectorizer()
    X = vectorizer.fit_transform(df["cleaned"])
    y = df["sentiment"]

    # Save vectorizer
    joblib.dump(vectorizer, models_dir() / "tfidf_vectorizer.pkl")
    print("Vectorizer saved with enhanced parameters")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train multiple models and select best
    print("\nTraining multiple models...")

    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight='balanced',
            solver='saga'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=50,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    }

    best_score = 0
    best_model = None
    best_name = ""
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)

        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")

        results[name] = {
            'model': model,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'predictions': test_preds
        }

        if test_acc > best_score:
            best_score = test_acc
            best_model = model
            best_name = name

    print(f"\n{'=' * 60}")
    print(f"Best Model: {best_name} with accuracy: {best_score:.4f}")
    print(f"{'=' * 60}")

    # Save best model
    joblib.dump(best_model, models_dir() / "sentiment_model.pkl")

    # Detailed evaluation
    best_preds = results[best_name]['predictions']
    print("\nClassification Report:")
    print(classification_report(y_test, best_preds))

    # Confusion Matrix
    cm = confusion_matrix(y_test, best_preds)
    print("\nConfusion Matrix:")
    print(cm)

    # Save confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(y.unique()),
                yticklabels=sorted(y.unique()))
    plt.title(f'Confusion Matrix - {best_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(models_dir().parent / 'static' / 'confusion_matrix.png', dpi=100)
    plt.close()

    # Feature importance for tree-based models
    if hasattr(best_model, 'feature_importances_'):
        feature_names = vectorizer.get_feature_names_out()
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[-20:]  # Top 20

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Most Important Features')
        plt.tight_layout()
        plt.savefig(models_dir().parent / 'static' / 'feature_importance.png', dpi=100)
        plt.close()

    # Save model metadata
    # Save model metadata
    metadata = {
        'model_name': best_name,
        'accuracy': float(best_score),
        'features': X.shape[1],
        'training_samples': X_train.shape[0],  # <- FIXED
        'test_samples': X_test.shape[0]  # <- FIXED
    }

    import json
    with open(models_dir() / "model_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\nModel training complete!")
    print(f"Model saved to: {models_dir() / 'sentiment_model.pkl'}")
    return best_model, vectorizer, y_test, best_preds


if __name__ == "__main__":
    train_enhanced_model()