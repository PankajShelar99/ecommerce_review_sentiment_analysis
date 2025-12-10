# src/batch_processor.py - COMPLETE FIXED VERSION

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys
import os
from datetime import datetime
import json

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import your prediction function
try:
    from predict import predict_sentiment
except ImportError as e:
    print(f"⚠️ Could not import predict_sentiment: {e}")


    # Fallback function for testing
    def predict_sentiment(text):
        """
        Fallback sentiment analysis function
        Replace this with your actual model prediction
        """
        text_lower = str(text).lower()

        positive_words = ['good', 'great', 'excellent', 'awesome', 'love', 'amazing', 'fantastic', 'wonderful', 'best',
                          'perfect']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing', 'poor', 'waste',
                          'rubbish']

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            return "positive", ["Great review! Keep it up."], [
                {"aspect": "overall", "sentiment": "positive", "confidence": 0.8}]
        elif negative_count > positive_count:
            return "negative", ["Consider improvements in service quality."], [
                {"aspect": "overall", "sentiment": "negative", "confidence": 0.7}]
        else:
            return "neutral", ["Neutral feedback received."], [
                {"aspect": "overall", "sentiment": "neutral", "confidence": 0.6}]

try:
    from preprocess import clean_text
except ImportError:
    def clean_text(text):
        """Basic text cleaning fallback"""
        return str(text).strip().lower()


def get_project_root():
    """Get the project root directory"""
    return Path(__file__).resolve().parents[1]


def get_results_dir():
    """Get or create the batch results directory"""
    p = get_project_root() / "batch_results"
    p.mkdir(exist_ok=True)
    return p


def get_uploads_dir():
    """Get or create the uploads directory"""
    p = get_project_root() / "uploads"
    p.mkdir(exist_ok=True)
    return p


def detect_review_column(df):
    """
    Automatically detect the review/text column in the dataframe
    """
    # Common column names for reviews
    review_keywords = ['review', 'text', 'comment', 'feedback', 'content', 'message',
                       'description', 'opinion', 'response', 'answer']

    for col in df.columns:
        col_lower = str(col).lower()
        # Exact matches first
        if any(keyword == col_lower for keyword in review_keywords):
            return col
        # Partial matches
        if any(keyword in col_lower for keyword in review_keywords):
            return col

    # If no obvious review column, use the first string/object column
    for col in df.columns:
        if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
            # Check if it contains reasonable text data
            sample_value = str(df[col].iloc[0]) if len(df) > 0 else ""
            if len(sample_value) > 10:  # Reasonable text length
                return col

    # Fallback: use first column
    return df.columns[0]


def detect_label_column(df, review_col):
    """
    Detect label/sentiment column for evaluation
    """
    label_keywords = ['sentiment', 'label', 'rating', 'class', 'score', 'category',
                      'emotion', 'feeling', 'result', 'outcome']

    for col in df.columns:
        if col == review_col:
            continue

        col_lower = str(col).lower()
        # Exact matches first
        if any(keyword == col_lower for keyword in label_keywords):
            return col
        # Partial matches
        if any(keyword in col_lower for keyword in label_keywords):
            return col

    return None


def normalize_sentiment_label(label):
    """
    Normalize various sentiment labels to standard format
    """
    if pd.isna(label) or label is None:
        return "unknown"

    label_str = str(label).lower().strip()

    # Positive labels
    if label_str in ['positive', 'pos', '1', '4', '5', 'good', 'excellent', 'great', 'positive']:
        return "positive"

    # Negative labels
    elif label_str in ['negative', 'neg', '0', '1', '2', 'bad', 'poor', 'terrible', 'negative']:
        return "negative"

    # Neutral labels
    elif label_str in ['neutral', 'neut', '3', 'average', 'ok', 'okay', 'medium']:
        return "neutral"

    # Numeric ratings (assuming 1-5 scale)
    elif label_str.isdigit():
        rating = int(label_str)
        if rating >= 4:
            return "positive"
        elif rating <= 2:
            return "negative"
        else:
            return "neutral"

    return "unknown"


def validate_file(filepath):
    """
    Validate the uploaded file
    """
    if not os.path.exists(filepath):
        return False, "File does not exist"

    file_size = os.path.getsize(filepath)
    if file_size == 0:
        return False, "File is empty"

    if file_size > 16 * 1024 * 1024:  # 16MB limit
        return False, "File size exceeds 16MB limit"

    return True, "Valid"


def process_batch_file(filepath, user_id=None):
    """
    Process a CSV/Excel file with multiple reviews for sentiment analysis

    Args:
        filepath (str): Path to the uploaded file
        user_id (int): User ID for tracking

    Returns:
        dict: Processing results with metrics and file paths
    """
    print(f"\n{'=' * 60}")
    print(f"PROCESSING BATCH FILE: {Path(filepath).name}")
    print(f"User ID: {user_id}")
    print(f"{'=' * 60}")

    # Validate file
    is_valid, validation_msg = validate_file(filepath)
    if not is_valid:
        return {'error': f'File validation failed: {validation_msg}'}

    # Detect and load file
    file_ext = Path(filepath).suffix.lower()

    try:
        if file_ext == '.csv':
            # Try different encodings for CSV
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(filepath, encoding=encoding, on_bad_lines='skip')
                    print(f"✓ Successfully loaded CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue

            if df is None:
                return {'error': 'Could not read CSV file with any encoding'}

        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
            print(f"✓ Successfully loaded Excel file")
        else:
            return {'error': 'Unsupported file format. Use CSV or Excel files.'}
    except Exception as e:
        return {'error': f'Error reading file: {str(e)}'}

    # Basic dataframe validation
    if df.empty:
        return {'error': 'File contains no data'}

    print(f"✓ Loaded file with {len(df)} rows and {len(df.columns)} columns")
    print(f"✓ Columns: {list(df.columns)}")

    # Detect review column
    review_col = detect_review_column(df)
    print(f"✓ Using '{review_col}' column for reviews")

    # Detect label column for evaluation
    label_col = detect_label_column(df, review_col)
    has_labels = label_col is not None

    if has_labels:
        print(f"✓ Found labels in '{label_col}' column - will evaluate accuracy")
    else:
        print("ℹ No label column found - prediction only mode")

    # Process reviews
    results = []
    predictions_list = []
    actuals_list = []
    processing_errors = []

    print(f"\nProcessing {len(df)} reviews...")

    for idx, row in df.iterrows():
        try:
            review_text = str(row[review_col]).strip()

            # Skip empty reviews
            if not review_text or review_text.lower() in ['nan', 'none', 'null', '']:
                continue

            # Skip very short reviews (likely invalid)
            if len(review_text) < 3:
                continue

            # Predict sentiment
            sentiment, suggestions, details = predict_sentiment(review_text)

            result = {
                'review_id': idx + 1,
                'review_text': review_text[:500] + '...' if len(review_text) > 500 else review_text,
                'full_review': review_text,
                'predicted_sentiment': sentiment,
                'suggestions': suggestions,
                'details': details,
                'word_count': len(review_text.split())
            }

            # Handle actual labels if available
            if has_labels:
                actual_label = normalize_sentiment_label(row[label_col])
                result['actual_sentiment'] = actual_label

                if actual_label != "unknown":
                    actuals_list.append(actual_label)
                    predictions_list.append(sentiment)

            results.append(result)

            # Progress indicator
            if len(results) % 50 == 0:
                print(f"  Processed {len(results)} reviews...")

        except Exception as e:
            error_msg = f"Error processing review {idx}: {str(e)}"
            processing_errors.append(error_msg)
            print(f"  ⚠️ {error_msg}")
            continue

    print(f"✓ Successfully processed {len(results)} reviews")

    if processing_errors:
        print(f"⚠️  {len(processing_errors)} reviews had processing errors")

    if len(results) == 0:
        return {'error': 'No valid reviews found in file'}

    # Calculate metrics
    sentiment_counts = pd.Series([r['predicted_sentiment'] for r in results]).value_counts().to_dict()
    total_processed = len(results)

    metrics = {
        'total_reviews': total_processed,
        'positive': sentiment_counts.get('positive', 0),
        'negative': sentiment_counts.get('negative', 0),
        'neutral': sentiment_counts.get('neutral', 0),
        'positive_pct': f"{sentiment_counts.get('positive', 0) / total_processed * 100:.1f}%",
        'negative_pct': f"{sentiment_counts.get('negative', 0) / total_processed * 100:.1f}%",
        'neutral_pct': f"{sentiment_counts.get('neutral', 0) / total_processed * 100:.1f}%",
        'avg_word_count': round(np.mean([r['word_count'] for r in results]), 1),
        'processing_errors': len(processing_errors)
    }

    print(f"\n📊 Sentiment Distribution:")
    print(f"  Positive: {metrics['positive']} ({metrics['positive_pct']})")
    print(f"  Negative: {metrics['negative']} ({metrics['negative_pct']})")
    print(f"  Neutral:  {metrics['neutral']} ({metrics['neutral_pct']})")
    print(f"  Average words per review: {metrics['avg_word_count']}")

    # Evaluation metrics (if labels available)
    evaluation_metrics = None
    confusion_matrix_path = None

    if has_labels and len(actuals_list) > 0 and len(predictions_list) > 0:
        print("\n📈 Calculating evaluation metrics...")

        try:
            accuracy = accuracy_score(actuals_list, predictions_list)

            # Get unique labels that exist in both actual and predicted
            unique_labels = sorted(list(set(actuals_list + predictions_list)))

            if len(unique_labels) > 1:  # Need at least 2 classes for proper classification report
                report = classification_report(actuals_list, predictions_list,
                                               output_dict=True, zero_division=0,
                                               labels=unique_labels)

                evaluation_metrics = {
                    'accuracy': f"{accuracy * 100:.2f}%",
                    'precision': f"{report['weighted avg']['precision']:.3f}",
                    'recall': f"{report['weighted avg']['recall']:.3f}",
                    'f1_score': f"{report['weighted avg']['f1-score']:.3f}",
                    'classification_report': report
                }

                print(f"  ✅ Accuracy:  {evaluation_metrics['accuracy']}")
                print(f"  ✅ Precision: {evaluation_metrics['precision']}")
                print(f"  ✅ Recall:    {evaluation_metrics['recall']}")
                print(f"  ✅ F1-Score:  {evaluation_metrics['f1_score']}")

                # Generate confusion matrix plot
                cm = confusion_matrix(actuals_list, predictions_list, labels=unique_labels)

                results_dir = get_results_dir()
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                cm_filename = f'confusion_matrix_{timestamp}.png'
                cm_path = results_dir / cm_filename

                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=[l.title() for l in unique_labels],
                            yticklabels=[l.title() for l in unique_labels],
                            cbar_kws={'label': 'Count'})
                plt.title('Confusion Matrix - Batch Analysis', fontsize=16, fontweight='bold', pad=20)
                plt.ylabel('Actual Sentiment', fontsize=12, fontweight='bold')
                plt.xlabel('Predicted Sentiment', fontsize=12, fontweight='bold')
                plt.tight_layout()
                plt.savefig(cm_path, dpi=100, bbox_inches='tight')
                plt.close()

                confusion_matrix_path = f'batch_results/{cm_filename}'
                print(f"✓ Confusion matrix saved: {cm_filename}")
            else:
                print("  ⚠️  Not enough unique labels for evaluation metrics")
                evaluation_metrics = {'accuracy': f"{accuracy * 100:.2f}%", 'note': 'Single class detected'}

        except Exception as e:
            print(f"  ⚠️  Error calculating evaluation metrics: {e}")
            evaluation_metrics = {'error': str(e)}

    # Save results to CSV
    results_dir = get_results_dir()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_filename = f'batch_results_{timestamp}.csv'
    results_path = results_dir / results_filename

    try:
        # Prepare results for CSV export
        export_data = []
        for result in results:
            export_row = {
                'review_id': result['review_id'],
                'review_text': result['full_review'],
                'predicted_sentiment': result['predicted_sentiment'],
                'suggestions': ' | '.join(result['suggestions']),
                'word_count': result['word_count']
            }

            if has_labels:
                export_row['actual_sentiment'] = result.get('actual_sentiment', 'unknown')

            export_data.append(export_row)

        export_df = pd.DataFrame(export_data)
        export_df.to_csv(results_path, index=False, encoding='utf-8')
        print(f"✓ Results saved: {results_filename}")
    except Exception as e:
        print(f"  ⚠️  Error saving results CSV: {e}")
        results_filename = None

    # Generate sentiment distribution chart
    chart_filename = f'sentiment_distribution_{timestamp}.png'
    chart_path = results_dir / chart_filename

    try:
        plt.figure(figsize=(12, 6))

        # Prepare data for plotting
        sentiments = ['negative', 'neutral', 'positive']
        counts = [sentiment_counts.get(s, 0) for s in sentiments]
        colors = ['#e74c3c', '#f39c12', '#2ecc71']  # Red, Yellow, Green

        # Create bar chart
        bars = plt.bar(sentiments, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

        plt.title('Batch Analysis - Sentiment Distribution', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Sentiment', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Reviews', fontsize=12, fontweight='bold')
        plt.grid(axis='y', alpha=0.3, linestyle='--')

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            if count > 0:
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                         str(count), ha='center', va='bottom', fontweight='bold', fontsize=11)

        plt.tight_layout()
        plt.savefig(chart_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"✓ Chart saved: {chart_filename}")
    except Exception as e:
        print(f"  ⚠️  Error creating chart: {e}")
        chart_filename = None

    # Generate detailed statistics
    detailed_stats = {
        'processing_summary': {
            'total_rows_in_file': len(df),
            'successfully_processed': len(results),
            'processing_errors': len(processing_errors),
            'success_rate': f"{(len(results) / len(df)) * 100:.1f}%"
        },
        'sentiment_breakdown': sentiment_counts,
        'text_statistics': {
            'average_review_length': metrics['avg_word_count'],
            'total_words_analyzed': sum([r['word_count'] for r in results]),
            'longest_review': max([r['word_count'] for r in results]) if results else 0,
            'shortest_review': min([r['word_count'] for r in results]) if results else 0
        }
    }

    print(f"\n{'=' * 60}")
    print("✅ BATCH PROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"📁 Results file: {results_filename}")
    print(f"📊 Chart file: {chart_filename}")
    print(f"📈 Evaluation: {'Available' if evaluation_metrics else 'Not available'}")
    print(f"{'=' * 60}\n")

    # Return comprehensive results
    return {
        'success': True,
        'results': results[:100],  # First 100 for display
        'metrics': metrics,
        'evaluation_metrics': evaluation_metrics,
        'detailed_stats': detailed_stats,
        'results_file': f'batch_results/{results_filename}' if results_filename else None,
        'chart_path': f'batch_results/{chart_filename}' if chart_filename else None,
        'confusion_matrix_path': confusion_matrix_path,
        'has_evaluation': has_labels and evaluation_metrics is not None,
        'processing_errors': processing_errors[:10]  # First 10 errors for display
    }


def get_batch_summary(results):
    """Generate summary from batch results"""
    if not results:
        return {}

    sentiments = [r['predicted_sentiment'] for r in results]
    sentiment_counts = pd.Series(sentiments).value_counts()

    # Calculate percentages
    total = len(results)
    percentages = {sentiment: f"{(count / total) * 100:.1f}%"
                   for sentiment, count in sentiment_counts.items()}

    summary = {
        'total': total,
        'distribution': sentiment_counts.to_dict(),
        'percentages': percentages,
        'most_common': sentiment_counts.index[0] if len(sentiment_counts) > 0 else 'N/A',
        'most_common_count': sentiment_counts.iloc[0] if len(sentiment_counts) > 0 else 0
    }

    return summary


def cleanup_old_files(max_age_hours=24):
    """
    Clean up old batch result files to save disk space
    """
    try:
        results_dir = get_results_dir()
        current_time = datetime.now()

        for file_path in results_dir.glob('*'):
            if file_path.is_file():
                file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_age.total_seconds() > max_age_hours * 3600:
                    file_path.unlink()
                    print(f"Cleaned up old file: {file_path.name}")
    except Exception as e:
        print(f"Error during cleanup: {e}")


if __name__ == "__main__":
    print("Testing batch_processor.py...")

    # Create a test CSV file
    test_data = {
        'review_text': [
            "This product is amazing and works perfectly!",
            "Terrible quality, very disappointed.",
            "It's okay, nothing special but gets the job done.",
            "Absolutely love this! Best purchase ever.",
            "Waste of money, broke after one week.",
            "Good value for the price.",
            "Horrible customer service and poor quality.",
            "Excellent product, highly recommended!",
            "Average product, meets basic needs.",
            "Outstanding quality and fast delivery!"
        ],
        'sentiment': [
            'positive', 'negative', 'neutral', 'positive', 'negative',
            'positive', 'negative', 'positive', 'neutral', 'positive'
        ]
    }

    test_df = pd.DataFrame(test_data)
    test_file = get_uploads_dir() / 'test_batch.csv'
    test_df.to_csv(test_file, index=False)

    print(f"Created test file: {test_file}")

    # Test the processor
    results = process_batch_file(str(test_file), user_id=999)

    if 'error' in results:
        print(f"❌ Error: {results['error']}")
    else:
        print("✅ Batch processing test completed successfully!")
        print(f"Processed {results['metrics']['total_reviews']} reviews")

    # Cleanup test file
    if test_file.exists():
        test_file.unlink()