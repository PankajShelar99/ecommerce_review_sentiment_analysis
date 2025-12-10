# src/visualizations.py - FIXED VERSION
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def ensure_plots_dir():
    """Ensure plots directory exists"""
    plots_dir = Path('static/plots')
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def create_all_visualizations(user_id):
    """
    Create all visualizations based on user's prediction data from database
    Returns: (vis_paths dict, summary_stats dict)
    """
    try:
        # Import inside function to avoid circular imports
        from app import db
        from src.database import Prediction

        plots_dir = ensure_plots_dir()

        # Get user's predictions from database
        predictions = Prediction.query.filter_by(user_id=user_id).all()

        if not predictions or len(predictions) == 0:
            # Return empty data if no predictions
            return {
                'sentiment_dist': None,
                'sentiment_bars': None,
                'timeline': None,
                'review_length': None,
                'length_by_sentiment': None
            }, {
                'total': 0,
                'positive': 0,
                'negative': 0,
                'neutral': 0,
                'positive_pct': 0,
                'negative_pct': 0,
                'neutral_pct': 0,
                'avg_review_length': 0
            }

        # Convert to DataFrame
        data = {
            'sentiment': [p.sentiment for p in predictions],
            'created_at': [p.created_at for p in predictions],
            'review_text': [p.review_text for p in predictions],
            'review_length': [len(str(p.review_text).split()) for p in predictions]
        }
        df = pd.DataFrame(data)

        # Calculate stats
        sentiment_counts = df['sentiment'].value_counts()
        total_reviews = len(df)

        summary_stats = {
            'total': total_reviews,
            'positive': int(sentiment_counts.get('positive', 0)),
            'negative': int(sentiment_counts.get('negative', 0)),
            'neutral': int(sentiment_counts.get('neutral', 0)),
            'avg_review_length': round(df['review_length'].mean(), 1),
            'positive_pct': round((sentiment_counts.get('positive', 0) / total_reviews) * 100,
                                  1) if total_reviews > 0 else 0,
            'negative_pct': round((sentiment_counts.get('negative', 0) / total_reviews) * 100,
                                  1) if total_reviews > 0 else 0,
            'neutral_pct': round((sentiment_counts.get('neutral', 0) / total_reviews) * 100,
                                 1) if total_reviews > 0 else 0
        }

        vis_paths = {}

        # Color scheme
        colors = {
            'positive': '#28a745',
            'negative': '#dc3545',
            'neutral': '#ffc107'
        }

        # 1. Sentiment Distribution Pie Chart
        try:
            plt.figure(figsize=(8, 8))
            sentiment_colors = [colors.get(s, '#6c757d') for s in sentiment_counts.index]

            plt.pie(sentiment_counts.values,
                    labels=[s.capitalize() for s in sentiment_counts.index],
                    autopct='%1.1f%%',
                    colors=sentiment_colors,
                    startangle=90,
                    textprops={'fontsize': 12, 'weight': 'bold'})
            plt.title('Sentiment Distribution', fontsize=16, fontweight='bold', pad=20)

            pie_path = plots_dir / f'sentiment_distribution_{user_id}.png'
            plt.savefig(pie_path, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close()
            vis_paths['sentiment_dist'] = f'/static/plots/sentiment_distribution_{user_id}.png'
        except Exception as e:
            print(f"Error creating pie chart: {e}")
            vis_paths['sentiment_dist'] = None

        # 2. Sentiment Bar Chart
        try:
            plt.figure(figsize=(10, 6))
            bar_colors = [colors.get(s, '#6c757d') for s in sentiment_counts.index]

            bars = plt.bar([s.capitalize() for s in sentiment_counts.index],
                           sentiment_counts.values,
                           color=bar_colors,
                           edgecolor='black',
                           linewidth=1.5)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{int(height)}',
                         ha='center', va='bottom', fontsize=12, fontweight='bold')

            plt.title('Sentiment Count', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Sentiment', fontsize=12, fontweight='bold')
            plt.ylabel('Count', fontsize=12, fontweight='bold')
            plt.grid(axis='y', alpha=0.3, linestyle='--')

            bar_path = plots_dir / f'sentiment_bars_{user_id}.png'
            plt.savefig(bar_path, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close()
            vis_paths['sentiment_bars'] = f'/static/plots/sentiment_bars_{user_id}.png'
        except Exception as e:
            print(f"Error creating bar chart: {e}")
            vis_paths['sentiment_bars'] = None

        # 3. Timeline (if enough data)
        if len(df) >= 3:
            try:
                df['date'] = pd.to_datetime(df['created_at']).dt.date
                timeline = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)

                plt.figure(figsize=(12, 6))

                for sentiment in timeline.columns:
                    plt.plot(timeline.index, timeline[sentiment],
                             marker='o', linewidth=2, markersize=8,
                             label=sentiment.capitalize(),
                             color=colors.get(sentiment, '#6c757d'))

                plt.title('Sentiment Over Time', fontsize=16, fontweight='bold', pad=20)
                plt.xlabel('Date', fontsize=12, fontweight='bold')
                plt.ylabel('Count', fontsize=12, fontweight='bold')
                plt.legend(title='Sentiment', fontsize=10)
                plt.grid(True, alpha=0.3, linestyle='--')
                plt.xticks(rotation=45)

                timeline_path = plots_dir / f'sentiment_timeline_{user_id}.png'
                plt.savefig(timeline_path, dpi=100, bbox_inches='tight', facecolor='white')
                plt.close()
                vis_paths['timeline'] = f'/static/plots/sentiment_timeline_{user_id}.png'
            except Exception as e:
                print(f"Error creating timeline: {e}")
                vis_paths['timeline'] = None
        else:
            vis_paths['timeline'] = None

        # 4. Review Length Distribution
        try:
            plt.figure(figsize=(10, 6))
            plt.hist(df['review_length'], bins=20, color='#007bff',
                     alpha=0.7, edgecolor='black', linewidth=1.5)

            # Add mean line
            mean_length = df['review_length'].mean()
            plt.axvline(mean_length, color='red', linestyle='--',
                        linewidth=2, label=f'Mean: {mean_length:.1f} words')

            plt.title('Review Length Distribution', fontsize=16, fontweight='bold', pad=20)
            plt.xlabel('Number of Words', fontsize=12, fontweight='bold')
            plt.ylabel('Frequency', fontsize=12, fontweight='bold')
            plt.legend(fontsize=10)
            plt.grid(axis='y', alpha=0.3, linestyle='--')

            length_path = plots_dir / f'review_length_{user_id}.png'
            plt.savefig(length_path, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close()
            vis_paths['review_length'] = f'/static/plots/review_length_{user_id}.png'
        except Exception as e:
            print(f"Error creating length distribution: {e}")
            vis_paths['review_length'] = None

        # 5. Sentiment by Review Length (Boxplot)
        if len(df) >= 10:
            try:
                plt.figure(figsize=(10, 6))

                # Create boxplot
                sentiment_data = []
                labels = []
                for sentiment in ['positive', 'negative', 'neutral']:
                    if sentiment in df['sentiment'].values:
                        subset = df[df['sentiment'] == sentiment]['review_length']
                        sentiment_data.append(subset)
                        labels.append(sentiment.capitalize())

                if sentiment_data:
                    box_plot = plt.boxplot(sentiment_data, labels=labels, patch_artist=True)

                    # Color the boxes
                    for patch, sentiment in zip(box_plot['boxes'], labels):
                        patch.set_facecolor(colors.get(sentiment.lower(), '#6c757d'))
                        patch.set_alpha(0.7)

                    plt.title('Review Length by Sentiment', fontsize=16, fontweight='bold', pad=20)
                    plt.xlabel('Sentiment', fontsize=12, fontweight='bold')
                    plt.ylabel('Number of Words', fontsize=12, fontweight='bold')
                    plt.grid(axis='y', alpha=0.3, linestyle='--')

                    length_sent_path = plots_dir / f'length_by_sentiment_{user_id}.png'
                    plt.savefig(length_sent_path, dpi=100, bbox_inches='tight', facecolor='white')
                    plt.close()
                    vis_paths['length_by_sentiment'] = f'/static/plots/length_by_sentiment_{user_id}.png'

            except Exception as e:
                print(f"Error creating length by sentiment: {e}")
                vis_paths['length_by_sentiment'] = None
        else:
            vis_paths['length_by_sentiment'] = None

        print(f"✓ Generated {len([v for v in vis_paths.values() if v])} visualizations for user {user_id}")
        return vis_paths, summary_stats

    except Exception as e:
        print(f"Error in create_all_visualizations: {e}")
        return {}, {'total': 0, 'positive': 0, 'negative': 0, 'neutral': 0, 'avg_review_length': 0}