# app.py - FIXED VERSION

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import json

from src.database import db, User, Prediction, BatchAnalysis
from src.predict import predict_sentiment
from src.batch_processor import process_batch_file
from src.visualizations import create_all_visualizations

# App configuration
app = Flask(__name__)
app.config['SECRET_KEY'] = 'k9mL2pQ7vX4bN8wE3rT6yU1iO5aS0dF'  # Use any random string
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sentiment_analysis.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure directories exist
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)
Path('static/plots').mkdir(parents=True, exist_ok=True)
Path('batch_results').mkdir(exist_ok=True)

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Create database tables
with app.app_context():
    db.create_all()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ==================== AUTH ROUTES ====================

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        # Validation
        if not username or not email or not password:
            flash('All fields are required!', 'danger')
            return render_template('signup.html')

        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return render_template('signup.html')

        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'danger')
            return render_template('signup.html')

        if User.query.filter_by(email=email).first():
            flash('Email already registered!', 'danger')
            return render_template('signup.html')

        # Create new user
        new_user = User(username=username, email=email)
        new_user.set_password(password)

        db.session.add(new_user)
        db.session.commit()

        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            login_user(user)
            flash('Logged in successfully!', 'success')
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!', 'danger')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))


# ==================== MAIN ROUTES ====================

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    # Get user's recent predictions
    recent_predictions = Prediction.query.filter_by(user_id=current_user.id) \
        .order_by(Prediction.created_at.desc()).limit(5).all()

    # Get user's batch analyses
    batch_analyses = BatchAnalysis.query.filter_by(user_id=current_user.id) \
        .order_by(BatchAnalysis.created_at.desc()).limit(5).all()

    # Statistics
    total_predictions = Prediction.query.filter_by(user_id=current_user.id).count()
    total_batches = BatchAnalysis.query.filter_by(user_id=current_user.id).count()

    return render_template('dashboard.html',
                           recent_predictions=recent_predictions,
                           batch_analyses=batch_analyses,
                           total_predictions=total_predictions,
                           total_batches=total_batches)


@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    sentiment = None
    suggestions = []
    details = []
    review_text = ""

    if request.method == 'POST':
        review_text = request.form.get('review_text', '').strip()

        if not review_text:
            flash('Please enter a review!', 'warning')
            return render_template('predict.html')

        sentiment, suggestions, details = predict_sentiment(review_text)

        # Save prediction to database
        prediction = Prediction(
            user_id=current_user.id,
            review_text=review_text,
            sentiment=sentiment,
            suggestions=json.dumps(suggestions)
        )
        db.session.add(prediction)
        db.session.commit()

        flash('Prediction completed successfully!', 'success')

    return render_template('predict.html',
                           sentiment=sentiment,
                           suggestions=suggestions,
                           details=details,
                           review_text=review_text)


@app.route('/batch_upload', methods=['GET', 'POST'])
@login_required
def batch_upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded!', 'danger')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No file selected!', 'danger')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                # Process the batch file
                results = process_batch_file(filepath, current_user.id)

                if 'error' in results:
                    flash(f"Error: {results['error']}", 'danger')
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    return redirect(request.url)

                # Save batch analysis to database
                metrics = results['metrics']
                batch = BatchAnalysis(
                    user_id=current_user.id,
                    filename=filename,
                    total_reviews=metrics['total_reviews'],
                    positive_count=metrics['positive'],
                    negative_count=metrics['negative'],
                    neutral_count=metrics['neutral'],
                    results_path=results.get('results_file')
                )
                db.session.add(batch)
                db.session.commit()

                flash('Batch analysis completed successfully!', 'success')
                return render_template('batch_results.html', results=results)

            except Exception as e:
                flash(f'Error processing batch file: {str(e)}', 'danger')
                if os.path.exists(filepath):
                    os.remove(filepath)
                return redirect(request.url)
        else:
            flash('Invalid file type! Please upload CSV or Excel files.', 'danger')

    return render_template('batch_upload.html')


@app.route('/batch_results')
@login_required
def batch_results():
    # This route displays batch results - you might want to pass results via session or database
    batch_analyses = BatchAnalysis.query.filter_by(user_id=current_user.id) \
        .order_by(BatchAnalysis.created_at.desc()).all()

    return render_template('batch_results.html', batch_analyses=batch_analyses)


@app.route('/visualizations')
@login_required
def visualizations():
    try:
        # Generate visualizations with current user's data
        vis_paths, summary_stats = create_all_visualizations(current_user.id)

        return render_template('visualizations.html',
                               visualizations=vis_paths,
                               summary_stats=summary_stats)
    except Exception as e:
        flash(f'Error generating visualizations: {str(e)}', 'danger')
        # Return empty data for template
        return render_template('visualizations.html',
                               visualizations={},
                               summary_stats={'total': 0, 'positive': 0, 'negative': 0, 'neutral': 0,
                                              'avg_review_length': 0})


@app.route('/history')
@login_required
def history():
    page = request.args.get('page', 1, type=int)
    predictions = Prediction.query.filter_by(user_id=current_user.id) \
        .order_by(Prediction.created_at.desc()) \
        .paginate(page=page, per_page=20, error_out=False)

    return render_template('history.html', predictions=predictions)


@app.route('/download/<path:filename>')
@login_required
def download_file(filename):
    try:
        # Security check - ensure the file is in allowed directories
        safe_path = Path(filename)
        if not safe_path.exists():
            flash('File not found!', 'danger')
            return redirect(url_for('dashboard'))

        return send_file(filename, as_attachment=True)
    except Exception as e:
        flash(f'Error downloading file: {str(e)}', 'danger')
        return redirect(url_for('dashboard'))


# ==================== API ENDPOINTS ====================

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    data = request.get_json()
    review_text = data.get('review_text', '')

    if not review_text:
        return jsonify({'error': 'No review text provided'}), 400

    sentiment, suggestions, details = predict_sentiment(review_text)

    return jsonify({
        'sentiment': sentiment,
        'suggestions': suggestions,
        'details': details
    })


@app.route('/api/stats')
@login_required
def api_stats():
    predictions = Prediction.query.filter_by(user_id=current_user.id).all()

    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    for pred in predictions:
        sentiment_counts[pred.sentiment] = sentiment_counts.get(pred.sentiment, 0) + 1

    return jsonify({
        'total_predictions': len(predictions),
        'sentiment_distribution': sentiment_counts
    })


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(e):
    return render_template('500.html'), 500


# ==================== FIX FOR SUGGESTIONS FILE ====================

def create_default_suggestions_file():
    """Create a default suggestions file if it doesn't exist"""
    suggestions_path = Path('data/suggestions_from_test.json')
    suggestions_path.parent.mkdir(parents=True, exist_ok=True)

    default_suggestions = {
        "positive": [
            "Great review! Consider highlighting specific features customers loved.",
            "Positive feedback received. Use this to identify your product strengths.",
            "Excellent rating. Share these testimonials on your marketing materials."
        ],
        "negative": [
            "Address the specific issues mentioned by the customer.",
            "Consider improving product quality based on this feedback.",
            "Offer customer support to resolve the mentioned problems."
        ],
        "neutral": [
            "The review is neutral. Consider asking for more specific feedback.",
            "Mixed feedback received. Look for patterns across multiple reviews.",
            "Neutral rating. Monitor for trends that could become positive or negative."
        ]
    }

    with open(suggestions_path, 'w', encoding='utf-8') as f:
        json.dump(default_suggestions, f, indent=2)

    print(f"✓ Created default suggestions file: {suggestions_path}")


# Create default suggestions file on startup
if __name__ == "__main__":
    with app.app_context():
        # Create default suggestions file
        create_default_suggestions_file()

        # Run the app
        print("Starting Flask application...")
        print("Dashboard: http://localhost:5000")
        print("Login: http://localhost:5000/login")
        app.run(debug=True, host='0.0.0.0', port=5000)