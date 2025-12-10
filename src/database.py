# src/database.py

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()


class User(UserMixin, db.Model):
    """User model for authentication"""
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    predictions = db.relationship('Prediction', backref='user', lazy=True, cascade='all, delete-orphan')
    batch_analyses = db.relationship('BatchAnalysis', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Verify password"""
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'


class Prediction(db.Model):
    """Model for storing individual predictions"""
    __tablename__ = 'predictions'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    review_text = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=True)
    suggestions = db.Column(db.Text, nullable=True)  # JSON stored as text
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f'<Prediction {self.id} - {self.sentiment}>'


class BatchAnalysis(db.Model):
    """Model for storing batch analysis results"""
    __tablename__ = 'batch_analyses'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    filename = db.Column(db.String(200), nullable=False)
    total_reviews = db.Column(db.Integer, nullable=True)
    positive_count = db.Column(db.Integer, nullable=True)
    negative_count = db.Column(db.Integer, nullable=True)
    neutral_count = db.Column(db.Integer, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    results_path = db.Column(db.String(300), nullable=True)  # Path to results CSV

    def __repr__(self):
        return f'<BatchAnalysis {self.id} - {self.filename}>'


# Test function
if __name__ == "__main__":
    print("Testing database models...")

    # Test User
    user = User(username="testuser", email="test@example.com")
    user.set_password("testpass123")
    print(f"✓ User created: {user}")
    print(f"✓ Password check: {user.check_password('testpass123')}")

    # Test Prediction
    pred = Prediction(
        user_id=1,
        review_text="Great product!",
        sentiment="positive"
    )
    print(f"✓ Prediction created: {pred}")

    # Test BatchAnalysis
    batch = BatchAnalysis(
        user_id=1,
        filename="test.csv",
        total_reviews=100,
        positive_count=60,
        negative_count=30,
        neutral_count=10
    )
    print(f"✓ BatchAnalysis created: {batch}")

    print("\n✓ All database models working correctly!")