# setup.py - Automated Setup Script

import os
import sys
import subprocess
from pathlib import Path


def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def create_directory_structure():
    """Create required directories"""
    print_header("Creating Directory Structure")

    directories = [
        'src',
        'templates',
        'static/plots',
        'data',
        'models',
        'uploads',
        'batch_results'
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}/")

    # Create __init__.py in src
    (Path('src') / '__init__.py').touch()
    print("✓ Created: src/__init__.py")


def check_python_version():
    """Check if Python version is adequate"""
    print_header("Checking Python Version")

    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        return False

    print("✓ Python version is compatible")
    return True


def install_requirements():
    """Install required packages"""
    print_header("Installing Required Packages")

    requirements = [
        'Flask==3.0.0',
        'Flask-SQLAlchemy==3.1.1',
        'Flask-Login==0.6.3',
        'Werkzeug==3.0.1',
        'pandas==2.1.4',
        'numpy==1.24.3',
        'scikit-learn==1.3.2',
        'joblib==1.3.2',
        'matplotlib==3.8.2',
        'seaborn==0.13.0',
        'nltk==3.8.1',
        'openpyxl==3.1.2'
    ]

    # Create requirements.txt
    with open('requirements.txt', 'w') as f:
        f.write('\n'.join(requirements))
    print("✓ Created requirements.txt")

    print("\nInstalling packages (this may take a few minutes)...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("\n✓ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("\n❌ Error installing packages. Please run: pip install -r requirements.txt")
        return False


def download_nltk_data():
    """Download required NLTK data"""
    print_header("Downloading NLTK Data")

    try:
        import nltk
        nltk.download('stopwords', quiet=True)
        print("✓ NLTK stopwords downloaded")
        return True
    except:
        print("❌ Error downloading NLTK data")
        return False


def check_data_file():
    """Check if training data exists"""
    print_header("Checking Training Data")

    data_path = Path('data/train.csv')

    if data_path.exists():
        print(f"✓ Found training data: {data_path}")

        # Check file size
        size_mb = data_path.stat().st_size / (1024 * 1024)
        print(f"  File size: {size_mb:.2f} MB")

        # Quick peek at first few lines
        try:
            with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [next(f) for _ in range(3)]
            print(f"  First line: {lines[0][:60]}...")
        except:
            pass

        return True
    else:
        print(f"❌ Training data not found: {data_path}")
        print("\nPlease add your training data CSV file to data/train.csv")
        print("Format should be: rating,summary,review")
        return False


def train_model():
    """Train the sentiment analysis model"""
    print_header("Training Model")

    if not Path('data/train.csv').exists():
        print("⚠️  Skipping model training - no training data found")
        return False

    print("Training enhanced model (this may take several minutes)...")
    print("This will:")
    print("  - Compare multiple ML algorithms")
    print("  - Select the best performing model")
    print("  - Generate evaluation metrics")
    print("  - Save model and vectorizer\n")

    try:
        # Import and run training
        sys.path.insert(0, 'src')
        from train_model import train_model as train_enhanced_model

        train_enhanced_model()
        print("\n✓ Model trained successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Error training model: {e}")
        print("You can manually train later: python src/train_model_enhanced.py")
        return False


def initialize_database():
    """Initialize the SQLite database"""
    print_header("Initializing Database")

    try:
        from app import app, db

        with app.app_context():
            db.create_all()

        print("✓ Database initialized successfully!")
        print("  Location: sentiment_analysis.db")
        return True
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        return False


def generate_visualizations():
    """Generate initial visualizations"""
    print_header("Generating Visualizations")

    if not Path('data/train.csv').exists():
        print("⚠️  Skipping visualizations - no training data found")
        return False

    print("Creating data visualizations...")

    try:
        sys.path.insert(0, 'src')
        from visualizations import create_all_visualizations

        vis_paths, stats = create_all_visualizations()

        print("\n✓ Visualizations created:")
        for name, path in vis_paths.items():
            print(f"  - {name}: {path}")

        return True
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        print("You can manually generate later: python src/visualizations.py")
        return False


def create_sample_data():
    """Create sample training data if none exists"""
    print_header("Creating Sample Data")

    data_path = Path('data/train.csv')

    if data_path.exists():
        print("⚠️  Training data already exists, skipping sample data creation")
        return True

    print("Creating sample training data...")

    sample_data = """rating,summary,review
5,Excellent phone,Amazing phone with great camera and battery life. Highly recommend!
5,Love it,Best purchase ever. Fast performance and beautiful display.
4,Good product,Good phone overall. Camera could be better in low light.
4,Satisfied,Works well for the price. Battery lasts all day.
3,Average,It's okay. Nothing special but gets the job done.
3,Decent,Fair product. Has some issues but acceptable.
2,Disappointed,Battery drains too quickly. Not worth the price.
2,Not great,Camera quality is poor. Heating issues during gaming.
1,Terrible,Broke after one week. Waste of money!
1,Very bad,Worst phone ever. Slow and laggy. Would not recommend."""

    with open(data_path, 'w', encoding='utf-8') as f:
        f.write(sample_data)

    print("✓ Sample data created at data/train.csv")
    print("  Note: This is minimal sample data for testing")
    print("  For production, replace with your actual dataset")

    return True


def print_next_steps():
    """Print instructions for next steps"""
    print_header("Setup Complete!")

    print("Next steps:")
    print("\n1. Start the application:")
    print("   python app.py")
    print("\n2. Open your browser:")
    print("   http://localhost:5000")
    print("\n3. Create an account and start analyzing!")
    print("\n" + "=" * 60)
    print("\nOptional:")
    print("- Add your training data to data/train.csv")
    print("- Retrain model: python src/train_model_enhanced.py")
    print("- Generate visualizations: python src/visualizations.py")
    print("\nFor help, check README.md")
    print("=" * 60 + "\n")


def main():
    """Main setup function"""
    print("\n" + "=" * 60)
    print("  SENTIMENT ANALYSIS SYSTEM - AUTOMATED SETUP")
    print("=" * 60)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Create directories
    create_directory_structure()

    # Install packages
    if not install_requirements():
        print("\n⚠️  Installation incomplete. Please install packages manually.")
        sys.exit(1)

    # Download NLTK data
    download_nltk_data()

    # Check for training data
    has_data = check_data_file()

    if not has_data:
        response = input("\nCreate sample training data? (y/n): ").lower()
        if response == 'y':
            create_sample_data()
            has_data = True

    # Train model
    if has_data:
        response = input("\nTrain model now? (recommended, y/n): ").lower()
        if response == 'y':
            train_model()

    # Initialize database
    initialize_database()

    # Generate visualizations
    if has_data:
        response = input("\nGenerate visualizations? (y/n): ").lower()
        if response == 'y':
            generate_visualizations()

    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)