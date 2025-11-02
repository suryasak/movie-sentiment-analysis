"""
Example script demonstrating how to use the Movie Sentiment Analysis project.
"""

import sys
from pathlib import Path

# Add src to path (scripts are now in scripts/ directory)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import load_imdb_data
from src.models import BaselineModel
from src.utils import evaluate_model, print_metrics, plot_confusion_matrix
from sklearn.model_selection import train_test_split


def main():
    """Main example function."""
    print("="*60)
    print("Movie Sentiment Analysis - Example Usage")
    print("="*60)
    
    # Load dataset
    print("\n1. Loading IMDB dataset...")
    dataset = load_imdb_data()
    
    # Use a subset for faster execution (remove [:1000] for full dataset)
    print("\n2. Preparing data (using subset for demo)...")
    X = dataset['train']['text'][:1000]
    y = dataset['train']['label'][:1000]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"   Train samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Train model
    print("\n3. Training Naive Bayes model...")
    model = BaselineModel(model_type='naive_bayes', max_features=5000)
    model.train(X_train, y_train)
    
    # Evaluate
    print("\n4. Evaluating model...")
    metrics, y_pred = evaluate_model(model, X_test, y_test)
    print_metrics(metrics)
    
    # Save model
    print("\n5. Saving model...")
    Path('models').mkdir(exist_ok=True)
    model.save('models/example_naive_bayes.joblib')
    
    # Plot confusion matrix
    print("\n6. Generating confusion matrix...")
    Path('results').mkdir(exist_ok=True)
    plot_confusion_matrix(
        y_test, y_pred,
        save_path='results/example_confusion_matrix.png'
    )
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

