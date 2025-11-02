"""
Quick training script that trains one model and generates results.
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to path (scripts are now in scripts/ directory)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import load_imdb_data
from src.models import BaselineModel
from src.utils import evaluate_model, print_metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from src.utils import plot_confusion_matrix


def main():
    """Main training function."""
    print("="*70)
    print("Movie Sentiment Analysis - Quick Training")
    print("="*70)
    
    # Load dataset
    print("\n1. Loading IMDB dataset...")
    try:
        data = load_imdb_data(use_kaggle=False, force_download=False)
        
        # Handle both DataFrame and dataset formats, and both 'review' and 'text' columns
        if isinstance(data['train'], pd.DataFrame):
            # Check which column exists
            if 'review' in data['train'].columns:
                X_train_full = data['train']['review'].tolist()
                X_test_full = data['test']['review'].tolist()
            elif 'text' in data['train'].columns:
                X_train_full = data['train']['text'].tolist()
                X_test_full = data['test']['text'].tolist()
            else:
                raise ValueError("No 'review' or 'text' column found in dataset")
            y_train_full = data['train']['label'].tolist()
            y_test_full = data['test']['label'].tolist()
        else:
            # Hugging Face dataset format
            if hasattr(data['train'], 'to_pandas'):
                train_df = data['train'].to_pandas()
                test_df = data['test'].to_pandas()
                X_train_full = train_df['text'].tolist()
                X_test_full = test_df['text'].tolist()
                y_train_full = train_df['label'].tolist()
                y_test_full = test_df['label'].tolist()
            else:
                X_train_full = data['train']['text']
                y_train_full = data['train']['label']
                X_test_full = data['test']['text']
                y_test_full = data['test']['label']
        
        print(f"   Loaded {len(X_train_full)} train and {len(X_test_full)} test samples")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Use subset for faster training
    print("\n2. Preparing data (using 2000 samples for demo)...")
    max_samples = 2000
    X_train_full = X_train_full[:max_samples]
    y_train_full = y_train_full[:max_samples]
    X_test_full = X_test_full[:min(500, len(X_test_full))]
    y_test_full = y_test_full[:min(500, len(y_test_full))]
    
    # Split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )
    
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test_full)}")
    
    # Train Naive Bayes model (fastest)
    print("\n3. Training Naive Bayes model...")
    Path('models').mkdir(exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    
    model = BaselineModel(model_type='naive_bayes', max_features=5000)
    model.train(X_train, y_train)
    
    # Evaluate
    print("\n4. Evaluating on validation set...")
    metrics, y_pred = evaluate_model(model, X_val, y_val)
    print_metrics(metrics)
    
    # Save model
    model.save('models/baseline_naive_bayes.joblib')
    
    # Plot confusion matrix
    print("\n5. Generating confusion matrix...")
    plot_confusion_matrix(
        y_val, y_pred,
        save_path='results/confusion_matrix_naive_bayes.png'
    )
    
    # Test set evaluation
    print("\n6. Evaluating on test set...")
    test_metrics, test_pred = evaluate_model(model, X_test_full, y_test_full)
    print("\n" + "="*70)
    print("TEST SET RESULTS")
    print("="*70)
    print_metrics(test_metrics)
    
    plot_confusion_matrix(
        y_test_full, test_pred,
        save_path='results/test_confusion_matrix.png'
    )
    
    # Save results
    results = {
        'model': 'naive_bayes',
        'validation_metrics': metrics,
        'test_metrics': test_metrics,
        'timestamp': datetime.now().isoformat(),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test_full)
    }
    
    with open('results/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70)
    print("\nGenerated files:")
    print("  - models/baseline_naive_bayes.joblib: Trained model")
    print("  - results/confusion_matrix_naive_bayes.png: Validation confusion matrix")
    print("  - results/test_confusion_matrix.png: Test confusion matrix")
    print("  - results/results.json: Detailed results")
    print("="*70)


if __name__ == "__main__":
    main()

