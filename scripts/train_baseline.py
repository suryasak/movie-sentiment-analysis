"""
Complete baseline model training script with results generation.
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
from src.utils import evaluate_model, print_metrics, plot_confusion_matrix
from src.preprocessing import clean_text
from sklearn.model_selection import train_test_split
import pandas as pd


def train_all_models(X_train, X_test, y_train, y_test, models_dir='models', results_dir='results'):
    """
    Train all baseline models and generate results.
    
    Args:
        X_train: Training texts
        X_test: Test texts
        y_train: Training labels
        y_test: Test labels
        models_dir: Directory to save models
        results_dir: Directory to save results
    """
    Path(models_dir).mkdir(exist_ok=True)
    Path(results_dir).mkdir(exist_ok=True)
    
    model_types = ['naive_bayes', 'logistic', 'svm', 'random_forest']
    results = {}
    
    print("\n" + "="*70)
    print("Training All Baseline Models")
    print("="*70)
    
    for model_type in model_types:
        print(f"\n{'='*70}")
        print(f"Training {model_type.upper().replace('_', ' ')} Model")
        print(f"{'='*70}")
        
        try:
            # Train model
            model = BaselineModel(model_type=model_type, max_features=10000)
            model.train(X_train, y_train)
            
            # Evaluate
            metrics, y_pred = evaluate_model(model, X_test, y_test)
            results[model_type] = metrics
            
            # Print metrics
            print_metrics(metrics)
            
            # Save model
            model_path = f'{models_dir}/baseline_{model_type}.joblib'
            model.save(model_path)
            
            # Plot confusion matrix
            cm_path = f'{results_dir}/confusion_matrix_{model_type}.png'
            plot_confusion_matrix(y_test, y_pred, save_path=cm_path)
            
            print(f"✓ {model_type} model saved and evaluated successfully!")
            
        except Exception as e:
            print(f"✗ Error training {model_type}: {e}")
            results[model_type] = {'error': str(e)}
            continue
    
    # Save results summary
    results_df = pd.DataFrame(results).T
    results_path = f'{results_dir}/baseline_results_summary.csv'
    results_df.to_csv(results_path)
    print(f"\n✓ Results summary saved to {results_path}")
    
    # Save JSON results
    results_json = {
        'timestamp': datetime.now().isoformat(),
        'models': results,
        'test_samples': len(y_test),
        'train_samples': len(X_train)
    }
    json_path = f'{results_dir}/baseline_results.json'
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"✓ Detailed results saved to {json_path}")
    
    return results


def main():
    """Main training function."""
    print("="*70)
    print("Movie Sentiment Analysis - Baseline Models Training")
    print("="*70)
    
    # Load dataset
    print("\n1. Loading IMDB dataset...")
    try:
        # Try Kaggle first
        data = load_imdb_data(use_kaggle=True, force_download=False)
        
        # Extract texts and labels (handle both DataFrame and dataset formats)
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
        print("Please check Kaggle API setup or use Hugging Face fallback")
        return
    
    # Use subset for faster training (remove limits for full dataset)
    print("\n2. Preparing data (using subset for demo - remove limits for full dataset)...")
    max_samples = 5000  # Set to None for full dataset
    if max_samples:
        X_train_full = X_train_full[:max_samples]
        y_train_full = y_train_full[:max_samples]
        X_test_full = X_test_full[:min(max_samples, len(X_test_full))]
        y_test_full = y_test_full[:min(max_samples, len(y_test_full))]
    
    # Optional: Preprocess texts (cleaning can take time, commented out for speed)
    print("\n3. Optional: Preprocessing texts (skipping for faster training)...")
    print("   To enable preprocessing, uncomment the preprocessing lines below")
    # X_train_full = [clean_text(text, remove_stopwords=False) for text in X_train_full]
    # X_test_full = [clean_text(text, remove_stopwords=False) for text in X_test_full]
    
    # Split train into train/val for final evaluation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )
    
    print(f"   Train samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    print(f"   Test samples: {len(X_test_full)}")
    
    # Train all models
    print("\n4. Training all baseline models...")
    results = train_all_models(
        X_train, X_val, y_train, y_val,
        models_dir='models',
        results_dir='results'
    )
    
    # Final evaluation on test set with best model
    print("\n5. Final evaluation on test set with best model...")
    best_model_type = max(results.keys(), 
                         key=lambda k: results[k].get('accuracy', 0) 
                         if 'accuracy' in results[k] else 0)
    
    if 'accuracy' in results[best_model_type]:
        print(f"\nBest model: {best_model_type}")
        print(f"Validation accuracy: {results[best_model_type]['accuracy']:.4f}")
        
        # Load and evaluate on test set
        model = BaselineModel(model_type=best_model_type)
        model.load(f'models/baseline_{best_model_type}.joblib')
        
        test_metrics, test_pred = evaluate_model(model, X_test_full, y_test_full)
        print("\n" + "="*70)
        print("FINAL TEST SET RESULTS")
        print("="*70)
        print_metrics(test_metrics)
        
        # Save test results
        test_results = {
            'model': best_model_type,
            'test_metrics': test_metrics,
            'timestamp': datetime.now().isoformat()
        }
        with open('results/test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        
        # Plot test confusion matrix
        plot_confusion_matrix(
            y_test_full, test_pred,
            save_path='results/test_confusion_matrix.png'
        )
    
    print("\n" + "="*70)
    print("Training completed successfully!")
    print("="*70)
    print("\nGenerated files:")
    print("  - models/: Trained model files")
    print("  - results/: Evaluation metrics and visualizations")
    print("="*70)


if __name__ == "__main__":
    main()

