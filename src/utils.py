"""
Utility functions for evaluation and visualization.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import numpy as np


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model with predict() method
        X_test: Test features
        y_test: True labels
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }
    
    return metrics, y_pred


def plot_confusion_matrix(y_true, y_pred, labels=['Negative', 'Positive'], save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    if save_path:
        plt.close()  # Close figure when saving (for batch processing)
    else:
        plt.show()


def print_metrics(metrics):
    """Print evaluation metrics in a formatted way."""
    print("\n" + "="*50)
    print("Model Evaluation Metrics")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    print("="*50 + "\n")


def plot_training_history(history, save_path=None):
    """
    Plot training history (for models with training history).
    
    Args:
        history: Dictionary with training metrics
        save_path: Path to save the plot
    """
    if not history:
        print("No training history available.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    if 'accuracy' in history:
        axes[0].plot(history['accuracy'], label='Train Accuracy')
        if 'val_accuracy' in history:
            axes[0].plot(history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
    
    # Plot loss
    if 'loss' in history:
        axes[1].plot(history['loss'], label='Train Loss')
        if 'val_loss' in history:
            axes[1].plot(history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
        plt.close()  # Close figure when saving
    else:
        plt.show()

