"""
Model definitions for sentiment analysis.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path


class BaselineModel:
    """Baseline model using TF-IDF and simple classifiers."""
    
    def __init__(self, model_type='naive_bayes', max_features=10000):
        """
        Initialize baseline model.
        
        Args:
            model_type (str): Type of classifier ('naive_bayes', 'logistic', 'svm', 'random_forest')
            max_features (int): Maximum number of features for TF-IDF
        """
        self.model_type = model_type
        self.max_features = max_features
        
        # Create vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        # Create classifier based on type
        if model_type == 'naive_bayes':
            classifier = MultinomialNB(alpha=1.0)
        elif model_type == 'logistic':
            classifier = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'svm':
            classifier = SVC(kernel='linear', random_state=42, probability=True)
        elif model_type == 'random_forest':
            classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create pipeline
        self.model = Pipeline([
            ('tfidf', vectorizer),
            ('classifier', classifier)
        ])
    
    def train(self, X_train, y_train):
        """Train the model."""
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        print("Training completed!")
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        return self.model.predict_proba(X)
    
    def save(self, filepath):
        """Save model to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from disk."""
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from data_loader import load_imdb_data
    
    # Load data
    dataset = load_imdb_data()
    
    # Prepare data
    X = dataset['train']['review'].tolist()[:1000] if isinstance(dataset['train'], pd.DataFrame) else dataset['train']['text'][:1000]
    y = dataset['train']['label'].tolist()[:1000] if isinstance(dataset['train'], pd.DataFrame) else dataset['train']['label'][:1000]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = BaselineModel(model_type='naive_bayes')
    model.train(X_train, y_train)
    
    # Evaluate
    from sklearn.metrics import accuracy_score
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

