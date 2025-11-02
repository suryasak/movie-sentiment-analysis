# Movie Sentiment Analysis

A comprehensive machine learning project for analyzing sentiment in movie reviews using various approaches including baseline models (Naive Bayes, Logistic Regression, SVM) and transformer-based models (BERT).

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements multiple approaches for sentiment analysis on the IMDB movie reviews dataset:

1. **Baseline Models**: Traditional ML approaches using TF-IDF features and classifiers
   - Naive Bayes
   - Logistic Regression
   - Support Vector Machine (SVM)
   - Random Forest

2. **Deep Learning**: Fine-tuned BERT models for state-of-the-art performance

## ✨ Features

- Automated data loading and preprocessing
- Multiple baseline model implementations
- BERT fine-tuning capabilities
- Comprehensive evaluation metrics
- Visualization tools for model performance
- Modular and extensible codebase

## 📁 Project Structure

> 📖 **For detailed directory structure, see [DIRECTORY_STRUCTURE.md](DIRECTORY_STRUCTURE.md)**

```
movie-sentiment-analysis/
├── data/                  # Dataset storage
│   └── imdb/             # IMDB dataset (downloaded from Kaggle/Hugging Face)
├── docs/                  # Documentation
│   ├── SETUP.md          # Setup instructions
│   ├── CONTRIBUTING.md    # Contribution guidelines
│   └── PROJECT_SUMMARY.md # Project overview
├── models/                # Saved trained model files
├── notebooks/             # Jupyter notebooks for experiments
│   ├── 01_baseline.ipynb # Baseline model experiments
│   └── 02_bert_finetune.ipynb # BERT fine-tuning experiments
├── results/               # Experimental results and visualizations
├── scripts/                # Executable scripts
│   ├── quick_train.py     # Quick training script (demo)
│   ├── train_baseline.py  # Full baseline training pipeline
│   ├── example_usage.py   # Usage examples
│   └── setup_kaggle.py   # Kaggle API setup helper
├── src/                   # Source code modules
│   ├── __init__.py
│   ├── data_loader.py     # Dataset loading utilities
│   ├── preprocessing.py   # Text preprocessing functions
│   ├── models.py          # Model definitions
│   └── utils.py           # Evaluation and visualization utilities
├── .github/                # GitHub workflows
│   └── workflows/
├── .gitignore            # Git ignore file
├── requirements.txt      # Python dependencies
├── setup.py              # Package setup script
└── README.md             # Project documentation
```

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip or conda

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/movie-sentiment-analysis.git
cd movie-sentiment-analysis
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK data (if needed):
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## 📊 Usage

### Quick Start - Run Complete Pipeline

```bash
# Quick training with one model (recommended for first run)
python scripts/quick_train.py

# Full training with all baseline models
python scripts/train_baseline.py

# Example usage demonstration
python scripts/example_usage.py

# Setup Kaggle API (optional)
python scripts/setup_kaggle.py
```

### Loading the Dataset

The IMDB dataset can be loaded using the data loader module:

```python
from src.data_loader import load_imdb_data

# Load from Kaggle (falls back to Hugging Face if not available)
dataset = load_imdb_data(use_kaggle=True)

# Or force Hugging Face
dataset = load_imdb_data(use_kaggle=False)

# Access train and test splits (returns pandas DataFrames)
train_data = dataset['train']  # DataFrame with 'review'/'text' and 'label' columns
test_data = dataset['test']

# Extract texts and labels
X_train = train_data['review'].tolist()  # or 'text' for Hugging Face
y_train = train_data['label'].tolist()
```

Or using the Jupyter notebook:

```bash
jupyter notebook notebooks/01_baseline.ipynb
```

### Training Baseline Models

```python
from src.models import BaselineModel
from src.utils import evaluate_model, print_metrics
from sklearn.model_selection import train_test_split

# Load data
dataset = load_imdb_data()
X = dataset['train']['text']
y = dataset['train']['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = BaselineModel(model_type='naive_bayes')
model.train(X_train, y_train)

# Evaluate
metrics, y_pred = evaluate_model(model, X_test, y_test)
print_metrics(metrics)

# Save model
model.save('models/baseline_naive_bayes.joblib')
```

### Text Preprocessing

```python
from src.preprocessing import clean_text, preprocess_dataset

# Clean a single text
cleaned = clean_text("This movie is great!", remove_stopwords=True)

# Preprocess entire dataset
cleaned_dataset = preprocess_dataset(dataset, remove_stopwords=True)
```

### Evaluation and Visualization

```python
from src.utils import plot_confusion_matrix

# Plot confusion matrix
plot_confusion_matrix(y_test, y_pred, save_path='results/cm.png')
```

## 📚 Dataset

This project supports downloading the IMDB Movie Reviews Dataset from two sources:

1. **Kaggle** (Primary): [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
   - 50,000 movie reviews (CSV format)
   - Automatically splits into train/test sets
   
2. **Hugging Face** (Fallback): [IMDB Dataset](https://huggingface.co/datasets/imdb)
   - **Train**: 25,000 movie reviews
   - **Test**: 25,000 movie reviews
   - Automatically used if Kaggle is not configured

**Labels**: Binary (0 = negative, 1 = positive)  
**Features**: Raw text reviews

The dataset is automatically downloaded and cached in `data/imdb/` on first use.  
If Kaggle API is not set up, the code automatically falls back to Hugging Face.

## 🤖 Models

### Baseline Models

1. **Naive Bayes**: Fast and efficient for text classification
2. **Logistic Regression**: Simple linear classifier
3. **SVM**: Support Vector Machine with linear kernel
4. **Random Forest**: Ensemble method with multiple decision trees

All baseline models use TF-IDF vectorization for feature extraction.

### Deep Learning Models

- **BERT**: Fine-tuned BERT models for improved accuracy
- Implementation details in `notebooks/02_bert_finetune.ipynb`

## 📈 Results

Model performance results and visualizations will be saved in the `results/` directory.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Hugging Face for providing the datasets library and IMDB dataset
- scikit-learn for machine learning utilities
- NLTK for natural language processing tools

---

**Author**: [Your Name]  
**Last Updated**: 2024

