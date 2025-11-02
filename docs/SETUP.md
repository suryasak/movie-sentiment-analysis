# Setup Guide

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Setup Kaggle (Optional):**
   - If you want to use Kaggle dataset, run: `python scripts/setup_kaggle.py`
   - Follow the instructions to set up Kaggle API credentials
   - If Kaggle setup is not done, the code will automatically use Hugging Face dataset

3. **Download NLTK data:**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Kaggle Setup (Optional)

The project supports downloading IMDB dataset from Kaggle, but will fall back to Hugging Face if Kaggle is not configured.

### To use Kaggle:

1. Create a Kaggle account at https://www.kaggle.com/
2. Go to Account Settings → API → Create New API Token
3. Download `kaggle.json`
4. Place it in `~/.kaggle/kaggle.json`
5. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`
6. Install kaggle: `pip install kaggle`

### Alternative: Use Hugging Face (No Setup Required)

The code automatically falls back to Hugging Face if Kaggle is not available. No additional setup needed!

## Running the Project

### Option 1: Run Complete Training Pipeline

```bash
# Quick training (recommended for first run)
python scripts/quick_train.py

# Full training with all models
python scripts/train_baseline.py
```

This will:
- Download dataset (Kaggle or Hugging Face)
- Train baseline models
- Generate results and visualizations
- Save models to `models/` directory

### Option 2: Use Jupyter Notebook

```bash
jupyter notebook notebooks/01_baseline.ipynb
```

### Option 3: Run Example Script

```bash
python scripts/example_usage.py
```

## Troubleshooting

### ModuleNotFoundError

If you get import errors, make sure you've activated your virtual environment:
```bash
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### Kaggle API Errors

If Kaggle download fails:
- Check that `kaggle.json` is in the correct location
- Verify API credentials are correct
- The code will automatically use Hugging Face as fallback

### NLTK Data Errors

Run this once:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

