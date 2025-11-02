"""
Data loading utilities for IMDB dataset.
"""

import pandas as pd
from pathlib import Path
import os
import zipfile

# Optional Kaggle import
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False


def download_from_kaggle(dataset_name='lakshmi25npathi/imdb-dataset-of-50k-movie-reviews', 
                         data_dir='data/imdb', force_download=False):
    """
    Download IMDB dataset from Kaggle.
    
    Args:
        dataset_name (str): Kaggle dataset identifier
        data_dir (str): Directory where dataset is stored
        force_download (bool): If True, re-download dataset even if it exists
        
    Returns:
        str: Path to the downloaded data directory
    """
    data_path = Path(data_dir)
    csv_path = data_path / 'IMDB Dataset.csv'
    
    # Check if already downloaded
    if csv_path.exists() and not force_download:
        print(f"Dataset already exists at {csv_path}")
        return str(data_path)
    
    # Create directory
    os.makedirs(data_dir, exist_ok=True)
    
    if not KAGGLE_AVAILABLE:
        raise ImportError("Kaggle package not installed. Install with: pip install kaggle")
    
    try:
        # Authenticate and download
        print("Authenticating with Kaggle API...")
        api = KaggleApi()
        api.authenticate()
        
        print(f"Downloading dataset '{dataset_name}' from Kaggle...")
        api.dataset_download_files(dataset_name, path=data_dir, unzip=True)
        
        print(f"Dataset downloaded successfully to {data_dir}")
        
        # Find the CSV file (it might be in a subdirectory after unzipping)
        csv_files = list(data_path.glob('*.csv'))
        if csv_files:
            # Move CSV to data_dir if it's in a subdirectory
            for csv_file in csv_files:
                if csv_file.parent != data_path:
                    import shutil
                    shutil.move(str(csv_file), str(data_path / csv_file.name))
        
    except Exception as e:
        print(f"Error downloading from Kaggle: {e}")
        print("\nPlease ensure you have:")
        print("1. Kaggle API credentials (kaggle.json) in ~/.kaggle/")
        print("2. Accepted the dataset's terms of use on Kaggle")
        print("3. Installed kaggle package: pip install kaggle")
        raise
    
    return str(data_path)


def load_imdb_data_from_kaggle(data_dir='data/imdb', force_download=False, 
                               use_huggingface_fallback=True):
    """
    Load IMDB dataset from Kaggle CSV file.
    
    Args:
        data_dir (str): Directory where dataset is stored
        force_download (bool): If True, re-download dataset even if it exists
        use_huggingface_fallback (bool): Fallback to Hugging Face if Kaggle fails
        
    Returns:
        dict: Dictionary with 'train' and 'test' dataframes
    """
    data_path = Path(data_dir)
    csv_path = data_path / 'IMDB Dataset.csv'
    
    # Try to download from Kaggle
    if not csv_path.exists() or force_download:
        try:
            download_from_kaggle(data_dir=data_dir, force_download=force_download)
        except Exception as e:
            if use_huggingface_fallback:
                print(f"\nKaggle download failed: {e}")
                print("Falling back to Hugging Face dataset...")
                return load_imdb_data_from_huggingface(data_dir, force_download)
            else:
                raise
    
    # Load CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Map sentiment to binary labels
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Split into train and test (80/20 split)
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    return {
        'train': train_df,
        'test': test_df
    }


def load_imdb_data_from_huggingface(data_dir='data/imdb', force_download=False):
    """
    Load IMDB dataset from Hugging Face (fallback option).
    
    Args:
        data_dir (str): Directory where dataset is stored
        force_download (bool): If True, re-download dataset even if it exists
        
    Returns:
        dict: Dictionary with 'train' and 'test' dataframes
    """
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")
    
    data_path = Path(data_dir)
    hf_data_path = data_path / 'hf_dataset'
    
    # Load from disk if exists
    if hf_data_path.exists() and not force_download:
        print(f"Loading dataset from {hf_data_path}...")
        dataset = load_from_disk(str(hf_data_path))
    else:
        print("Downloading IMDB dataset from Hugging Face...")
        dataset = load_dataset('imdb')
        
        # Save to disk for future use
        os.makedirs(hf_data_path, exist_ok=True)
        dataset.save_to_disk(str(hf_data_path))
        print(f"Dataset saved to {hf_data_path}")
    
    # Convert to pandas dataframes
    train_df = dataset['train'].to_pandas()
    test_df = dataset['test'].to_pandas()
    
    # Add sentiment column for consistency
    train_df['sentiment'] = train_df['label'].map({1: 'positive', 0: 'negative'})
    test_df['sentiment'] = test_df['label'].map({1: 'positive', 0: 'negative'})
    
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    return {
        'train': train_df,
        'test': test_df
    }


def load_imdb_data(data_dir='data/imdb', force_download=False, use_kaggle=True):
    """
    Load IMDB dataset (main function).
    
    Args:
        data_dir (str): Directory where dataset is stored
        force_download (bool): If True, re-download dataset even if it exists
        use_kaggle (bool): If True, try Kaggle first; otherwise use Hugging Face
        
    Returns:
        dict: Dictionary with 'train' and 'test' dataframes
    """
    if use_kaggle:
        return load_imdb_data_from_kaggle(data_dir, force_download, use_huggingface_fallback=True)
    else:
        return load_imdb_data_from_huggingface(data_dir, force_download)


if __name__ == "__main__":
    # Test loading
    dataset = load_imdb_data()
    print("\nSample train example:")
    print(dataset['train'][0])

