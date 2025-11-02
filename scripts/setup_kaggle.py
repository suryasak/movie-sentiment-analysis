"""
Setup script for Kaggle API credentials.
Run this script to help set up Kaggle API for downloading datasets.
"""

import os
from pathlib import Path


def setup_kaggle():
    """Guide user through Kaggle API setup."""
    print("="*70)
    print("Kaggle API Setup Guide")
    print("="*70)
    
    print("\nTo download datasets from Kaggle, you need to:")
    print("\n1. Go to https://www.kaggle.com/ and create an account")
    print("2. Go to your account settings: https://www.kaggle.com/account")
    print("3. Scroll to 'API' section and click 'Create New API Token'")
    print("4. This will download a file called 'kaggle.json'")
    print("\n5. Place the kaggle.json file in one of these locations:")
    
    home = Path.home()
    kaggle_dir = home / '.kaggle'
    
    print(f"   - {kaggle_dir}/kaggle.json")
    print(f"   - Or set KAGGLE_CONFIG_DIR environment variable")
    
    print("\n6. Set proper permissions (on Linux/Mac):")
    print(f"   chmod 600 {kaggle_dir}/kaggle.json")
    
    # Check if kaggle.json exists
    kaggle_json = kaggle_dir / 'kaggle.json'
    if kaggle_json.exists():
        print(f"\n✓ Found kaggle.json at {kaggle_json}")
    else:
        print(f"\n⚠ kaggle.json not found at {kaggle_json}")
        print("   Please follow the steps above to set it up.")
    
    print("\n7. Install kaggle package:")
    print("   pip install kaggle")
    
    print("\n" + "="*70)
    print("Note: If Kaggle setup fails, the code will automatically")
    print("      fall back to using Hugging Face dataset.")
    print("="*70)


if __name__ == "__main__":
    setup_kaggle()

