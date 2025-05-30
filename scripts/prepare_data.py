"""
Script to prepare and organize news sentiment data.

This script processes various news datasets and prepares them for training.
"""

import pandas as pd
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing import NewsPreprocessor

def prepare_financial_news_data():
    """Prepare financial news sentiment data."""
    data_dir = Path("./data/financial_news")
    
    if not data_dir.exists():
        print("Financial news data not found. Please ensure data is in ./data/financial_news/")
        return    
    # Process financial news files (only original data files, not processed ones)
    for file_path in data_dir.glob("*.csv"):
        # Skip already processed files
        if file_path.name.startswith("processed_"):
            continue
        print(f"Processing {file_path}")
          # Try different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                # Read CSV without headers since the file doesn't have proper column names
                df = pd.read_csv(file_path, encoding=encoding, header=None, names=['sentiment', 'text'])
                print(f"Successfully read {file_path.name} with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            print(f"Could not read {file_path.name} with any encoding, skipping...")
            continue
            
        # Display the structure for verification
        print(f"  Data shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Sample sentiment values: {df['sentiment'].value_counts().head()}")
        print(f"  First text sample: {df['text'].iloc[0][:100]}...")
        
        # Clean any whitespace in sentiment labels
        df['sentiment'] = df['sentiment'].str.strip()
        
        # Initialize preprocessor
        preprocessor = NewsPreprocessor(tokenizer_name="ProsusAI/finbert")
        
        # Preprocess the dataset
        processed_df = preprocessor.preprocess_dataset(df)
        
        # Save processed data
        output_path = data_dir / f"processed_{file_path.name}"
        processed_df.to_csv(output_path, index=False)
        print(f"Saved processed data to {output_path}")

def create_train_val_test_splits():
    """Create train/validation/test splits for all datasets."""
    data_dir = Path("./data")
    
    for dataset_dir in data_dir.iterdir():
        if dataset_dir.is_dir():
            print(f"Creating splits for {dataset_dir.name}")
            
            for csv_file in dataset_dir.glob("processed_*.csv"):
                df = pd.read_csv(csv_file)
                
                # Split data
                train_size = int(0.8 * len(df))
                val_size = int(0.1 * len(df))
                
                train_df = df[:train_size]
                val_df = df[train_size:train_size + val_size]
                test_df = df[train_size + val_size:]
                
                # Save splits
                base_name = csv_file.stem.replace("processed_", "")
                train_df.to_csv(dataset_dir / f"{base_name}_train.csv", index=False)
                val_df.to_csv(dataset_dir / f"{base_name}_val.csv", index=False)
                test_df.to_csv(dataset_dir / f"{base_name}_test.csv", index=False)
                
                print(f"  Split {csv_file.name}: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

if __name__ == "__main__":
    print("Preparing news sentiment data...")
    
    # Create output directories
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # Prepare financial news data
    prepare_financial_news_data()
    
    # Create splits
    create_train_val_test_splits()
    
    print("Data preparation complete!")
