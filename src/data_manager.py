import pandas as pd
import json
import os
from datetime import datetime
import glob

DATA_DIR = "d:/AI/Antigravity/SKB/data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")

# Ensure directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

def load_data(file_path):
    """
    Load CSV data securely.
    """
    try:
        # Basic security check for extension
        if not file_path.lower().endswith('.csv'):
            raise ValueError("Only CSV files are supported.")
        
        return load_csv_robust(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")

def load_csv_robust(filepath_or_buffer):
    """
    Try to load CSV with multiple encodings (UTF-8, Big5, CP950).
    """
    encodings = ['utf-8', 'utf-8-sig', 'big5', 'cp950', 'gb18030', 'latin1']
    
    for enc in encodings:
        try:
            # If it's a file-like object (buffer), reset pointer
            if hasattr(filepath_or_buffer, 'seek'):
                filepath_or_buffer.seek(0)
            
            df = pd.read_csv(filepath_or_buffer, encoding=enc)
            print(f"Successfully loaded with encoding: {enc}") # Debug info
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            # If it's not an encoding error, raise it immediately
            raise e
            
    raise ValueError(f"Unable to read CSV. Tried encodings: {encodings}")

def save_metadata(mapping, filename="column_metadata.json"):
    """
    Save column metadata mapping to disk.
    """
    path = os.path.join(METADATA_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=4, ensure_ascii=False)
    return path

def load_metadata(filename="column_metadata.json"):
    path = os.path.join(METADATA_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_processed_data(df, base_filename="p1_processed"):
    """
    Feature Store: Save processed dataframe with a version tag/timestamp.
    Returns the path to the saved file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{base_filename}_v{timestamp}.csv"
    path = os.path.join(PROCESSED_DIR, filename)
    
    # Save with index=False
    df.to_csv(path, index=False)
    return path

def list_processed_files():
    """List all files in the processed directory."""
    files = glob.glob(os.path.join(PROCESSED_DIR, "*.csv"))
    files.sort(key=os.path.getmtime, reverse=True)
    return files
