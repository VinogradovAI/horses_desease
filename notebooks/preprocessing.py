# notebooks/preprocessing.py

import pandas as pd
from src.constants import RAW_DATA_PATH, CLEANED_DATA_PATH, TARGET_COLUMN

def load_raw_data():
    """Load raw data from CSV."""
    return pd.read_csv(RAW_DATA_PATH)

def clean_data(df):
    """Clean data: handle missing values."""
    # Drop rows where the target variable is missing
    df = df.dropna(subset=[TARGET_COLUMN])
    # Fill missing values in other columns
    for col in df.columns:
        if df[col].dtype in ["float", "int"]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def save_cleaned_data(df):
    """Save cleaned data to CSV."""
    df.to_csv(CLEANED_DATA_PATH, index=False)

def main():
    df = load_raw_data()
    df = clean_data(df)
    save_cleaned_data(df)
    print(f"Cleaned data saved to {CLEANED_DATA_PATH}")

if __name__ == "__main__":
    main()