# notebooks/feature_engineering.py

import pandas as pd
import matplotlib.pyplot as plt
import os
from src.constants import (
    CLEANED_DATA_PATH,
    FEATURES_DATA_PATH,
    FEATURE_PLOTS_PATH,
    OUTLIER_COLUMNS,
    CATEGORICAL_COLUMNS,
    OUTCOME_MAPPING
)


def load_cleaned_data():
    """Load cleaned data from CSV."""
    return pd.read_csv(CLEANED_DATA_PATH)


def remove_outliers(df, columns):
    """
    Remove outliers based on the IQR method for specified columns.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        columns (list): List of column names to process.

    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


def plot_outliers_removal(df_before, df_after, columns):
    """
    Visualize the effect of outlier removal with boxplots.

    Parameters:
        df_before (pd.DataFrame): DataFrame before outlier removal.
        df_after (pd.DataFrame): DataFrame after outlier removal.
        columns (list): List of column names to visualize.
    """
    os.makedirs(FEATURE_PLOTS_PATH, exist_ok=True)
    for col in columns:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.boxplot(df_before[col].dropna(), vert=False)
        plt.title(f"Before Outlier Removal: {col}")

        plt.subplot(1, 2, 2)
        plt.boxplot(df_after[col].dropna(), vert=False)
        plt.title(f"After Outlier Removal: {col}")

        plt.savefig(f"{FEATURE_PLOTS_PATH}/{col}_outliers_removal.png")
        plt.close()


def encode_categorical_features(df, target_column="outcome"):
    """
    Encode categorical features into numerical values, excluding the target column.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Name of the target column to exclude from encoding.

    Returns:
        pd.DataFrame: DataFrame with encoded categorical features.
    """
    # Ordinal encoding
    for col, mapping in CATEGORICAL_COLUMNS["ordinal"].items():
        if col in df.columns and col != target_column:
            df[col] = df[col].map(mapping)

    # One-Hot Encoding
    nominal_features = [col for col in CATEGORICAL_COLUMNS["nominal"] if col != target_column]
    df = pd.get_dummies(df, columns=nominal_features, drop_first=True)

    return df


def process_target_column(df, target_column="outcome"):
    """
    Process the target column (outcome) by encoding it into numerical labels.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Name of the target column.

    Returns:
        pd.DataFrame: DataFrame with processed target column.
    """
    target_mapping = {"lived": 0, "died": 1, "euthanized": 2}
    df[target_column] = df[target_column].map(target_mapping)
    return df


def save_features(df):
    """Save transformed features to a CSV file."""
    df.to_csv(FEATURES_DATA_PATH, index=False)


def main():
    # Load cleaned data
    df = load_cleaned_data()

    # Remove outliers
    df_before = df.copy()
    df = remove_outliers(df, OUTLIER_COLUMNS)
    plot_outliers_removal(df_before, df, OUTLIER_COLUMNS)

    # Process the target column
    target_column = "outcome"
    df = process_target_column(df, target_column)

    # Encode categorical features (excluding the target column)
    df = encode_categorical_features(df, target_column)

    # Save transformed features
    save_features(df)
    print(f"Features saved to {FEATURES_DATA_PATH}")


if __name__ == "__main__":
    main()