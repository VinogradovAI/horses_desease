
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from src.constants import RAW_DATA_PATH, EDA_PLOTS_PATH


def load_raw_data():
    """Load raw data from CSV."""
    return pd.read_csv(RAW_DATA_PATH)


def check_missing_values(df):
    """Check for missing values and save the result."""
    missing = df.isnull().sum()
    print("\nMissing Values:")
    print(missing)


def plot_distributions(df):
    """Plot distributions of numerical features."""
    os.makedirs(EDA_PLOTS_PATH, exist_ok=True)
    numeric_features = df.select_dtypes(include=["int", "float"]).columns
    for col in numeric_features:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.savefig(f"{EDA_PLOTS_PATH}/{col}_distribution.png")
        plt.close()


def plot_correlation_heatmap(df):
    """Plot a correlation heatmap."""
    # Convert non-numeric data to NaN to avoid conversion errors
    df = df.apply(pd.to_numeric, errors='coerce')

    # Drop columns that became entirely NaN after conversion
    df = df.dropna(axis=1, how='all')

    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig(f"{EDA_PLOTS_PATH}/correlation_heatmap.png")
    plt.close()


def plot_outliers(df):
    """Plot boxplots to visualize outliers."""
    numeric_features = df.select_dtypes(include=["int", "float"]).columns
    for col in numeric_features:
        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f"Outliers in {col}")
        plt.savefig(f"{EDA_PLOTS_PATH}/{col}_outliers.png")
        plt.close()


def main():
    df = load_raw_data()
    check_missing_values(df)
    plot_distributions(df)
    plot_correlation_heatmap(df)
    plot_outliers(df)


if __name__ == "__main__":
    main()