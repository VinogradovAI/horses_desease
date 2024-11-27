from warnings import filterwarnings
filterwarnings("ignore")

import pandas as pd
from models.logistic_regression import train_logistic_regression
from models.random_forest import train_random_forest
from models.multinomial_naive_bayes import train_naive_bayes
from src.evaluation import evaluate_model
from src.utils import save_model
from src.constants import FEATURES_DATA_PATH


def load_features():
    """Load features from CSV."""
    return pd.read_csv(FEATURES_DATA_PATH)


def main():
    # Load features
    df = load_features()
    X = df.drop(columns=["outcome"])
    y = df["outcome"]

    # Train models
    models = {
        "Logistic Regression": train_logistic_regression(X, y),
        "Random Forest": train_random_forest(X, y),
        "Multinomial Naive Bayes": train_naive_bayes(X, y),

    }

    # Evaluate and save models
    for model_name, model in models.items():
        print(f"\nEvaluating {model_name}")
        metrics = evaluate_model(model, X, y, model_name)
        save_model(model, model_name)
        print(metrics)





if __name__ == "__main__":
    main()
