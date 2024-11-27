# models/naive_bayes.py

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

def train_naive_bayes(X_train, y_train):
    """
    Train Multinomial Naive Bayes for multiclass classification.
    """
    # Define parameter grid for tuning
    param_grid = {
        "alpha": [0.1, 0.5, 1.0],  # Smoothing parameter
    }

    # Initialize MultinomialNB
    nb_model = MultinomialNB()

    # Perform GridSearchCV
    grid_search = GridSearchCV(
        nb_model,
        param_grid,
        cv=5,
        scoring="f1_weighted",
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    print(f"Best parameters for Naive Bayes: {grid_search.best_params_}")
    return grid_search.best_estimator_
