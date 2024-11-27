# models/logistic_regression.py

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from src.constants import MODEL_PARAMS, SEED

def train_logistic_regression(X_train, y_train):
    """
    Train logistic regression for multiclass classification using GridSearchCV.
    """
    param_grid = MODEL_PARAMS["logistic_regression"]
    grid_search = GridSearchCV(
        LogisticRegression(random_state=SEED, multi_class="multinomial", solver="lbfgs"),
        param_grid,
        cv=5,
        scoring="f1_weighted",
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    print(f"Best parameters for Logistic Regression: {grid_search.best_params_}")
    return grid_search.best_estimator_
