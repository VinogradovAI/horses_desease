# models/lightgbm.py

from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from src.constants import SEED

def train_lightgbm(X_train, y_train):
    """
    Train LightGBM for multiclass classification using GridSearchCV.
    """
    # Define parameter grid for tuning
    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1],
        "max_depth": [4, 6, 8],
        "num_leaves": [15, 31],
    }

    # Initialize LGBMClassifier
    lightgbm_model = LGBMClassifier(
        random_state=SEED,
        objective="multiclass",  # Multiclass classification
        metric="multi_logloss"  # Multiclass log loss
    )

    # Perform GridSearchCV
    grid_search = GridSearchCV(
        lightgbm_model,
        param_grid,
        cv=5,
        scoring="f1_weighted",
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    print(f"Best parameters for LightGBM: {grid_search.best_params_}")
    return grid_search.best_estimator_
