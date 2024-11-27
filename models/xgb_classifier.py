# models/xgboost.py

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from src.constants import SEED

def train_xgboost(X_train, y_train):
    """
    Train XGBoostClassifier for multiclass classification using GridSearchCV.
    """
    # Define parameter grid for tuning
    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.1],
        "max_depth": [4, 6, 8],
        "subsample": [0.8, 1.0],
    }

    # Initialize the XGBClassifier
    xgb_model = XGBClassifier(
        random_state=SEED,
        objective="multi:softprob",  # Multiclass classification
        eval_metric="mlogloss",     # Multiclass log loss
        use_label_encoder=False
    )

    # Perform GridSearchCV
    grid_search = GridSearchCV(
        xgb_model,
        param_grid,
        cv=5,
        scoring="f1_weighted",
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    print(f"Best parameters for XGBoost: {grid_search.best_params_}")
    return grid_search.best_estimator_
