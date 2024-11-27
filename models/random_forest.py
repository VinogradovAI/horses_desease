# models/random_forest.py

import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from src.constants import MODEL_PARAMS, SEED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_random_forest(X_train, y_train):
    """
    Train Random Forest for multiclass classification using GridSearchCV.
    """
    try:
        param_grid = MODEL_PARAMS.get("random_forest", {})
        if not param_grid:
            raise ValueError("Parameter grid for Random Forest is empty or not defined.")

        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=SEED),
            param_grid,
            cv=5,
            scoring="f1_weighted",
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        logger.info(f"Best parameters for Random Forest: {grid_search.best_params_}")
        return grid_search.best_estimator_
    except Exception as e:
        logger.error(f"Error training Random Forest: {e}")
        raise
