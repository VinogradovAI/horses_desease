# models/gradient_boosting.py

import logging
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from src.constants import MODEL_PARAMS, SEED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_gradient_boosting(X_train, y_train):
    """
    Train Gradient Boosting for multiclass classification using GridSearchCV.
    """
    try:
        param_grid = MODEL_PARAMS.get("gradient_boosting", {})
        if not param_grid:
            raise ValueError("Parameter grid for Gradient Boosting is empty or not defined.")

        grid_search = GridSearchCV(
            GradientBoostingClassifier(random_state=SEED),
            param_grid,
            cv=5,
            scoring="f1_weighted",
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        logger.info(f"Best parameters for Gradient Boosting: {grid_search.best_params_}")
        return grid_search.best_estimator_
    except Exception as e:
        logger.error(f"Error training Gradient Boosting: {e}")
        raise