# models/decision_tree.py

import logging
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from src.constants import MODEL_PARAMS, SEED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_decision_tree(X_train, y_train):
    """
    Train Decision Tree for multiclass classification using GridSearchCV.
    """
    try:
        param_grid = MODEL_PARAMS.get("decision_tree", {})
        if not param_grid:
            raise ValueError("Parameter grid for Decision Tree is empty or not defined.")

        grid_search = GridSearchCV(
            DecisionTreeClassifier(random_state=SEED),
            param_grid,
            cv=5,
            scoring="f1_weighted",
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        logger.info(f"Best parameters for Decision Tree: {grid_search.best_params_}")
        return grid_search.best_estimator_
    except Exception as e:
        logger.error(f"Error training Decision Tree: {e}")
        raise