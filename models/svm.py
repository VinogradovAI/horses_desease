# models/svm.py

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from src.constants import SEED

def train_svm(X_train, y_train):
    """
    Train Support Vector Machine (SVM) for multiclass classification using GridSearchCV.
    """
    # Define parameter grid for tuning
    param_grid = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    }

    # Initialize SVC
    svm_model = SVC(
        random_state=SEED,
        decision_function_shape="ovr",  # One-vs-Rest for multiclass classification
        probability=True  # Enable probability outputs for ROC-AUC
    )

    # Perform GridSearchCV
    grid_search = GridSearchCV(
        svm_model,
        param_grid,
        cv=5,
        scoring="f1_weighted",
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    print(f"Best parameters for SVM: {grid_search.best_params_}")
    return grid_search.best_estimator_
