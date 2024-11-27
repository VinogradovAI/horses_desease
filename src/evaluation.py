# src/evaluation.py

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    roc_curve,
    auc
)
import os
from src.constants import MODEL_PLOTS_PATH, METRICS_FILE_PATH

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a single model and save results and ROC curve.
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")
    except ValueError:
        roc_auc = None  # ROC-AUC is not applicable if probabilities are not provided

    # Save metrics to a file
    with open(METRICS_FILE_PATH, "a") as f:
        f.write(f"\nModel: {model_name}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"F1-score (weighted): {f1:.4f}\n")
        if roc_auc is not None:
            f.write(f"ROC-AUC (weighted): {roc_auc:.4f}\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\n" + "=" * 50 + "\n")

    # Plot and save ROC curves for each class
    if roc_auc is not None:
        os.makedirs(MODEL_PLOTS_PATH, exist_ok=True)
        for i in range(y_pred_proba.shape[1]):
            fpr, tpr, _ = roc_curve(y_test == i, y_pred_proba[:, i])
            plt.plot(fpr, tpr, label=f"Class {i} (AUC = {auc(fpr, tpr):.2f})")

        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name}")
        plt.legend(loc="lower right")
        plt.savefig(f"{MODEL_PLOTS_PATH}/roc_curve_{model_name}.png")
        plt.close()

    return {"accuracy": accuracy, "f1_score": f1, "roc_auc": roc_auc}