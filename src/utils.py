# src/utils.py

import joblib
import os
from src.constants import MODEL_SAVE_PATH

def save_model(model, filename):
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_SAVE_PATH, f"{filename}.joblib"))

def load_model(filename):
    return joblib.load(os.path.join(MODEL_SAVE_PATH, f"{filename}.joblib"))