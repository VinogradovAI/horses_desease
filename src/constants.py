# src/constants.py

# Paths to data and outputs

PROJECT_LOCAL_PATH = "D:/ml_school/Skillfactory/horses_case/"
RAW_DATA_PATH = PROJECT_LOCAL_PATH + "data/raw/horse.csv"
CLEANED_DATA_PATH = PROJECT_LOCAL_PATH + "data/processed/cleaned.csv"
FEATURES_DATA_PATH = PROJECT_LOCAL_PATH + "data/processed/features.csv"
EDA_PLOTS_PATH = PROJECT_LOCAL_PATH + "plots/eda/"
FEATURE_PLOTS_PATH = PROJECT_LOCAL_PATH + "plots/feature_engineering/"
MODEL_PLOTS_PATH = PROJECT_LOCAL_PATH + "plots/models/"
METRICS_FILE_PATH = PROJECT_LOCAL_PATH + "metrics/evaluation_metrics.txt"
MODEL_SAVE_PATH = PROJECT_LOCAL_PATH + "models/"

# Columns for outlier removal
OUTLIER_COLUMNS = ["pulse", "respiratory_rate", "rectal_temp"]

# Columns not important (non usable for features)
COLUMNS_TO_EXCLUDE = ["hospital_number", ]

# Encoding configuration for categorical columns
CATEGORICAL_COLUMNS = {
    "ordinal": {
        "pain": {"depressed": 0, "mild_pain": 1, "severe_pain": 2, "extreme_pain": 3, "alert": 4},
        "capillary_refill_time": {"less_3_sec": 0, "3": 1, "more_3_sec": 2},
        "peristalsis": {"absent": 0, "hypomotile": 1, "normal": 2, "hypermotile": 3},
        "abdominal_distention": {"none": 0, "slight": 1, "moderate": 2, "severe": 3},
        "temp_of_extremities": {"cool": 0, "cold": 1, "normal": 2, "warm": 3},
        "peripheral_pulse": {"reduced": 0, "normal": 1, "absent": 2, "increased": 3},
        "rectal_exam_feces": {"absent": 0, "decreased": 1, "normal": 2, "increased": 3},
        "abdomen": {"distend_small": 0, "normal": 1, "distend_large": 2, "firm": 3, "other": 4},
        "abdomo_appearance": {"serosanguious": 0, "clear": 1, "cloudy": 2}
    },
    "nominal": [
        "surgery",
        "age",
        "mucous_membrane",
        "nasogastric_tube",
        "nasogastric_reflux",
        "surgical_lesion",
        "cp_data"
    ],
}

# Target column
TARGET_COLUMN = "outcome"

# Mapping for the target column (outcome)
OUTCOME_MAPPING = {"lived": 0, "died": 1, "euthanized": 2}

# Model hyperparameters
MODEL_PARAMS = {
    "random_forest": {
        "n_estimators": [50, 100, 150],
        "max_depth": [5, 10, 15],
        "random_state": [42]
    },
    "logistic_regression": {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["lbfgs"],
        "max_iter": [10000]
    },
    "gradient_boosting": {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7]
    },
    "decision_tree": {
        "max_depth": [3, 5, 7, 10],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "random_state": [42]
    },

    "xgboost": {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "random_state": [42]
    },

    "lightgbm": {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1, 0.2],
        "random_state": [42]
    },

    "svm": {
        "C": [0.01, 0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale"]
    },

    "naive_bayes": {}
}

# Random seed for reproducibility
SEED = 42