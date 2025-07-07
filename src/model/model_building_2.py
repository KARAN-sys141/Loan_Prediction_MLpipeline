
import pandas as pd
import pickle
import os
import yaml
import logging




from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


logger = logging.getLogger('model_building_2')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter("%(asctime)s - %(name)s -%(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

def load_model_params(filepath="params.yaml"):
    try:
        with open(filepath, "r") as f:
            params = yaml.safe_load(f)
        return params.get("model_building_2", {})
    except FileNotFoundError:
        logger.error(f"Parameter file '{filepath}' not found.")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")

def load_training_data(input_dir):
    try:
        X_train = pd.read_csv(os.path.join(input_dir, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(input_dir, "X_test.csv"))
        y_train = pd.read_csv(os.path.join(input_dir, "y_train.csv")).squeeze()
        y_test = pd.read_csv(os.path.join(input_dir, "y_test.csv")).squeeze()
        return X_train, X_test, y_train, y_test
    except FileNotFoundError as e:
        logger.error(f"Training data not found: {e}")
    except pd.errors.ParserError:
        logger.error("Error reading training data files.")
    except Exception as e:
        logger.error(f"Unexpected error during data loading: {e}")

def fix_none(d):
    return {k: None if v is None or v == "null" else v for k, v in d.items()}

def initialize_models(config):
    try:
        rf_params = fix_none(config.get("random_forest", {}))
        xgb_params = fix_none(config.get("xgboost", {}))
        gb_params = fix_none(config.get("gradient_boosting", {}))

        models = {
            "RF": RandomForestClassifier(**rf_params),
            "Xgboost": XGBClassifier(**xgb_params),
            "GB": GradientBoostingClassifier(**gb_params)
        }
        return models
    except Exception as e:
        logger.error(f"Error initializing models: {e}")

def train_models(models, X_train, y_train):
    try:
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
    except Exception as e:
        logger.error(f"Error during model training: {e}")

def save_model(model, output_path):
    try:
        with open(output_path, "wb") as f:
            pickle.dump(model, f)
        print(f"\n✅ Tuned model saved as '{output_path}'")
    except Exception as e:
        logger.error(f"Error saving the model: {e}")

def run_model_training():
    config = load_model_params()

    input_dir = config.get("input_dir", "data/processed_model")
    output_model_file = config.get("output_model_file", "model.pkl")
    best_model_key = config.get("best_model_key", "GB")

    X_train, X_test, y_train, y_test = load_training_data(input_dir)
    models = initialize_models(config)
    train_models(models, X_train, y_train)

    best_model = models.get(best_model_key, models["GB"])
    save_model(best_model, output_model_file)

if __name__ == "__main__":
    try:
        run_model_training()
    except Exception as e:
        logger.error("❌ Error during model training:", e)
