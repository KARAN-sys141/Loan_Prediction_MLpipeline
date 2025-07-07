
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pandas as pd
import os
import yaml
import logging


logger = logging.getLogger('model_building_1')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter("%(asctime)s - %(name)s -%(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

def load_building_params(filepath="params.yaml"):
    try:
        with open(filepath, "r") as f:
            params = yaml.safe_load(f)
        return params.get("data_building_1", {})
    except FileNotFoundError:
        logger.error(f"Parameter file '{filepath}' not found.")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")

def load_features(filepath="data/interim/df_features.csv"):
    try:
        df = pd.read_csv(filepath)
        if 'loan_status' not in df.columns:
            logger.error("Missing 'loan_status' column in feature file.")
        X = df.drop(columns=['loan_status'])
        y = df['loan_status']
        return X, y
    except FileNotFoundError:
        logger.error(f"Features file '{filepath}' not found.")
    except pd.errors.ParserError:
        logger.error("Error parsing the features CSV.")
    except Exception as e:
        logger.error(f"Unexpected error while loading features: {e}")

def split_data(X, y, test_size, random_state):
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    except Exception as e:
        logger.error(f"Error during train-test split: {e}")

def apply_smote(X_train, y_train, smote_random_state):
    try:
        smote = SMOTE(random_state=smote_random_state)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        return X_res, pd.DataFrame(y_res, columns=["loan_status"])
    except Exception as e:
        logger.error(f"Error applying SMOTE: {e}")

def save_split_data(X_train, X_test, y_train, y_test, output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)
        X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
        X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
        y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
        y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
        print(f"✅ Data splitting and SMOTE completed. Files saved in '{output_dir}'")
    except Exception as e:
        logger.error(f"Error saving split files: {e}")

def run_data_building():
    config = load_building_params()
    test_size = config.get("test_size", 0.2)
    random_state = config.get("random_state", 42)
    smote_random_state = config.get("smote_random_state", 42)
    output_dir = config.get("output_dir", "data/processed_model")

    X, y = load_features()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    X_train_res, y_train_res = apply_smote(X_train, y_train, smote_random_state)
    y_test_df = pd.DataFrame(y_test, columns=['loan_status'])

    save_split_data(X_train_res, X_test, y_train_res, y_test_df, output_dir)

if __name__ == "__main__":
    try:
        run_data_building()
    except Exception as e:
        logger.error("❌ Error during data building step:", e)
