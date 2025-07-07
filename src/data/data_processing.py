import pandas as pd
import numpy as np
import os
import yaml
import warnings
import logging

warnings.filterwarnings("ignore")


logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter("%(asctime)s - %(name)s -%(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

def load_preprocessing_params(filepath="params.yaml"):
    try:
        with open(filepath, "r") as f:
            params = yaml.safe_load(f)
        return params["data_preprocessing"]
    except FileNotFoundError:
        logger.error(f"Parameter file '{filepath}' not found.")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
    except KeyError:
        logger.error("Missing 'data_preprocessing' section in params.yaml")

def load_cleaned_data(input_file):
    try:
        return pd.read_csv(input_file)
    except FileNotFoundError:
        logger.error(f"Input file '{input_file}' not found.")
    except pd.errors.EmptyDataError:
        logger.error("Input file is empty or corrupted.")
    except pd.errors.ParserError:
        logger.error("Error parsing the input CSV file.")

def cast_column_types(df, type_cast_map):
    for col, dtype in type_cast_map.items():
        try:
            df[col] = df[col].astype(dtype)
        except Exception as e:
            logger.error(f"[TypeCast Error] Column '{col}' to {dtype} failed: {e}")
    return df

def apply_log_transformations(df, log_columns):
    for col in log_columns:
        try:
            if col == "person_emp_exp":
                df[col] = np.log1p(df[col] + 1)
            else:
                df[col] = np.log1p(df[col])
        except Exception as e:
            logger.error(f"[Log Transform Error] Column '{col}': {e}")
    return df

def clip_person_age(df, clip_config):
    try:
        df['person_age'] = df['person_age'].clip(
            lower=clip_config.get("lower", 18),
            upper=clip_config.get("upper", 85)
        )
    except Exception as e:
        logger.error(f"[Clipping Error] 'person_age' column: {e}")
    return df

def save_preprocessed_data(df, output_dir, output_file):
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
        df.to_csv(output_path, index=False)
        logger.error("✅ Preprocessed data saved to:", output_path)
    except Exception as e:
        logger.error(f"Error saving preprocessed file: {e}")

def run_data_preprocessing():
    config = load_preprocessing_params()

    input_file = config.get("input_file", "data/raw/cleaned_df.csv")
    output_dir = config.get("output_dir", "data/processed")
    output_file = config.get("output_file", "preprocessed_df.csv")
    clip_config = config.get("clip_person_age", {"lower": 18, "upper": 85})
    log_columns = config.get("log_transform_columns", ["person_income", "loan_amnt", "person_emp_exp"])
    type_cast_map = config.get("column_type_cast", {})

    df = load_cleaned_data(input_file)
    df = cast_column_types(df, type_cast_map)
    df = apply_log_transformations(df, log_columns)
    df = clip_person_age(df, clip_config)

    logger.error("🔍 Missing values:\n", df.isnull().sum())

    save_preprocessed_data(df, output_dir, output_file)

if __name__ == "__main__":
    try:
        run_data_preprocessing()
    except Exception as e:
        logger.error("❌ Error during preprocessing:", e)







