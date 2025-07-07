import pandas as pd
import os
import yaml
import logging

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler("erros.log")
file_handler.setLevel("ERROR")

formatter = logging.Formatter("%(asctime)s - %(name)s -%(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(filepath="params.yaml"):
    try:
        with open(filepath, "r") as f:
            params = yaml.safe_load(f)
        return params["data_ingestion"]
    
    except FileNotFoundError:
        # raise FileNotFoundError(f"Parameter file '{filepath}' not found.")
        logger.error(f"Parameter file '{filepath}' not found.")
        
    except yaml.YAMLError as e:
        # raise ValueError(f"Error parsing YAML file: {e}")
        logger.error(f"Error parsing YAML file: {e}")
        
    except KeyError:
        # raise KeyError("Missing 'data_ingestion' section in params.yaml")
        logger.error("Missing 'data_ingestion' section in params.yaml")

def load_data(input_path):
    try:
        return pd.read_csv(input_path)
    except FileNotFoundError:
        # raise FileNotFoundError(f"Input data file '{input_path}' not found.")
        logger.error(f"Input data file '{input_path}' not found.")
    except pd.errors.EmptyDataError:
        # raise ValueError("Input data file is empty or corrupted.")
        logger.error("Input data file is empty or corrupted.")
    except pd.errors.ParserError:
        # raise ValueError("Error parsing the input CSV file.")
        logger.error("Error parsing the input CSV file.")

def process_data(df, drop_duplicates):
    try:
        if drop_duplicates:
            return df.drop_duplicates()
        return df
    except Exception as e:
        # raise RuntimeError(f"Error during data processing: {e}")
        logger.error(f"Error during data processing: {e}")

def save_data(df, output_dir, output_file):
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
        df.to_csv(output_path, index=False)
        print("Data saved to:", output_path)
    except Exception as e:
        # raise IOError(f"Error saving file: {e}")
        logger.error(f"Error saving file: {e}")

def run_data_ingestion():
    config = load_params()
    raw_df = load_data(config["input_path"])
    cleaned_df = process_data(raw_df, config["drop_duplicates"])
    print("Data shape after processing:", cleaned_df.shape)
    save_data(cleaned_df, config["output_dir"], config["output_file"])

# Run the function only if script is run directly (optional safety)
if __name__ == "__main__":
    try:
        run_data_ingestion()
    except Exception as e:
        # print("❌ Error:", e)
        logger.error("❌ Error:", e)


