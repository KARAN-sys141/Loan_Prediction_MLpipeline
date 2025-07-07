import pandas as pd
import pickle
import json
import yaml
import logging
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

formatter = logging.Formatter("%(asctime)s - %(name)s -%(levelname)s - %(message)s")
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)

def load_evaluation_config(filepath="params.yaml"):
    try:
        with open(filepath, "r") as f:
            params = yaml.safe_load(f)
        return params.get("model_evaluation", {})
    except FileNotFoundError:
        logger.error(f"Parameter file '{filepath}' not found.")
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")

def load_model(model_file):
    try:
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        logger.error(f"Model file '{model_file}' not found.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

def load_data(input_dir):
    try:
        X_test = pd.read_csv(f"{input_dir}/X_test.csv")
        y_test = pd.read_csv(f"{input_dir}/y_test.csv").squeeze()
        X_train = pd.read_csv(f"{input_dir}/X_train.csv")
        y_train = pd.read_csv(f"{input_dir}/y_train.csv").squeeze()
        return X_train, X_test, y_train, y_test
    except FileNotFoundError as e:
        logger.error(f"Missing data file: {e}")
    except pd.errors.ParserError:
        logger.error("Error parsing CSV during data load.")
    except Exception as e:
        logger.error(f"Error loading data: {e}")

def calculate_metrics(model, X_train, X_test, y_train, y_test, average_type, roc_auc_config):
    try:
        y_pred_test = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred_test),
            "test_f1_score": f1_score(y_test, y_pred_test, average=average_type),
            "test_precision": precision_score(y_test, y_pred_test, average=average_type),
            "test_recall": recall_score(y_test, y_pred_test, average=average_type),
            "train_accuracy": accuracy_score(y_train, y_pred_train),
            "train_f1_score": f1_score(y_train, y_pred_train, average=average_type),
            "train_precision": precision_score(y_train, y_pred_train, average=average_type),
            "train_recall": recall_score(y_train, y_pred_train, average=average_type)
        }

        if roc_auc_config.get("enabled", True):
            try:
                y_proba_test = model.predict_proba(X_test)
                metrics["test_roc_auc"] = roc_auc_score(
                    y_test, y_proba_test, multi_class=roc_auc_config.get("multi_class", "ovr")
                )
            except Exception as e:
                metrics["test_roc_auc"] = None
                print(f"[ROC AUC Test Error] {e}")

            try:
                y_proba_train = model.predict_proba(X_train)
                metrics["train_roc_auc"] = roc_auc_score(
                    y_train, y_proba_train, multi_class=roc_auc_config.get("multi_class", "ovr")
                )
            except Exception as e:
                metrics["train_roc_auc"] = None
                print(f"[ROC AUC Train Error] {e}")
        else:
            metrics["test_roc_auc"] = None
            metrics["train_roc_auc"] = None

        return metrics
    except Exception as e:
        logger.error(f"Error calculating evaluation metrics: {e}")

def save_metrics(metrics, output_file):
    try:
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=4)
        print("✅ Evaluation complete. Metrics saved to:", output_file)
        print(f"Training accuracy: {metrics['train_accuracy']:.4f}")
        print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
    except Exception as e:
        logger.error(f"Error saving metrics: {e}")

def run_model_evaluation():
    config = load_evaluation_config()
    
    model_file = config.get("model_file", "model.pkl")
    input_dir = config.get("input_dir", "data/processed_model")
    output_metrics_file = config.get("output_metrics_file", "model_metrics.json")
    average_type = config.get("average_type", "weighted")
    roc_auc_config = config.get("roc_auc", {"enabled": True, "multi_class": "ovr"})

    model = load_model(model_file)
    X_train, X_test, y_train, y_test = load_data(input_dir)
    metrics = calculate_metrics(model, X_train, X_test, y_train, y_test, average_type, roc_auc_config)
    save_metrics(metrics, output_metrics_file)

if __name__ == "__main__":
    try:
        run_model_evaluation()
    except Exception as e:
        logger.error("❌ Error during evaluation:", e)
