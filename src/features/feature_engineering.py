# ------------------ Feature Engineering ------------------

import pandas as pd
import os
import yaml
from sklearn.preprocessing import LabelEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
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

# ---------- Load YAML config ----------
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

config = params["feature_engineering"]

input_file = config.get("input_file", "data/processed/preprocessed_df.csv")
output_file = config.get("output_file", "data/interim/df_features.csv")
label_encode_columns = config.get("label_encode_columns", [])
one_hot_encode_columns = config.get("one_hot_encode_columns", [])
drop_first = config.get("drop_first_dummy", True)
vif_drop_cols = config.get("drop_high_vif_columns", [])

# ---------- Load preprocessed data ----------
df_preprocessed = pd.read_csv(input_file)

# ---------- Label Encoding ----------
le = LabelEncoder()
for col in label_encode_columns:
    try:
        df_preprocessed[col] = le.fit_transform(df_preprocessed[col])
    except Exception as e:
        logger.error(f"[Label Encoding Error] {col}: {e}")

# ---------- One-Hot Encoding ----------
try:
    df_preprocessed = pd.get_dummies(
        df_preprocessed,
        columns=one_hot_encode_columns,
        drop_first=drop_first
    )
except Exception as e:
    logger.error(f"[One-Hot Encoding Error]: {e}")

# ---------- Convert bools to ints ----------
bool_cols = df_preprocessed.select_dtypes(include='bool').columns
df_preprocessed[bool_cols] = df_preprocessed[bool_cols].astype(int)

# ---------- VIF Calculation ----------
X = df_preprocessed.drop(columns=['loan_status'])
y = df_preprocessed['loan_status']

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# ---------- Calculate VIF ----------
vif = pd.DataFrame({
    'Feature': X_scaled.columns,
    'VIF': [variance_inflation_factor(X_scaled.values, i) for i in range(X_scaled.shape[1])]
}).sort_values(by='VIF', ascending=False)

# ---------- Drop high VIF columns ----------
X_scaled.drop(columns=vif_drop_cols, inplace=True, errors='ignore')

# ---------- Reattach target ----------
X_scaled['loan_status'] = y.values

# ---------- Save final features ----------
os.makedirs(os.path.dirname(output_file), exist_ok=True)
X_scaled.to_csv(output_file, index=False)

logger.error(f"Feature engineering completed. Saved to {output_file}")
