import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle
import os

REQUIRED_COLUMNS = ["Protocol", "Packet_Size", "Duration"]

def preprocess_data(df, save_scaler: bool = False, model_dir: str = "model"):
    """
    Encodes, fills missing values, and scales dataset.
    Returns: X_scaled, scaler
    """
    # Create a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Step 1: Convert all numeric columns and fill missing values
    for col in df_processed.select_dtypes(include=[np.number]).columns:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        df_processed[col].fillna(df_processed[col].median(), inplace=True)
    
    # Ensure we have at least the required columns
    for col in ["Packet_Size", "Duration"]:
        if col not in df_processed.columns:
            df_processed[col] = df_processed.get(col, 0)
    
    # Handle Protocol specifically
    if "Protocol" in df_processed.columns:
        encoder = LabelEncoder()
        df_processed["Protocol"] = encoder.fit_transform(df_processed["Protocol"].astype(str))
    else:
        df_processed["Protocol"] = 0  # Default value if Protocol is missing
    
    # Get all numeric columns for scaling
    numeric_cols = [col for col in df_processed.select_dtypes(include=[np.number]).columns]
    
    # Ensure we have at least one column to scale
    if not numeric_cols:
        raise ValueError("No numeric columns found for scaling")
    
    # Scale the data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df_processed[numeric_cols])

    if save_scaler:
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

    return X_scaled, scaler

def load_scaler(model_dir: str = "model"):
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Scaler not found. Please train the model first.")
    with open(scaler_path, "rb") as f:
        return pickle.load(f)
