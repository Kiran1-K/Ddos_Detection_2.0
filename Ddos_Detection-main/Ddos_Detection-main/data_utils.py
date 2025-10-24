import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import os

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
# Core required columns for basic functionality
REQUIRED_COLUMNS = ["Packet_Size", "Duration", "Protocol"]

# DDoS-specific features (expected in input or will be calculated)
DDoS_FEATURES = [
    "Packets_Per_Second",
    "Bytes_Per_Second",
    "Unique_IPs_Per_Second",
    "SYN_Count",
    "ACK_Count",
    "RST_Count",
    "FIN_Count",
    "Avg_Packet_Size",
    "Flow_Duration"
]

# Optional columns that are good to have but not strictly required
OPTIONAL_COLUMNS = ["Src_IP", "Dst_IP", "Timestamp", "Flags"]

def allowed_file_extension(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS

def read_dataset(file_path: str):
    _, ext = os.path.splitext(file_path.lower())
    if ext == ".csv":
        return pd.read_csv(file_path)
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file extension: " + ext)

def extract_relevant_columns(df: pd.DataFrame):
    """
    Process the input DataFrame to extract and calculate features needed for DDoS detection.
    Handles both raw network data and pre-processed data.
    """
    df = df.copy()
    
    # Ensure required columns are present
    present_cols = [c for c in REQUIRED_COLUMNS if c in df.columns]
    if not present_cols:
        raise ValueError("Dataset must have at least one required column: " + 
                        ", ".join(REQUIRED_COLUMNS))
    
    # Add optional columns if present
    present_cols += [c for c in OPTIONAL_COLUMNS if c in df.columns]
    df = df[present_cols]
    
    # Convert numeric columns and handle missing values
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill missing values with column median for numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Handle timestamp conversion
    for time_col in ['Timestamp', 'Time', 'timestamp', 'time']:
        if time_col in df.columns:
            df['Timestamp'] = pd.to_datetime(df[time_col], errors='coerce')
            if not df['Timestamp'].isnull().all():
                break
    
    # Calculate DDoS-specific features if we have the required data
    if 'Timestamp' in df.columns and not df['Timestamp'].isnull().all():
        df = calculate_time_based_features(df)
    
    # Extract TCP flags if available
    if 'Flags' in df.columns:
        df = extract_tcp_flags(df)
    
    # Ensure we have all required DDoS features, fill with 0 if missing
    for feature in DDoS_FEATURES:
        if feature not in df.columns:
            df[feature] = 0
    
    # Ensure all numeric columns are properly typed
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove any remaining non-numeric columns that might cause issues
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df[numeric_cols]

def calculate_time_based_features(df, window='1S'):
    """Calculate time-based features like packets/bytes per second."""
    try:
        if 'Timestamp' not in df.columns or df['Timestamp'].isnull().all():
            return df
            
        # Make a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Ensure Timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        
        # Sort by timestamp
        df = df.sort_values('Timestamp')
        
        # Calculate time differences
        time_diff = df['Timestamp'].diff().dt.total_seconds().fillna(0)
        
        # Calculate packets per second (avoid division by zero)
        time_diff = time_diff.replace(0, 1e-6)  # Replace zeros with small value
        df['Packets_Per_Second'] = 1 / time_diff
        
        # Calculate bytes per second if packet size is available
        if 'Packet_Size' in df.columns:
            df['Bytes_Per_Second'] = df['Packet_Size'] / time_diff
            df['Avg_Packet_Size'] = df['Packet_Size'].rolling(window=10, min_periods=1).mean()
        
        # Count unique source IPs per time window if available
        if 'Src_IP' in df.columns:
            try:
                # More robust unique IP counting
                df_temp = df.set_index('Timestamp').copy()
                df['Unique_IPs_Per_Second'] = df_temp['Src_IP'].groupby(pd.Grouper(freq=window)).nunique().reindex(df_temp.index, method='ffill').values
            except Exception as e:
                print(f"Warning: Could not calculate unique IPs: {str(e)}")
                df['Unique_IPs_Per_Second'] = 0
        
        return df
        
    except Exception as e:
        print(f"Error in calculate_time_based_features: {str(e)}")
        # Return the original dataframe with minimal processing
        return df

def extract_tcp_flags(df):
    """Extract TCP flags from the Flags column if present."""
    if 'Flags' not in df.columns:
        return df
    
    # Common TCP flags
    flags_to_extract = ['SYN', 'ACK', 'RST', 'FIN']
    
    for flag in flags_to_extract:
        df[f'{flag}_Count'] = df['Flags'].str.upper().str.contains(flag).astype(int)
    
    return df

def validate_dataset(file_path: str, min_rows:int=5):
    """
    High-level validate function:
    - reads file
    - extracts relevant columns
    - returns tuple (valid:bool, df_or_none, message:str)
    """
    try:
        df = read_dataset(file_path)
    except Exception as e:
        return False, None, f"Error reading file: {e}"

    try:
        df_clean = extract_relevant_columns(df)
    except Exception as e:
        return False, None, str(e)

    if df_clean.shape[0] < min_rows:
        return False, None, f"Dataset too small: {df_clean.shape[0]} rows (min {min_rows})."

    return True, df_clean, "Dataset valid."

def save_uploaded_file(flask_file, upload_folder: str) -> str:
    filename = secure_filename(flask_file.filename)
    os.makedirs(upload_folder, exist_ok=True)
    filepath = os.path.join(upload_folder, filename)
    flask_file.save(filepath)
    return filepath
