"""
data_loader.py
Aviation Data Pipeline for Machine Learning Course Project.
Loads hourly airport operations data and prepares features for:
1. Regression Task: Predicting 'avg_departure_delay' (or 'total_flights')
2. Classification Task: Predicting 'congestion_level' (0: Low, 1: Medium, 2: High)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DATA_FILE = os.path.join(DATA_DIR, "hourly_airport_traffic.csv")

FEATURE_COLS = [
    "arriving_flights",
    "departing_flights",
    "total_flights",
    "avg_arrival_delay",
    "weather_delay_factor",
    "taxi_out_avg",
    "airport_capacity",
    "runway_utilization",
    "hour_of_day",
    "day_of_week"
]

def load_air_traffic_data(data_path: str = DATA_FILE) -> pd.DataFrame:
    """Loads the hourly airport operations dataset."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    df = pd.read_csv(data_path)
    return df

def get_regression_data(df: pd.DataFrame = None, target_col: str = "avg_departure_delay", test_size: float = 0.2, random_state: int = 42):
    """
    Prepares train/test split for regression tasks.
    Target: avg_departure_delay (continuous flight delay in minutes).
    """
    if df is None:
        df = load_air_traffic_data()
        
    X = df[FEATURE_COLS].copy()
    y = df[target_col].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLS, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FEATURE_COLS, index=X_test.index)
    
    return {
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": FEATURE_COLS,
        "target_name": target_col
    }

def get_classification_data(df: pd.DataFrame = None, target_col: str = "congestion_level", test_size: float = 0.2, random_state: int = 42):
    """
    Prepares train/test split for classification tasks.
    Target: congestion_level (0: Low, 1: Medium, 2: High).
    """
    if df is None:
        df = load_air_traffic_data()
        
    # We exclude runway_utilization from direct features for classification in some experiments 
    # to avoid trivial collinearity, or keep it to evaluate maximal separation.
    X = df[FEATURE_COLS].copy()
    y = df[target_col].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=FEATURE_COLS, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=FEATURE_COLS, index=X_test.index)
    
    return {
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "feature_names": FEATURE_COLS,
        "target_name": target_col
    }

if __name__ == "__main__":
    df = load_air_traffic_data()
    print("Dataset shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Congestion Level Counts:\n", df["congestion_level"].value_counts())
