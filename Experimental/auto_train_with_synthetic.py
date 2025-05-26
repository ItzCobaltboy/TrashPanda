import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import os

# CONFIG
OUTPUT_DATA_FILE = "trashcan_data.csv"
NUM_BINS = 5
DAYS = 50
PREDICT_FORWARD = 2

# --- Step 1: Generate Synthetic Trashcan Data ---
def generate_synthetic_data():
    records = []
    start_date = datetime(2025, 1, 1)

    for bin_id in range(1, NUM_BINS + 1):
        level = 0
        for day in range(DAYS):
            timestamp = start_date + timedelta(days=day)
            if day % 15 == 0:  # simulate emptying
                level = 0
            level += np.random.randint(5, 12)  # daily fill
            level = min(100, level)
            records.append({
                'trashcan_id': f'bin{bin_id}',
                'timestamp': timestamp,
                'fill_level': level
            })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DATA_FILE, index=False)
    print(f"✅ Synthetic data saved to {OUTPUT_DATA_FILE}")

# --- Step 2: Retrain & Predict with XGBoost ---
def train_and_predict(data_file):
    df = pd.read_csv(data_file, parse_dates=['timestamp'])
    df = df.sort_values(by=["trashcan_id", "timestamp"])
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['month'] = df['timestamp'].dt.month

    predictions = []

    for bin_id in df['trashcan_id'].unique():
        bin_data = df[df['trashcan_id'] == bin_id].copy()
        bin_data = bin_data.sort_values(by="timestamp")
        bin_data['daily_fill_rate'] = bin_data['fill_level'].diff().fillna(0)
        bin_data['avg_fill_rate_3d'] = bin_data['daily_fill_rate'].rolling(3).mean().fillna(0)
        bin_data['days_to_full'] = np.where(
            bin_data['avg_fill_rate_3d'] > 0,
            (100 - bin_data['fill_level']) / bin_data['avg_fill_rate_3d'],
            np.nan
        )
        bin_data['future_fill'] = bin_data['fill_level'].shift(-PREDICT_FORWARD)
        bin_data['target'] = (bin_data['future_fill'] >= 80).astype(int)
        bin_data = bin_data.dropna()

        if len(bin_data) < 5:
            continue

        features = ['fill_level', 'daily_fill_rate', 'avg_fill_rate_3d',
                    'days_to_full', 'day_of_week', 'is_weekend', 'month']

        X = bin_data[features]
        y = bin_data['target']

        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X, y)

        # Predict today
        today_data = bin_data.iloc[[-1]]
        if today_data[features].isnull().values.any():
            continue

        pred = model.predict(today_data[features])[0]
        predictions.append({
            "trashcan_id": bin_id,
            "predicted_full": bool(pred),
            "current_fill": today_data['fill_level'].values[0],
            "days_to_full": today_data['days_to_full'].values[0]
        })

    # Save predictions
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv("predicted_bins.csv", index=False)
    print("📈 Predictions saved to predicted_bins.csv")

# --- Execute ---
if __name__ == "__main__":
    generate_synthetic_data()
    train_and_predict(OUTPUT_DATA_FILE)
