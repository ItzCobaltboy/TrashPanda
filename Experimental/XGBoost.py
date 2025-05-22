import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from datetime import datetime

# --- Load Data ---
df = pd.read_csv("trashcan_data.csv", parse_dates=['timestamp'])

# --- Sort and Feature Engineering ---
df = df.sort_values(by=["trashcan_id", "timestamp"])
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['month'] = df['timestamp'].dt.month

# --- Initialize ---
models = {}
predictions = []

# --- Train Per-Bin Models ---
for bin_id in df['trashcan_id'].unique():
    bin_data = df[df['trashcan_id'] == bin_id].copy()
    bin_data = bin_data.sort_values(by="timestamp")

    # Feature Engineering
    bin_data['daily_fill_rate'] = bin_data['fill_level'].diff().fillna(0)
    bin_data['avg_fill_rate_3d'] = bin_data['daily_fill_rate'].rolling(3).mean().fillna(0)
    bin_data['days_to_full'] = np.where(
        bin_data['avg_fill_rate_3d'] > 0,
        (100 - bin_data['fill_level']) / bin_data['avg_fill_rate_3d'],
        np.nan
    )
    bin_data['future_fill'] = bin_data['fill_level'].shift(-2)
    bin_data['target'] = (bin_data['future_fill'] >= 80).astype(int)
    bin_data = bin_data.dropna()

    if len(bin_data) < 5:  # relaxed threshold for small data
        continue

    features = ['fill_level', 'daily_fill_rate', 'avg_fill_rate_3d',
                'days_to_full', 'day_of_week', 'is_weekend', 'month']

    X = bin_data[features]
    y = bin_data['target']

    # Train model
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    models[bin_id] = model

    # Predict on today's (latest) data
    today_data = bin_data.iloc[[-1]]
    if today_data[features].isnull().values.any():
        continue

    pred = model.predict(today_data[features])[0]

    predictions.append({
        "trashcan_id": bin_id,
        "predicted_full": bool(pred),
        "current_fill": today_data['fill_level'].values[0],
        "daily_rate": today_data['daily_fill_rate'].values[0],
        "days_to_full": today_data['days_to_full'].values[0]
    })

# --- Save Output ---
output_df = pd.DataFrame(predictions)

# Convert to vertical format
if not output_df.empty:
    vertical_df = pd.melt(output_df,
                          id_vars=['trashcan_id'],
                          var_name='attribute',
                          value_name='value')
    vertical_df.to_csv("predicted_bins_vertical.csv", index=False)
    print("✅ Saved predictions to predicted_bins_vertical.csv")
else:
    print("⚠️ No valid predictions were made. Check your data.")

