import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from datetime import datetime

# Load your CSV dataset
df = pd.read_csv("trashcan_data.csv", parse_dates=['timestamp'])

# Sort and prepare time-based features
df = df.sort_values(by=["trashcan_id", "timestamp"])
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['month'] = df['timestamp'].dt.month

# Initialize dictionary to store models
models = {}
predictions = []

# Train per-bin models
for bin_id in df['trashcan_id'].unique():
    bin_data = df[df['trashcan_id'] == bin_id].copy()

    bin_data['daily_fill_rate'] = bin_data['fill_level'].diff()
    bin_data['avg_fill_rate_3d'] = bin_data['daily_fill_rate'].rolling(3).mean()
    bin_data['days_to_full'] = (100 - bin_data['fill_level']) / bin_data['avg_fill_rate_3d']
    bin_data['future_fill'] = bin_data['fill_level'].shift(-2)
    bin_data['target'] = (bin_data['future_fill'] >= 80).astype(int)
    bin_data = bin_data.dropna()

    if len(bin_data) < 10:
        continue

    features = ['fill_level', 'daily_fill_rate', 'avg_fill_rate_3d', 'days_to_full', 'day_of_week', 'is_weekend', 'month']
    X = bin_data[features]
    y = bin_data['target']

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    models[bin_id] = model

    # Predict today's data (most recent entry per bin)
    today_data = bin_data.iloc[[-1]]
    pred = model.predict(today_data[features])[0]

    predictions.append({
        "trashcan_id": bin_id,
        "predicted_full": bool(pred),
        "current_fill": today_data['fill_level'].values[0],
        "daily_rate": today_data['daily_fill_rate'].values[0],
        "days_to_full": today_data['days_to_full'].values[0]
    })

# Output predictions in wide format
output_df = pd.DataFrame(predictions)

# Convert to vertical format without timestamp
vertical_df = pd.melt(output_df, 
                      id_vars=['trashcan_id'], 
                      var_name='attribute', 
                      value_name='value')

vertical_df.to_csv("predicted_bins_vertical.csv", index=False)
print("Saved vertical predictions to predicted_bins_vertical.csv")
