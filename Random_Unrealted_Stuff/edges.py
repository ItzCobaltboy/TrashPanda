import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import timedelta

# Load Data
data = pd.read_csv("trashcan_data.csv")
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Parameters
history_days = 3  # Number of past days to use as features
fill_threshold = 0.85  # Full bin threshold

# Step 1: Prepare Training Data
bins = data['bin_id'].unique()
features = []
labels = []

for bin_id in bins:
    bin_data = data[data['bin_id'] == bin_id].sort_values('timestamp')
    for i in range(history_days, len(bin_data)):
        past_levels = bin_data.iloc[i-history_days:i]['fill_level'].values
        today_fill = bin_data.iloc[i]['fill_level']
        label = "Full" if today_fill >= 0.85 else "FullSoon" if today_fill >= 0.65 else "OK"
        features.append(past_levels)
        labels.append(label)

# Step 2: Train KNN Model
X = pd.DataFrame(features)
y = LabelEncoder().fit_transform(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Step 3: Predict Today’s Status
# Get the latest fill levels for each bin
latest_date = data['timestamp'].max()
today_data = []

for bin_id in bins:
    bin_history = data[data['bin_id'] == bin_id].sort_values('timestamp')
    if len(bin_history) >= history_days:
        past_levels = bin_history.iloc[-history_days:]['fill_level'].values
        today_data.append((bin_id, past_levels))

# Predict
results = []
for bin_id, features in today_data:
    pred_label_encoded = knn.predict([features])[0]
    pred_label = LabelEncoder().fit(["OK", "FullSoon", "Full"]).inverse_transform([pred_label_encoded])[0]
    results.append((bin_id, pred_label))

# Output: Full Today
print("🔴 Must Visit (Full Now):")
for bin_id, status in results:
    if status == "Full":
        print(f"Bin {bin_id}")

print("\n🟡 Optional Visit (Full Soon):")
for bin_id, status in results:
    if status == "FullSoon":
        print(f"Bin {bin_id}")
