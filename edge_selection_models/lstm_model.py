import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# === 1. Load Data ===
df = pd.read_csv("/content/MyDrive/MyDrive/Colab Notebooks/small_trash.csv")  # <-- change path if needed
edge_ids = df["edgeID"].values
time_series = df.drop(columns=["edgeID", "trashcanID"]).to_numpy()

# === 2. Normalize Data ===
scaler = StandardScaler()
norm_data = scaler.fit_transform(time_series)

# === 3. Create Sequence Dataset ===
WINDOW_SIZE = 10  # <-- TUNE: Increase to use more historical days per sample
def create_sequences(data, window_size=WINDOW_SIZE):
    X, y = [], []
    for series in data:
        if len(series) < window_size + 1:
            continue  # skip if not enough data
        X.append(series[-window_size-1:-1])
        y.append(series[-1])
    return np.array(X), np.array(y)

X_all, y_all = create_sequences(norm_data)
X_all = X_all.reshape((X_all.shape[0], X_all.shape[1], 1))  # LSTM expects 3D input

# === 4. Define and Train LSTM Model ===
model = Sequential([
    LSTM(64, input_shape=(X_all.shape[1], 1)),  # <-- TUNE: You can increase units or stack LSTM layers
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

EPOCHS = 10  # <-- TUNE: Increase epochs for bigger datasets, or use EarlyStopping
BATCH_SIZE = 32  # <-- TUNE: May adjust for memory and convergence speed
model.fit(X_all, y_all, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# === 5. Predict on Full Dataset ===
y_pred = model.predict(X_all, verbose=0).flatten()

# Inverse transform to get back original scale
y_actual_real = time_series[:, -1]
y_pred_real = y_pred * scaler.scale_[-1] + scaler.mean_[-1]

# === 6. Calculate Accuracy Metrics ===
mae = mean_absolute_error(y_actual_real, y_pred_real)
rmse = np.sqrt(mean_squared_error(y_actual_real, y_pred_real))

print(f"\nModel Accuracy:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# === 7. Apply Percentile-Based Classification ===
# <-- TUNE: You can adjust the thresholds or method if needed
p95 = np.percentile(y_pred_real, 95)
p98 = np.percentile(y_pred_real, 98)

selection_dict = {}
must_visit_ids = []

for i, edge_id in enumerate(edge_ids):
    val = y_pred_real[i]
    if val >= p98:
        label = 2  # Must Visit
        must_visit_ids.append(edge_id)
    elif val >= p95:
        label = 1  # Should Visit
    else:
        label = 0  # Skip
    selection_dict[str(edge_id)] = label

print(f"\nTotal Must Visit (label 2): {len(must_visit_ids)}")
print("Edge IDs marked as MUST VISIT:")
print(must_visit_ids)

# === 8. Optional: Plot Predicted vs Actual ===
# <-- TUNE: Adjust N to control how many samples to visualize
N = min(100, len(y_pred_real))
plt.figure(figsize=(10, 4))
plt.plot(y_actual_real[:N], label="Actual", marker='o')
plt.plot(y_pred_real[:N], label="Predicted", marker='x')
plt.title("Predicted vs Actual Trash Fill Levels")
plt.xlabel("Sample Index")
plt.ylabel("Trash Fill Level")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
