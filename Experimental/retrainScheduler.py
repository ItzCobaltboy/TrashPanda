from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
import subprocess
import pandas as pd
import numpy as np
import os
import yagmail
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# === Config ===
EMAIL_USER = "your_email@gmail.com"
EMAIL_PASS = "your_app_password"
EMAIL_RECEIVER = "your_email@gmail.com"

TRAIN_SCRIPT = "train_model.py"
DATA_PATH = "trashcan_data.csv"
THRESHOLD_DRIFT = 0.10  # 10% accuracy drop allowed

# === Email Alert ===
def send_email(subject, body):
    yag = yagmail.SMTP(EMAIL_USER, EMAIL_PASS)
    yag.send(to=EMAIL_RECEIVER, subject=subject, contents=body)

# === Check model drift ===
def detect_drift():
    try:
        df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
        df = df.sort_values(by=["trashcan_id", "timestamp"])

        accuracies = []

        for bin_id in df['trashcan_id'].unique():
            bin_data = df[df['trashcan_id'] == bin_id].copy()
            bin_data['fill_diff'] = bin_data['fill_level'].diff().fillna(0)
            bin_data['avg_rate'] = bin_data['fill_diff'].rolling(3).mean().fillna(0)
            bin_data['future'] = bin_data['fill_level'].shift(-2)
            bin_data['target'] = (bin_data['future'] >= 80).astype(int)
            bin_data = bin_data.dropna()

            if len(bin_data) < 8:
                continue

            features = ['fill_level', 'fill_diff', 'avg_rate']
            X = bin_data[features]
            y = bin_data['target']

            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            model.fit(X[:-1], y[:-1])
            y_pred = model.predict(X[-5:])
            acc = accuracy_score(y[-5:], y_pred)
            accuracies.append(acc)

        avg_acc = np.mean(accuracies)
        print(f"[{datetime.now()}] 🧠 Model Accuracy Check: {avg_acc:.2f}")

        return avg_acc < (1 - THRESHOLD_DRIFT)

    except Exception as e:
        print(f"[{datetime.now()}] ❌ Drift check failed: {e}")
        send_email("TrashPanda: Drift Check Failed", str(e))
        return True  # assume retraining needed

# === Retraining logic ===
def retrain_if_needed():
    try:
        if detect_drift():
            print(f"[{datetime.now()}] 🔁 Retraining triggered...")
            subprocess.run(["python", TRAIN_SCRIPT], check=True)
            send_email("✅ TrashPanda Retrained", "Model retrained due to drift or schedule.")
        else:
            print(f"[{datetime.now()}] ✅ No retraining needed. Model healthy.")
    except Exception as e:
        print(f"[{datetime.now()}] ❌ Retraining failed: {e}")
        send_email("❌ TrashPanda Retrain Failed", str(e))

# === Scheduler Setup ===
scheduler = BlockingScheduler()
scheduler.add_job(retrain_if_needed, 'cron', hour=2)  # daily at 2AM

print(f"[{datetime.now()}] 🚀 TrashPanda Retrain Scheduler started...")
try:
    scheduler.start()
except (KeyboardInterrupt, SystemExit):
    print("Scheduler stopped.")
