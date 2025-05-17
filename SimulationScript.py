import requests
import json

BASE_URL = "http://127.0.0.1:8000"

# --- Step 1: Upload Files ---
def upload_data():
    files = {
        "city_map": open("data/city_map.json", "rb"),
        "trashcan_data": open("data/trashcan.csv", "rb")
    }
    response = requests.post(f"{BASE_URL}/upload", files=files)
    print("Upload:", response.json())

# --- Step 2: Train Models ---
def train_models():
    response = requests.get(f"{BASE_URL}/train")
    print("Train:", response.json())

# --- Step 3: Predict ---
def predict_route():
    # Example today's trash values for prediction
    latest_data = {
        "can1": 70,
        "can2": 65,
        "can3": 40,
        "can4": 90,
        "can5": 45
    }
    files = {
        "latest_data_file": ("latest.json", json.dumps(latest_data), "application/json")
    }
    response = requests.post(f"{BASE_URL}/predict", files=files)
    print("Predicted Path:", response.json())

if __name__ == "__main__":
    upload_data()
    train_models()
    predict_route()
