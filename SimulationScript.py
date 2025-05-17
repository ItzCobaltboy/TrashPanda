import requests
import json
import random
import time

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/predict"   # Update if hosted elsewhere
TRASHCAN_IDS = ["can1", "can2", "can3"]     # Must exactly match uploaded CSV
DAYS_TO_SIMULATE = 10
FILL_INCREMENT_RANGE = (5, 30)              # Trash fill per day per bin
EMPTY_THRESHOLD = 100                       # Max capacity (when to overflow)
SLEEP_BETWEEN_DAYS = 1                      # In seconds (for demo delay)

# --- Initialization ---
# Start with empty or predefined trash levels
trashcan_levels = {can: random.randint(0, 40) for can in TRASHCAN_IDS}

def predict_route(current_fill):
    files = {
        'latest_data_file': (
            'latest_data.json',
            json.dumps(current_fill),
            'application/json'
        )
    }
    try:
        response = requests.post(API_URL, files=files)
        if response.status_code == 200:
            return response.json().get("path", [])
        else:
            print(f"[ERROR] Status {response.status_code}: {response.text}")
            return []
    except Exception as e:
        print("[EXCEPTION] During prediction:", str(e))
        return []

# --- Simulation Loop ---
for day in range(1, DAYS_TO_SIMULATE + 1):
    print(f"\n===== Day {day} =====")

    # 1. Simulate filling
    for can in TRASHCAN_IDS:
        fill = random.randint(*FILL_INCREMENT_RANGE)
        trashcan_levels[can] = min(trashcan_levels[can] + fill, EMPTY_THRESHOLD)

    print("Trashcan levels before collection:", trashcan_levels)

    # 2. Predict today's route
    route = predict_route(trashcan_levels)
    print("Predicted route:", route)

    # 3. Find trashcans on that route (simulate collection)
    cans_collected = set()
    for edge in route:
        for can in TRASHCAN_IDS:
            if can in edge:   # Assumes trashcan ID is related to edge naming
                cans_collected.add(can)

    # 4. Empty bins visited
    for can in cans_collected:
        trashcan_levels[can] = 0

    print("Bins collected today:", cans_collected)
    print("Trashcan levels after collection:", trashcan_levels)

    time.sleep(SLEEP_BETWEEN_DAYS)  # Delay to simulate next day
