import json
import networkx as nx
import random
import numpy as np
import pandas as pd
import os
import requests
from datetime import datetime, timedelta

# --- Configuration ---
random.seed(69)
np.random.seed(69)

TRASHCAN_NUMBER = 20
TRASHCAN_NAMES = [f"trashcan_{i}" for i in range(1, TRASHCAN_NUMBER + 1)]
FILL_DAYS = 200
MIN_EMPTY_THRESHOLD = 80
FILL_RATE_RANGE = (3, 20)  # Daily base fill rate range
NOISE_STD_FRAC = 0.2  # Noise on daily fill
SEASONAL_MEAN_NOISE_FRAC = 0.1  # Noise on mean fill
START_DATE = datetime(2023, 1, 1)

CITY_FILE = os.path.join(os.path.dirname(__file__), "city_map.json")
TRASHCAN_FILE = os.path.join(os.path.dirname(__file__), "trashcan_data.csv")
SERVER_ADDRESS = "http://localhost:8000"  # Change this if needed

TRASHCAN_FILL_RATES = {}  # To store base mean fills per trashcan

# --- Load City Graph ---
with open(CITY_FILE, "r") as f:
    city_data = json.load(f)

G = nx.Graph()
for node in city_data["nodes"]:
    G.add_node(node["id"])

edge_list = []
for edge in city_data["edges"]:
    G.add_edge(edge["source"], edge["target"], weight=edge["weight"], id=edge["id"])
    edge_list.append(edge["id"])

# --- Helper: Date list ---
date_list = [START_DATE + timedelta(days=i) for i in range(FILL_DAYS)]
date_columns = [d.strftime("%Y-%m-%d") for d in date_list]

# --- Helper: Seasonal fill modifier ---
def get_seasonal_mean_fill(base_fill, date):
    is_weekend = date.weekday() >= 5  # Saturday, Sunday
    seasonal_multiplier = 1.2 if is_weekend else 1.0
    seasonal_fill = base_fill * seasonal_multiplier

    noisy_seasonal_fill = np.random.normal(loc=seasonal_fill, scale=seasonal_fill * SEASONAL_MEAN_NOISE_FRAC)
    return np.clip(noisy_seasonal_fill, 0, 100)

# --- Simulate Trashcan Fill Data ---
records = []

for trashcan in TRASHCAN_NAMES:
    edge_id = random.choice(edge_list)
    base_mean_fill = random.uniform(*FILL_RATE_RANGE)
    TRASHCAN_FILL_RATES[trashcan] = base_mean_fill

    current_fill = random.uniform(0, 30)
    day_values = []

    for date in date_list:
        seasonal_mean = get_seasonal_mean_fill(base_mean_fill, date)
        noise_std = seasonal_mean * NOISE_STD_FRAC
        daily_add = np.random.normal(loc=seasonal_mean, scale=noise_std)
        daily_add = np.clip(daily_add, 0, 100)

        current_fill = min(current_fill + daily_add, 100)

        # Possibly empty
        if current_fill >= MIN_EMPTY_THRESHOLD:
            if random.random() < 0.3:
                current_fill = 0

        day_values.append(round(current_fill, 2))

    records.append([trashcan, edge_id] + day_values)

# --- Save DataFrame ---
columns = ["trashcanID", "edgeID"] + date_columns
df = pd.DataFrame(records, columns=columns)
df.to_csv(TRASHCAN_FILE, index=False)
print(f"[INFO] Simulated dataset saved as '{os.path.basename(TRASHCAN_FILE)}'")


# --- Fill Generator for New Days ---
def generate_fill_continuation(no_of_days=1):
    df = pd.read_csv(TRASHCAN_FILE)
    latest_cols = df.columns[2:]
    last_date = datetime.strptime(latest_cols[-1], "%Y-%m-%d")
    date_list = [last_date + timedelta(days=i + 1) for i in range(no_of_days)]

    fill_dict = {}

    for _, row in df.iterrows():
        trashcan_id = row["trashcanID"]
        last_fill = row[latest_cols[-1]]
        base_fill = TRASHCAN_FILL_RATES.get(trashcan_id, random.uniform(*FILL_RATE_RANGE))
        current_fill = last_fill
        values = []

        for date in date_list:
            seasonal_mean = get_seasonal_mean_fill(base_fill, date)
            noise_std = seasonal_mean * NOISE_STD_FRAC
            daily_add = np.random.normal(loc=seasonal_mean, scale=noise_std)
            daily_add = np.clip(daily_add, 0, 100)
            current_fill = min(current_fill + daily_add, 100)
            values.append(round(current_fill, 2))

        fill_dict[trashcan_id] = values if no_of_days > 1 else values[0]

    return fill_dict, date_list


# --- Upload Files to Server ---
def upload_files():
    with open(CITY_FILE, "rb") as city_f, open(TRASHCAN_FILE, "rb") as trashcan_f:
        files = {
            "city_map": ("city_map.json", city_f, "application/json"),
            "trashcan_data": ("trashcan_data.csv", trashcan_f, "text/csv")
        }
        response = requests.post(f"{SERVER_ADDRESS}/upload", files=files)
    if response.status_code == 200:
        print("[INFO] Files uploaded successfully.")
    else:
        raise Exception(f"Upload failed: {response.status_code} - {response.text}")


# --- Train Model on Server ---
def train_model():
    print("[INFO] Training model, please wait...")
    response = requests.get(f"{SERVER_ADDRESS}/train")
    if response.status_code == 200:
        info = response.json()
        print(f"[INFO] Training finished: {info.get('INFO')}, Time taken: {info.get('Time_taken')} seconds")
    else:
        raise Exception(f"Training failed: {response.status_code} - {response.text}")


# --- Predict Truck Route ---
def predict_route(latest_fill_dict, day_label):
    latest_data_json = json.dumps(latest_fill_dict)
    files = {
        "latest_data_file": ("latest_fill.json", latest_data_json, "application/json"),
        "start_node": (None, "Node1"),
        "day_name": (None, f"{day_label}")
    }
    response = requests.post(f"{SERVER_ADDRESS}/predict", files=files)
    if response.status_code == 200:
        return response.json().get("route", []), response.json().get("cost", -1), response.json().get("reward", -1)
    else:
        raise Exception(f"Prediction failed: {response.status_code} - {response.text}")


# --- Main Simulation Loop ---
def simulate_daily_cycle(num_days=10):
    df = pd.read_csv(TRASHCAN_FILE)
    operating_cost = 0
    operating_reward = 0

    for day in range(num_days):
        print(f"\n[INFO] Simulation day {day + 1}")

        latest_fill, new_dates = generate_fill_continuation(1)
        new_date_str = new_dates[0].strftime("%Y-%m-%d")

        route, round_cost, round_reward = predict_route(latest_fill, new_date_str)
        print(f"[INFO] Predicted truck route: {route}")

        route_edges = set()
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            if G.has_edge(u, v):
                edge_id = G[u][v]["id"]
                route_edges.add(edge_id)
            elif G.has_edge(v, u):
                edge_id = G[v][u]["id"]
                route_edges.add(edge_id)

        overflowed_cans = []
        emptied_cans = []

        df[new_date_str] = np.nan  # Add new column for the date

        for idx, row in df.iterrows():
            trashcan_id = row["trashcanID"]
            edge_id = row["edgeID"]
            current_fill = latest_fill.get(trashcan_id, None)

            if current_fill is None:
                raise ValueError(f"Missing fill data for {trashcan_id}")

            if edge_id in route_edges:
                new_fill = 0
                emptied_cans.append(trashcan_id)
            else:
                new_fill = current_fill

            if new_fill >= 100:
                overflowed_cans.append(trashcan_id)

            df.at[idx, new_date_str] = new_fill

        df.to_csv(TRASHCAN_FILE, index=False)

        print(f"Fill levels on {new_date_str}:")
        for idx, row in df.iterrows():
            fill_val = row[new_date_str]
            print(f"  {row['trashcanID']} (edge: {row['edgeID']}): {fill_val:.2f}% {'[OVERFLOW]' if row['trashcanID'] in overflowed_cans else ''} {'[EMPTIED]' if row['trashcanID'] in emptied_cans else ''}")

        if overflowed_cans:
            print(f"  [WARNING] Overflow detected: {', '.join(overflowed_cans)}")
        if round_cost == -1 or round_reward == -1:
            print(f"  [ERROR] Invalid prediction. Cost: {round_cost}, Reward: {round_reward}")
            continue

        print(f"  [INFO] Round cost: {round_cost}, Round reward: {round_reward}")
        operating_cost += round_cost
        operating_reward += round_reward

    print(f"\n[INFO] Total operating cost: {operating_cost}")
    print(f"[INFO] Total operating reward: {operating_reward}")


#################### USAGE ####################

upload_files()
train_model()
simulate_daily_cycle(50)
