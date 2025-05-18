import json
import networkx as nx
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import requests

random.seed(69)

# --- Configuration ---
TRASHCAN_NUMBER = 20
TRASHCAN_NAMES = [f"trashcan_{i}" for i in range(1, TRASHCAN_NUMBER + 1)]
FILL_DAYS = 60
MIN_EMPTY_THRESHOLD = 80
FILL_RATE_RANGE = (3, 20)  # Approx daily fill rate (mean)
NOISE_STD_FRAC = 0.2  # Noise = mean_fill * this value
CITY_FILE = os.path.join(os.path.dirname(__file__), "city_map.json")
TRASHCAN_FILE = os.path.join(os.path.dirname(__file__), "trashcan_data.csv")
SERVER_ADDRESS = "http://localhost:8000"  # Change to your server URL


TRASHCAN_FILL_RATES = {}  # To store mean fill rates per trashcan

# --- Load and Parse City Map ---
with open(CITY_FILE, "r") as f:
    city_data = json.load(f)

G = nx.Graph()
for node in city_data["nodes"]:
    G.add_node(node["id"])

edge_list = []
for edge in city_data["edges"]:
    G.add_edge(edge["source"], edge["target"], weight=edge["weight"], id=edge["id"])
    edge_list.append(edge["id"])

# --- Visualize Graph ---
nx.draw(G, with_labels=True, node_size=700, node_color="lightblue", font_size=10)
plt.title("City Map Graph")
plt.show()

# --- Simulate Trashcan Fill Data ---
records = []

for trashcan in TRASHCAN_NAMES:
    edge_id = random.choice(edge_list)
    mean_fill = random.uniform(*FILL_RATE_RANGE)
    noise_std = mean_fill * NOISE_STD_FRAC
    current_fill = random.uniform(0, 30)
    TRASHCAN_FILL_RATES[trashcan] = mean_fill  # Store for later use


    day_values = []
    for _ in range(FILL_DAYS):
        daily_add = np.random.normal(loc=mean_fill, scale=noise_std)
        daily_add = np.clip(daily_add, 0, 100)

        current_fill = min(current_fill + daily_add, 100)

        # Empty condition
        if current_fill >= MIN_EMPTY_THRESHOLD:
            if random.random() < 0.3:  # 30% chance to empty
                current_fill = 0

        day_values.append(round(current_fill, 2))

    records.append([trashcan, edge_id] + day_values)

# --- Save to CSV using Pandas ---
columns = ["trashcanID", "edgeID"] + [f"Day{d+1}" for d in range(FILL_DAYS)]
df = pd.DataFrame(records, columns=columns)
df.to_csv(TRASHCAN_FILE, index=False)

print(f"[INFO] Simulated dataset saved as '{os.path.basename(TRASHCAN_FILE)}'")


def generate_fill_continuation(no_of_days=1):
    """
    Generate continuation fill levels for trashcans without emptying.

    Args:
        no_of_days (int): How many days to simulate forward.

    Returns:
        dict: {
            "trashcan_1": [val_day1, val_day2, ..., val_dayN],
            ...
        }
    """
    df = pd.read_csv(TRASHCAN_FILE)
    fill_dict = {}

    for _, row in df.iterrows():
        trashcan_id = row["trashcanID"]
        last_fill = row.iloc[2:].values[-1]  # Last recorded value
        mean_fill = TRASHCAN_FILL_RATES.get(trashcan_id, random.uniform(*FILL_RATE_RANGE))  # fallback safe
        noise_std = mean_fill * NOISE_STD_FRAC

        current_fill = last_fill
        values = []

        for _ in range(no_of_days):
            daily_add = np.random.normal(loc=mean_fill, scale=noise_std)
            daily_add = np.clip(daily_add, 0, 100)
            current_fill = min(current_fill + daily_add, 100)
            values.append(round(current_fill, 2))

        fill_dict[trashcan_id] = values if no_of_days > 1 else values[0]

    return fill_dict

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

def train_model():
    print("[INFO] Training model, please wait...")
    response = requests.get(f"{SERVER_ADDRESS}/train")
    if response.status_code == 200:
        info = response.json()
        print(f"[INFO] Training finished: {info.get('INFO')}, Time taken: {info.get('Time_taken')} seconds")
    else:
        raise Exception(f"Training failed: {response.status_code} - {response.text}")

def predict_route(latest_fill_dict):
    # Prepare JSON file for latest_data_file upload
    latest_data_json = json.dumps(latest_fill_dict)
    files = {
        "latest_data_file": ("latest_fill.json", latest_data_json, "application/json")
    }
    response = requests.post(f"{SERVER_ADDRESS}/predict", files=files)
    if response.status_code == 200:
        return response.json().get("route", [])
    else:
        raise Exception(f"Prediction failed: {response.status_code} - {response.text}")

def simulate_daily_cycle(num_days=10):
    df = pd.read_csv(TRASHCAN_FILE)

    for day in range(num_days):
        print(f"\n[INFO] Simulation day {day+1}")

        # Generate fill continuation (no emptying here)
        latest_fill = generate_fill_continuation(1)
        
        # Send to server for prediction
        route = predict_route(latest_fill)
        print(f"[INFO] Predicted truck route: {route}")

        # Map route edges from node pairs
        # The edges in route = pairs (route[0], route[1]), (route[1], route[2]), etc.
        route_edges = set()
        for i in range(len(route) - 1):
            u, v = route[i], route[i+1]
            # Get edge id from graph G for edge (u,v) or (v,u)
            if G.has_edge(u, v):
                edge_id = G[u][v]["id"]
                route_edges.add(edge_id)
            elif G.has_edge(v, u):
                edge_id = G[v][u]["id"]
                route_edges.add(edge_id)

        # Empty trashcans on edges in route_edges
        overflowed_cans = []
        for idx, row in df.iterrows():
            trashcan_id = row["trashcanID"]
            edge_id = row["edgeID"]

            current_fill = latest_fill.get(trashcan_id, None)
            if current_fill is None:
                # If missing from prediction input, error would be raised by server but let's check here too
                raise ValueError(f"Missing fill data for trashcan {trashcan_id} in latest_fill")

            # If trashcan edge in route_edges -> emptied (fill=0), else fill = predicted fill
            if edge_id in route_edges:
                new_fill = 0
            else:
                new_fill = current_fill

            # Mark overflow (say fill >= 100)
            if new_fill >= 100:
                overflowed_cans.append(trashcan_id)

            # Update dataframe for new day column
            col_name = f"Day{FILL_DAYS + day + 1}"
            if col_name not in df.columns:
                df[col_name] = np.nan  # Initialize if needed

            df.at[idx, col_name] = new_fill

        # Save updated CSV file after day update
        df.to_csv(TRASHCAN_FILE, index=False)

        # Print fill status for all trashcans for this day
        print(f"Day {FILL_DAYS + day + 1} fill levels:")
        for idx, row in df.iterrows():
            fill_val = row[f"Day{FILL_DAYS + day + 1}"]
            print(f"  {row['trashcanID']} (edge: {row['edgeID']}): {fill_val:.2f}% {'[OVERFLOW]' if row['trashcanID'] in overflowed_cans else ''}")

        if overflowed_cans:
            print(f"  [WARNING] Overflow detected in trashcans: {', '.join(overflowed_cans)}")


#################### USAGE ####################

upload_files()
train_model()
simulate_daily_cycle(5)