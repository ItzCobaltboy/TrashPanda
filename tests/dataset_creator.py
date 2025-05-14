# The script to create a dataset for the TrashPanda model, artificially
# DataSet design

# Graph for city map in JSON
# Time series CSV for each trashcan and road


import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Config
num_trashcans = 4000
num_days = 200
start_date = datetime(2025, 1, 1)
dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(num_days)]
columns = ['edgeID', 'trashcanID'] + dates

def generate_behavior_pattern(behavior_type, num_days):
    fill = []
    current = 0
    day = 0

    while day < num_days:
        if current >= 100:
            current = 0  # Simulate trash pickup

        if behavior_type == 'normal':
            step_base = np.random.randint(1, 5)
            interval = np.random.randint(3, 7)
            for _ in range(interval):
                step = step_base + np.random.normal(0, 1)  # slight variation
                current = min(100, current + max(0, int(round(step))))
                fill.append(current)
                day += 1
                if day >= num_days:
                    break

        elif behavior_type == 'absent':
            interval = np.random.randint(5, 15)
            for _ in range(interval):
                fill.append(current)
                day += 1
                if day >= num_days:
                    break
            behavior_type = 'normal'

        elif behavior_type == 'party':
            spikes = np.random.randint(1, 3)
            for _ in range(spikes):
                spike = np.random.randint(25, 60) + np.random.normal(0, 5)
                current = min(100, current + max(0, int(round(spike))))
                fill.append(current)
                day += 1
                if day >= num_days:
                    break
            behavior_type = 'normal'

        elif behavior_type == 'fast':
            step_base = np.random.randint(10, 18)
            interval = np.random.randint(3, 6)
            for _ in range(interval):
                step = step_base + np.random.normal(0, 2)
                current = min(100, current + max(0, int(round(step))))
                fill.append(current)
                day += 1
                if day >= num_days:
                    break

        elif behavior_type == 'erratic':
            for _ in range(np.random.randint(4, 10)):
                change = np.random.randint(-3, 10) + np.random.normal(0, 2)
                current = min(100, max(0, current + int(round(change))))
                fill.append(current)
                day += 1
                if day >= num_days:
                    break
            behavior_type = 'normal'

    return fill


# Generate data
data = []
behavior_types = ['normal', 'absent', 'party', 'fast', 'erratic']

for i in range(1, num_trashcans + 1):
    edge_id = f"E{i}"
    trashcan_id = f"T{i}"
    behavior = random.choice(behavior_types)
    fill_levels = generate_behavior_pattern(behavior, num_days)
    row = [edge_id, trashcan_id] + fill_levels
    data.append(row)

# Create and save DataFrame
df = pd.DataFrame(data, columns=columns)
df.to_csv("synthetic_trashcan_fill_levels2.csv", index=False)
print("Synthetic dataset saved as 'synthetic_trashcan_fill_levels2.csv'")
