import pandas as pd
import os
import numpy as np
import random
import json

traffic_input_file = os.path.join(os.path.dirname(__file__), 'big_traffic.csv')
trash_input_file = os.path.join(os.path.dirname(__file__), 'big_trash.csv')
trash_source_file = os.path.join(os.path.dirname(__file__), '..','synthetic_trashcan_fill_levels2.csv')


# Manipulate the traffic data
tf = pd.read_csv(traffic_input_file)
tf.loc[:, 'EdgeID'] = [f"edge{i}" for i in range(len(tf))]

tf.to_csv(traffic_input_file, index=True)


# Manipulate the trash data
df = pd.read_csv(trash_input_file)
trash_source = pd.read_csv(trash_source_file)

target_range = range(0, 99)  # Assuming you want to replace with values from edge1 to edge100
target_ids = [f"edge{i}" for i in target_range]

# select random 400 rows from the source DataFrame
random_rows = trash_source.sample(n=400, random_state=542)


# Update the first column with random values from target_ids
random_rows.loc[:, 'edgeID'] = [random.choice(target_ids) for _ in range(len(random_rows))]
random_rows.loc[:, 'trashcanID'] = [f"trashcan_{i}" for i in range(len(random_rows))]

random_rows.to_csv(trash_input_file, index=True)

# generate a JSON file for city map with 99 edges

target_nodes = [f"Node{i}" for i in range(1, 100)]


city_map = {
    'nodes': [
        {"id": f"Node{i}"} for i in range(1, 100)
    ],
    'edges': [
        {
            "source": f"Node{random.randint(1,100)}",
            "target": f"Node{random.randint(1,100)}",  # Connect to the next node in a circular manner
            "weight": random.randint(1, 80) , # Random weight for the edge
            "id": f"edge{i}"
        } for i in range(1, 100)
    ]
}

city_map_file = os.path.join(os.path.dirname(__file__), 'city_map.json')
with open(city_map_file, 'w') as f:
    json.dump(city_map, f, indent=4)

