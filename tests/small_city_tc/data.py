# import pandas as pd

# # Load the CSV
# df = pd.read_csv("synthetic_trashcan_fill_levels2.csv")

# # Compute average fill level (assuming date columns start at index 2)
# df["avg_fill"] = df.iloc[:, 2:].mean(axis=1)

# # Sort by average fill level (lowest first)
# sorted_df = df.sort_values(by="avg_fill")

# # Save the top 20 greenest trashcans
# sorted_df.head(20).to_csv("greenest_trashcans.csv", index=False)


# import pandas as pd

# # Load the CSV
# df = pd.read_csv("synthetic_traffic_congestion.csv")

# # Compute average fill level (assuming date columns start at index 2)
# df["avg_fill"] = df.iloc[:, 2:].mean(axis=1)

# # Sort by average fill level (lowest first)
# sorted_df = df.sort_values(by="avg_fill")

# # Save the top 20 greenest trashcans
# sorted_df.head(20).to_csv("greenest_trashcans.csv", index=False)


import pandas as pd
import os
import numpy as np
import random

# Path to your main Excel file
input_file_trash = os.path.join(os.path.dirname(__file__), 'small_trash.csv')

# Column name where the ID exists (e.g., 'ID' or 'id')
id_column = 'edgeID'

# List of 20 specific IDs to search for
target_ids = [
    'edge1','edge2','edge3','edge4','edge5',
    'edge6','edge7','edge8','edge9','edge10',
    'edge11','edge12','edge13','edge14','edge15',
    'edge16','edge17','edge18','edge19','edge20'
]

# Load the Excel data
df = pd.read_csv(input_file_trash)

# Update first column with random values from target_values
df.loc[:, 'edgeID'] = [random.choice(target_ids) for _ in range(len(df))]
df.loc[:, 'trashcanID'] = [f"trashcan_{i}" for i in range(len(df))]

df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'],inplace=True, errors='ignore')

# print(df.head(10))
df.to_csv(input_file_trash, index=True)

input_file_traffic = os.path.join(os.path.dirname(__file__), 'small_traffic.csv')

tf = pd.read_csv(input_file_traffic)
tf.loc[:, 'EdgeID'] = [f'edge{i}' for i in range(len(tf))]

tf.to_csv(input_file_traffic, index=True)
