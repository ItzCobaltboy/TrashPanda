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

# Path to your main Excel file
input_file = 'synthetic_trashcan_fill_levels2.csv'

# Column name where the ID exists (e.g., 'ID' or 'id')
id_column = 'edgeID'

# List of 20 specific IDs to search for
target_ids = [
    'E2659','E3646','E2936','E3064','E167',
    'E1751','E2984','E2160','E2445','E2938',
    'E3915','E3730','E3936','E2592','E2805',
    'E2110','E2692','E1423','E2446','E316'
]

# Load the Excel data
df = pd.read_csv(input_file)

# Filter rows with matching IDs
filtered_df = df[df[id_column].isin(target_ids)]

# Save to a new CSV
filtered_df.to_csv('filtered_rows.csv', index=False)

print("Filtered rows saved to 'filtered_rows.csv'")
