# import pandas as pd

# # Step 1: Load the traffic CSV file
# traffic_df = pd.read_csv('synthetic_traffic_congestion.csv')

# # Step 2: Compute total and average congestion for each EdgeID
# # Exclude 'EdgeID' column, apply numeric sum/mean row-wise
# congestion_values = traffic_df.drop(columns=['EdgeID'])

# traffic_df['TotalCongestion'] = congestion_values.sum(axis=1)
# traffic_df['AvgCongestion'] = congestion_values.mean(axis=1)

# # Step 3: Select Top 100 based on either Total or Average
# top_100 = traffic_df.nlargest(100, 'TotalCongestion')  # Change to 'AvgCongestion' if needed

# # Step 4: Save the selected top 100 edges to new CSV
# top_100.drop(columns=['TotalCongestion', 'AvgCongestion']).to_csv('big_traffic.csv', index=False)

# # Step 5: Load the trashcan.csv file
# trashcan_df = pd.read_csv('synthetic_traffic_congestion.csv')

# # Step 6: Filter rows where EdgeID is in top 100 list
# top_edge_ids = top_100['edgeID']
# filtered_trashcan_df = trashcan_df[trashcan_df['EdgeID'].isin(top_edge_ids)]

# # Step 7: Save filtered trashcan data
# filtered_trashcan_df.to_csv('filtered_trashcan.csv', index=False)



import pandas as pd

# Step 1: Load the 100 EdgeIDs from big_traffic.csv
big_traffic_df = pd.read_csv("big_traffic.csv")
top_edge_ids = big_traffic_df["EdgeID"].unique()

# Step 2: Load the trashcan data
trashcan_df = pd.read_csv("synthetic_trashcan_fill_levels2.csv")

# Step 3: Filter rows where EdgeID is in the top 100 list
filtered_trashcan_df = trashcan_df[trashcan_df["edgeID"].isin(top_edge_ids)]

# Step 4: Save to big_trash.csv
filtered_trashcan_df.to_csv("big_trash.csv", index=False)

print("Filtered trashcan data saved to big_trash.csv")
