# Install dependencies if needed:
# !pip install overpy folium scikit-learn numpy

import overpy
import folium
import numpy as np

# Q-learning imports
import random

# Step 1: Fetch bin locations (same as before)
api = overpy.Overpass()
query = """
node["amenity"="waste_basket"](17.35,78.45,17.40,78.50);
out;
"""
result = api.query(query)

bins = []
for node in result.nodes:
    bins.append({'id': node.id, 'lat': float(node.lat), 'lon': float(node.lon)})

print(f"Fetched {len(bins)} bins")

# Step 2: Simulate fill levels (0-100%)
np.random.seed(42)
fill_levels = np.random.randint(0, 101, size=len(bins))

# Step 3: Calculate distance matrix between bins (Haversine formula)
from math import radians, cos, sin, asin, sqrt

def haversine(lat1, lon1, lat2, lon2):
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    # haversine formula
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in km
    return c * r

n_bins = len(bins)
dist_matrix = np.zeros((n_bins, n_bins))
for i in range(n_bins):
    for j in range(n_bins):
        if i != j:
            dist_matrix[i, j] = haversine(
                bins[i]['lat'], bins[i]['lon'], bins[j]['lat'], bins[j]['lon']
            )
        else:
            dist_matrix[i, j] = 0

# Step 4: Q-learning parameters
alpha = 0.1   # learning rate
gamma = 0.9   # discount factor
epsilon = 0.2 # exploration rate
episodes = 1000

# Initialize Q-table: states x actions (bins x bins)
Q = np.zeros((n_bins, n_bins))

# Reward function:
# Reward = fill_level of next bin (scaled) minus travel cost (distance)
# We'll scale fill_level to 0-10, distance cost to 0-10

fill_scaled = fill_levels / 10  # 0-10 scale

max_dist = np.max(dist_matrix)
dist_scaled = dist_matrix / max_dist * 10  # scale distance to 0-10

def reward(s, a):
    return fill_scaled[a] - dist_scaled[s, a]

# Step 5: Train Q-learning agent
for episode in range(episodes):
    state = random.randint(0, n_bins-1)  # start from random bin
    visited = set([state])
    for _ in range(n_bins - 1):
        if random.uniform(0,1) < epsilon:
            # Explore: choose random next bin not visited
            possible_actions = [a for a in range(n_bins) if a not in visited]
            if not possible_actions:
                break
            action = random.choice(possible_actions)
        else:
            # Exploit: choose best action from Q table (not visited)
            q_values = np.copy(Q[state])
            for v in visited:
                q_values[v] = -np.inf  # don't revisit
            action = np.argmax(q_values)
            if q_values[action] == -np.inf:
                break  # no unvisited actions left

        r = reward(state, action)
        # Update Q-value
        Q[state, action] = Q[state, action] + alpha * (r + gamma * np.max(Q[action]) - Q[state, action])
        visited.add(action)
        state = action

# Step 6: Extract best route from Q-table starting from bin with max fill level
start_bin = np.argmax(fill_levels)
route = [start_bin]
visited = set(route)

current = start_bin
while len(visited) < n_bins:
    q_values = np.copy(Q[current])
    for v in visited:
        q_values[v] = -np.inf
    next_bin = np.argmax(q_values)
    if q_values[next_bin] == -np.inf:
        break
    route.append(next_bin)
    visited.add(next_bin)
    current = next_bin

print("Optimal collection route (bin indices):", route)

# Step 7: Visualize route on map
m = folium.Map(location=[17.375, 78.475], zoom_start=13)

# Color bins by fill level as before
def fill_color(fill):
    if fill > 75:
        return 'red'
    elif fill > 40:
        return 'orange'
    else:
        return 'green'

# Add bin markers
for i, bin in enumerate(bins):
    folium.CircleMarker(
        location=[bin['lat'], bin['lon']],
        radius=7,
        color=fill_color(fill_levels[i]),
        fill=True,
        fill_opacity=0.7,
        popup=(f"Bin ID: {bin['id']}<br>"
               f"Fill level: {fill_levels[i]}%")
    ).add_to(m)

# Add route lines
for i in range(len(route)-1):
    start = bins[route[i]]
    end = bins[route[i+1]]
    folium.PolyLine(locations=[[start['lat'], start['lon']], [end['lat'], end['lon']]],
                    color='blue', weight=3, opacity=0.7).add_to(m)

# Add start marker
folium.Marker(
    location=[bins[start_bin]['lat'], bins[start_bin]['lon']],
    popup="Start (Highest fill)",
    icon=folium.Icon(color='green', icon='play')
).add_to(m)

m.save("bins_rl_route_map.html")
print("Map with RL optimized route saved as bins_rl_route_map.html")
