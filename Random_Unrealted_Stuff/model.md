# Model H5 goes here

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import math
import random

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# --- Generate synthetic historical data for multiple dustbins with emptying events ---
def generate_dustbin_data(num_bins=5, time_steps=100):
    data = {}
    for bin_id in range(num_bins):
        noise = np.random.normal(0, 3, time_steps)
        trend = np.linspace(0, 1, time_steps) * 100
        series = (np.sin(np.arange(time_steps)/10 + bin_id) + 1) * 40 + noise + trend

        # Simulate emptying every 20 steps
        for t in range(0, time_steps, 20):
            series[t] = 0

        series = np.clip(series, 0, 100)
        data[bin_id] = series
    return data

# --- Prepare LSTM data ---
def prepare_lstm_data(series, n_steps=10):
    X, y = [], []
    for i in range(len(series) - n_steps):
        X.append(series[i:i + n_steps])
        y.append(series[i + n_steps])
    return np.array(X), np.array(y)

# --- Build LSTM model ---
def build_lstm(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# --- Predict next fill level ---
def predict_fill(model, last_seq, steps=1):
    preds = []
    seq = last_seq.copy()
    for _ in range(steps):
        input_seq = seq.reshape(1, -1, 1)
        pred = model.predict(input_seq, verbose=0)[0][0]
        preds.append(pred)
        seq = np.append(seq[1:], [[pred]], axis=0)
    return np.array(preds)

# --- Generate dustbin locations ---
def generate_locations(num_bins=5):
    return {i: (random.randint(0, 100), random.randint(0, 100)) for i in range(num_bins)}

# --- Euclidean distance ---
def euclidean(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# --- Build MST based on predicted full bins ---
def build_mst(locations):
    G = nx.Graph()
    for i in locations:
        G.add_node(i, pos=locations[i])
    keys = list(locations.keys())
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            dist = euclidean(locations[keys[i]], locations[keys[j]])
            G.add_edge(keys[i], keys[j], weight=dist)
    mst = nx.minimum_spanning_tree(G)
    return G, mst

# --- Visualize MST ---
def plot_mst(G, mst):
    pos = nx.get_node_attributes(G, 'pos')
    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, with_labels=True, node_color='lightgray', edge_color='gray')
    nx.draw(mst, pos, with_labels=True, node_color='green', edge_color='red', width=2)
    plt.title("Predicted Full Dustbins - MST")
    plt.show()

# --- Main ---
def main():
    num_bins = 7
    horizon = 10  # Predict for the next 10 days
    threshold = 60  # Collection threshold

    data = generate_dustbin_data(num_bins)
    locations = generate_locations(num_bins)
    scaler = MinMaxScaler()

    predicted_levels = {}
    to_visit = []  # To store visit status for each bin

    for bin_id, series in data.items():
        # Preprocess the data for LSTM
        scaled_series = scaler.fit_transform(series.reshape(-1, 1))
        X, y = prepare_lstm_data(scaled_series, n_steps=10)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Train the LSTM model for each bin
        model = build_lstm((X.shape[1], 1))
        model.fit(X, y, epochs=10, batch_size=8, verbose=0)

        # Predict future fill level (horizon days ahead)
        last_seq = scaled_series[-10:]
        preds = predict_fill(model, last_seq, steps=horizon)  # Predict for next 'horizon' days

        # Get the predicted fill level for the next day (you can change this as per your need)
        predicted_levels[bin_id] = preds[0]  # Let's assume we care only about the first predicted day

        # Check if the bin needs to be collected (above threshold)
        if predicted_levels[bin_id] >= threshold:
            to_visit.append(1)  # Visit this bin
        else:
            to_visit.append(0)  # Don't visit this bin

    print("Predicted Fill Levels:", predicted_levels)
    print("Bins to Collect:", to_visit)

    # Build and plot MST if there are bins to collect
    full_bins = {bin_id: locations[bin_id] for bin_id, visit in zip(range(num_bins), to_visit) if visit == 1}
    
    if len(full_bins) >= 2:
        G, mst = build_mst(full_bins)
        plot_mst(G, mst)
    else:
        print("Not enough full bins to form MST")

if __name__ == "__main__":
    main()
