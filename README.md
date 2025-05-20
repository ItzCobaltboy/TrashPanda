# TrashPanda
A Repository for an ML based solution to Trash Truck routing for real world applications. A Custom model defined on Tensorflow for accurate prediction of Trash Truck's routing for a city, depending on parameters of how fast the trashcans at each street fill up and traffic, such that the operating costs for the company is minimal.

The software is built to be ready to deploy with API endpoints built using FastAPI and telemetry implemented using PostgreSQL.

The repository also contains some sample input data for testing the system
<br>

---

# Features

### LSTM-Based Trash Fill Prediction  
- Predicts individual trashcan fill levels using an LSTM model trained on simulated time-series data.  
- Window size is fully configurable to adapt to different prediction horizons or sensor sampling rates.

### City-Scale Simulation Environment  
- Simulates dynamic urban environments with customizable parameters:  
  - Number of streets and bins  
  - Trash generation patterns  
- Designed to evaluate route optimization strategies in varied scenarios.

### Heuristic-Based Routing Algorithm  
- Implements a cost-vs-reward decision tree for route generation.  
- Step-by-step:  
  1. Identify mandatory pickups (bins close to full).  
  2. Generate a baseline route.  
  3. For each nearby bin, predict future fill state.  
  4. If the projected reward exceeds the detour cost, include it in the final route.  
- Balances fuel/time cost with overflow prevention and efficiency.

### FastAPI Backend  
- Provides RESTful endpoints for control and operations.  
- Easily integrates with visualizations or external dashboards.

### GPU-Accelerated Batch Training  
- Automatically detects GPU availability and offloads training if possible.  
- Supports parallel training of multiple models in configurable batches.  
- Designed for rapid experimentation and ensemble model training.  
- Tested on RTX 3050 Ti Mobile 4GB with concurrent training of up to 20 LSTM models.
- In case of no GPU, System automatically resorts to sequential model training to not overload the system.

### Configurable and Extensible Architecture  
- Modular codebase with clear separation of concerns (simulation, model, API, routing).  
- Centralized configuration for model parameters, path planner settings, and system behavior.  
- Easy to extend with new models, routing strategies, or visualization layers.
- Togglable extended debug logging for Dev support.

### Experiment-Ready Setup  
- Designed for iterative experimentation with routing heuristics and prediction models.  
- Outputs include raw logs, simulation state dumps, and performance metrics.

---

# Core requirements

- Python 3.10.x (otherwise Tensorflow will break)
- Tensorflow 2.12
- FastAPI
- Numpy/Pandas
- PostgreSQL for database (Optional)

#### Dev Tools
- Postman (for pinging)

<br>

---

# Installation from Source
## Clone Repository and Setup virtual environment

```
git clone <repository_link>
cd TrashPanda
python -m venv venv
venv\Scripts\activate
```
-- Replace `<repository_link>` with the correct link
-- use `deactivate` command to deactivate Virtual envirnment

## Dependancies

```
pip install -r requirements.txt
```
This will install all the required dependancies 


## Running
```
python main.py
```

---

# Running Sample Simulations

We have provided some sample city simulations that you can utilise to test the system, visit `tests` and you will see some folders with test cases

Each testcase has 3 parts, `city_map.json` is the city map defined by the user, `simulation_script` has parameters like `TRASHCAN_NUMBER` and `SIMULATION_DAYS` as configurations, the script will automatically analyse the city map and generate a dataset for it depending on parameters and it will ping the given host address to TrashPanda, and simulate its responses.

The Simulation script is universal and will work with any valid city_map provided to it, thus it also serves as a Template for testcases

---

# API Endpoint Structure

## **1. POST `/upload`**
### Description

Upload the city map (.json) and trashcan data (.csv) files. These are validated and stored for model training and predictions.

### Request
- Content Type: `multipart/form-data`
- Fields: 
    -`city_map`: JSON file describing city structure
    -`trashcan_data`: CSV file mapping trashcan to edges, and storing time series data for training across coloumns
```
city_map.json
{
  "nodes": [
    {"id": "Node1"}, {"id": "Node2"}, {"id": "Node3"},
    {"id": "Node4"}, {"id": "Node5"}, {"id": "Node6"},
    {"id": "Node7"}, {"id": "Node8"}
  ],
  "edges": [
    {"source": "Node1", "target": "Node2", "weight": 1, "id": "edge1"},
    {"source": "Node2", "target": "Node3", "weight": 2, "id": "edge2"},
    {"source": "Node3", "target": "Node4", "weight": 3, "id": "edge3"},
    {"source": "Node4", "target": "Node5", "weight": 4, "id": "edge4"},
    {"source": "Node5", "target": "Node6", "weight": 5, "id": "edge5"},
    {"source": "Node6", "target": "Node7", "weight": 6, "id": "edge6"},
    {"source": "Node7", "target": "Node8", "weight": 7, "id": "edge7"}
  ]
}
```

```
trashcan.csv
EdgeID,TrashcanID,Day 1, Day 2, Day 3
edge1,can1,30,40,50
edge2,can2,60,70,80
edge3,can3,10,20,30

```

### Validations
- File types:
    - `city_map` must be a JSON file.
    - `trashcan_data` must be a CSV file.
- Data Integrity:
    - `city_map`: Must be a valid map with `weight` and `id` as attributes for edges, and edges must be valid.
    - `trashcan_data`: All trashcans should exist on valid edgeID in the corrosponding `city_map`

### Expected output
```
{
"INFO": "Files uploaded Successfully"
}
```
## **2. GET `/train`**
### Description
Requests the System to train the models based on latest provided data, will return error if data is not provided

NOTE: The system will take decent amount of time to train depending on dataset size, HTTP 200 is sent after successful training, expect long request time without timeout

### Request
- None, empty request

### Expected output
```
{
    "INFO": "Models Trained Successfully",
    "Time_taken": time_in_seconds
}
```

## **3. POST `/predict`**
### Description
Predicts trashcan fill levels and returns an optimized path for collection. For latest data to be provided alongside

### Request
- Content Type: `multipart/form-data`
- Fields:  
    - `latest_data_file`: JSON file with key value pairs for new values of trash for each trashcan
```
{
    "can1": 70,
    "can2": 69,
    "can3": 68
}
```

### Validations 
1) `latest_data_file` must be a JSON file.
2) For each ID:value pair in file, ID should corrospond to a valid trashcan defined in masterdata set during upload, for any missing IDs, error will be returned

### Output
System will process the data and make a wise decision on path planning (Described in next topics)
```
{
  "route": ["Node1", "Node2", "Node3", "Node4"]
}
```

---

# System Architecture – Trash Collection Optimization API

This system is designed to process city infrastructure and real-time trashcan data to **predict trashcan fill levels** and generate **optimized collection routes** for large scale use.



## Overview

The system is composed of the following key modules:

### **1. `EdgeSelector`** (Core Module)
Working:
- Setups, initializes models for each trashcan in dataset, and trains a model for each trashcan using parallel training if GPU is present using tensorflow.
- Divides all trashcans into three categories namely `must visit (MV)`, `visit if worth it (VIWI)`, and `no urgency (NU)` based on two informations.
    1) `Latest trashcan fills` for today, given by user.
    2) `Predicted trashcan fills` predicted individually for each trashcan with its own model, these trashcans are not full now but will get full soon hence can be visited if worht it
- Trashcans that will Overflow soon are alwas added to `MV` category, reducing chances of overflowing.
- Trashcans that are not full now but predicted to be full soon are added to `VIWI` category.
- Dynamically selects and returns `Edge List` and `Edge Rewards` for the map, mapping trashcans onto edges.
    1) `Edge_list` is dict of `edgeID : 0/1/2`, describing edges into `must visit (MV)`, `visit if worth it (VIWI)`, and `no urgency (NU)` categories
    2) `Edge_rewards` is a dict of Reward values for visiting each edge in `VIWI` category


### **2. `PathPlanner`** (Core Module)
Working:
- Uses `Edge_list` and `Edge__rewards` for finding optimal route using optimized Dijkstra's.
- We initilize the `city_map` and since its fairly static, we leverage hashing for all route, hence we create a hash of all best possible route from node `u` to `v` in the map at minimum cost.
- During prediction, our algorithm uses preprocessed shortest paths instead of calculating it everytime, thus speeding up response time.
- We use a `start` position and create a cyclic route by adding all `MV` edges to route at minimum cost.
- For each `VIWI` edge, we calculate cost of taking detour from our shortest route to add that edge and compare against the reward we get for it, this comparison can be tuned by altering scalers defined in `config.yaml` file. Depending on that the system chooses to add the detour at minimum cost or ignore the `VIWI` edge.
- Finally we return a cyclic path such that all `MW` edges are present and `VIWI` edges added if the detour is worth it

### **3. `trashcan_model`** (Subcomponent of EdgeSelector)
Responsible for:
- Houses model methods for one single trashcan
- We use a LSTM Based neural network for prediction of trashcan values on basis of last `X` days (configurable by user)
- Automates data handling and stuff for each trashcan for itself
- Used internally by `EdgeSelector` during training and prediction.

---

## Supporting Modules

### **`logger`**
- Custom logging wrapper for consistent logs across the system.
- Logs debug, info, warning, and error messages.
- Used by all core modules and the FastAPI app.
- Log files are recorded and stored in logging directory
- Has a debug mode for Extended logging

### **`telemetry`** *(Optional / Future Use)*
- Takes complete care of anything related to database
- Handles telemetry like database logging, performance metrics, and analytics.
- Tracks uploads and model training stats.
- Currently commented out but structured for integration with database.



### **`preprocessor`**
- Supports in data validation and handles file updation for datasets.
- Helps in preprocessing maps and datasets for ease of use.

# **Happy Cleaning** 🐼🐼🎍🎍
