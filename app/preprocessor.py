import json
import os
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from .logger import logger


logger = logger()
logger.user = "Preprocessor"

# open the config file and load the parameters
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
config = load_config()    
# Parameters Go here (RN none)
maps_dir = config["uploads"]["map_upload_dir"]
trash_data_dir = config["uploads"]["trash_data_dir"]
traffic_data_dir = config["uploads"]["traffic_data_dir"]


class GraphHandler:
    def __init__(self, city_map_file):
        self.Graph = nx.Graph()
        self.__city_map_file = city_map_file
        self.preprocess_city_map()

    def __load_city_map(self):
        # City map file path
        file_path = os.path.join(os.path.dirname(__file__), '..', 'uploads', maps_dir,self.__city_map_file)

        with open(file_path, 'r') as f:
            city_map = json.load(f)
            logger.log_info(f"City map loaded from {self.__city_map_file}.")
        return city_map

    def get_edge_by_ID(self, edgeID):
        for u, v, data in self.Graph.edges(data=True):
            if data['edgeID'] == edgeID:
                return u, v
        return None

    # Add Nodes and Edges to the Graph
    def preprocess_city_map(self):
        city_map = self.__load_city_map()
        # Add nodes
        for node in city_map['nodes']:
            self.Graph.add_node(node['id'])
            logger.log_debug(f"Node {node['id']} added to the graph.")
        # Add edges
        for edge in city_map['edges']:
            try:    
                self.Graph.add_edge(edge['source'], edge['target'], edgeID = edge['id'], weight=edge['weight'])
                logger.log_debug(f"Edge {edge['source']} -> {edge['target']} added to the graph with weight {edge['weight']}.")
            except Exception as e:
                logger.log_error(f"Error adding edge {edge['source']} -> {edge['target']}: {e}")
        logger.log_info("City map preprocessed and graph created successfully.")
    def get_all_edges_IDS(self):
        output = []

        for edge in self.Graph.edges(data=True):
            output.append(edge[2]['edgeID'])

        return output


class TrashcanDataHandler:
    def __init__(self, trashcan_data_file):
        self.trashcan_data_file = trashcan_data_file
        self.trashcan_data = pd.DataFrame()
        self.preprocess_trashcan_data()

    def __load_trashcan_data(self):
        # Trashcan data file path
        file_path = os.path.join(os.path.dirname(__file__), '..', 'uploads', trash_data_dir, self.trashcan_data_file)
        self.trashcan_data = pd.read_csv(file_path)
        logger.log_info(f"Trashcan data loaded from {self.trashcan_data_file}.")
        return self.trashcan_data
    
    def preprocess_trashcan_data(self):
        # Load trashcan data if not already loaded
        if self.trashcan_data.empty:
            self.__load_trashcan_data()

        logger.log_info("Starting trashcan data preprocessing...")

        # Get only the timestamp columns (exclude first two: edgeID, trashcanID)
        data_only = self.trashcan_data.iloc[:, 2:]

        # Iterate over rows
        for index, row in data_only.iterrows():
            values = row.values
            for i in range(len(values)):
                if pd.isna(values[i]):
                    left = values[i - 1] if i > 0 else None
                    right = values[i + 1] if i < len(values) - 1 else None

                    # Calculate average if both neighbors are valid
                    if left is not None and right is not None and not pd.isna(left) and not pd.isna(right):
                        if right - left < -10:
                            values[i] = (left+100) / 2
                        else:
                            values[i] = (left + right) / 2
                    elif left is not None and not pd.isna(left):
                        values[i] = left
                    elif right is not None and not pd.isna(right):
                        values[i] = right

                    # log the error handling
                    logger.log_debug(f"Missing value at index {index}, column {i} filled with {values[i]}.")
            # Update the row in DataFrame
            data_only.iloc[index] = values

        # Write the updated data back into the original DataFrame
        self.trashcan_data.iloc[:, 2:] = data_only

        # Save the cleaned data back to CSV
        file_path = os.path.join(os.path.dirname(__file__), '..', 'uploads', trash_data_dir, self.trashcan_data_file)
        self.trashcan_data.to_csv(file_path, index=False)
        logger.log_info("Trashcan data preprocessing complete. Missing values filled.")

        

    def append(self, new_data_dict, timestamp):
        """
        Appends a new column with the given timestamp, inserting the value
        for each trashcanID as specified in the dictionary.

        Parameters:
        - new_data_dict (dict): Dictionary of {trashcanID: fill_value}
        - timestamp (str): The timestamp label for the new column.
        """
        if self.trashcan_data.empty:
            self.__load_trashcan_data()

        # Initialize the new column with NaN
        self.trashcan_data[timestamp] = np.nan

        # Fill in the values using the dictionary
        for trashcan_id, value in new_data_dict.items():
            mask = self.trashcan_data["trashcanID"] == trashcan_id
            if not mask.any():
                logger.log_warning(f"TrashcanID {trashcan_id} not found in data. Skipping.")
                continue
            self.trashcan_data.loc[mask, timestamp] = value
            logger.log_debug(f"Appended value {value} for trashcanID {trashcan_id} at timestamp {timestamp}.")

        logger.log_info(f"New column for timestamp {timestamp} appended using trashcanID dictionary.")

        # Save the updated DataFrame back to the CSV
        file_path = os.path.join(os.path.dirname(__file__), '..', 'uploads', trash_data_dir, self.trashcan_data_file)
        self.trashcan_data.to_csv(file_path, index=False)
        logger.log_info(f"Updated trashcan data saved to {self.trashcan_data_file}.")


class TrafficDataHandler:
    def __init__(self, traffic_data_file):
        self.traffic_data_file = traffic_data_file
        self.traffic_data = pd.DataFrame()
        self.preprocess_traffic_data()


    def __load_traffic_data(self):
        # Traffic data file path
        file_path = os.path.join(os.path.dirname(__file__), '..', 'uploads', traffic_data_dir, self.traffic_data_file)
        self.traffic_data = pd.read_csv(file_path)
        logger.log_info(f"Traffic data loaded from {self.traffic_data_file}.")
        return self.traffic_data

    def preprocess_traffic_data(self):
        # Load traffic data if not already loaded
        if self.traffic_data.empty:
            self.__load_traffic_data()

        logger.log_info("Starting traffic data preprocessing...")

        # Get only the timestamp columns (exclude first column: edgeID)
        data_only = self.traffic_data.iloc[:, 1:]

        # Iterate over rows
        for index, row in data_only.iterrows():
            values = row.values
            for i in range(len(values)):
                if pd.isna(values[i]):
                    left = values[i - 1] if i > 0 else None
                    right = values[i + 1] if i < len(values) - 1 else None

                    # Fill missing values with average or neighbor
                    if left is not None and right is not None and not pd.isna(left) and not pd.isna(right):
                        values[i] = (left + right) / 2
                    elif left is not None and not pd.isna(left):
                        values[i] = left
                    elif right is not None and not pd.isna(right):
                        values[i] = right

                    logger.log_debug(f"Missing value at index {index}, column {i} filled with {values[i]}.")
            # Update row
            data_only.iloc[index] = values

        # Update the full DataFrame
        self.traffic_data.iloc[:, 1:] = data_only

        # Save back to CSV
        file_path = os.path.join(os.path.dirname(__file__), '..', 'uploads', traffic_data_dir, self.traffic_data_file)
        self.traffic_data.to_csv(file_path, index=False)
        logger.log_info("Traffic data preprocessing complete. Missing values filled.")
        

    def append(self, new_data_array, timestamp):
        """
        Appends a new column with the given timestamp and data values.

        Parameters:
        - new_data_array (list or Series): List of traffic values (must match number of rows).
        - timestamp (str): The timestamp label for the new column.
        """
        if self.traffic_data.empty:
            self.__load_traffic_data()

        if len(new_data_array) != len(self.traffic_data):
            logger.log_error("Length of new data does not match number of edges.")
            return

        self.traffic_data[timestamp] = new_data_array
        logger.log_info(f"New column for timestamp {timestamp} appended successfully.")

        # Save to CSV
        file_path = os.path.join(os.path.dirname(__file__), '..', 'uploads', traffic_data_dir, self.traffic_data_file)
        self.traffic_data.to_csv(file_path, index=False)
        logger.log_info(f"Updated traffic data saved to {self.traffic_data_file}.")



def validate_trashcan_data(trashcan_data_file = None, city_map_file = None):
    """
    Validates the trashcan data against the city map.

    Parameters:
    - trashcan_data_file (str): Path to the trashcan data CSV file.
    - city_map_file (str): Path to the city map JSON file.

    Returns:
    - bool: True if valid, False otherwise.
    """

    if trashcan_data_file is None or city_map_file is None:
        logger.log_error("Trashcan data file or city map file not provided.")
        return False

    city_map_file = os.path.join(os.path.dirname(__file__), '..', 'uploads', 'maps', city_map_file)
    trashcan_data_file = os.path.join(os.path.dirname(__file__), '..', 'uploads', 'trash_data', trashcan_data_file)

    logger.log_debug(f"City map file: {city_map_file}")
    logger.log_debug(f"Trashcan data file: {trashcan_data_file}")

     # Load the city map
    with open(city_map_file, 'r') as f:
        city_map = json.load(f)

    # Create a set of valid edge IDs from the city map
    valid_edge_ids = {edge['id'] for edge in city_map['edges']}  # optimized to use set

    # Load the trashcan data
    trashcan_data = pd.read_csv(trashcan_data_file)

    flag = True

    # Check if all edge IDs in the trashcan data are valid
    for edge_id in trashcan_data['edgeID']:
        if edge_id not in valid_edge_ids:
            logger.log_error(f"Invalid edge ID {edge_id} found in trashcan data.")
            flag = False

    logger.log_info("Trashcan data validation passed.")
    return flag

def validate_traffic_data(traffic_data_file = None, city_map_file = None):
    """
    Validates the traffic data against the city map.

    Parameters:
    - traffic_data_file (str): Path to the traffic data CSV file.
    - city_map_file (str): Path to the city map JSON file.

    Returns:
    - bool: True if valid, False otherwise.
    """

    if traffic_data_file is None or city_map_file is None:
        logger.log_error("Traffic data file or city map file not provided.")
        return False

    city_map_file = os.path.join(os.path.dirname(__file__), '..', 'uploads', 'maps', city_map_file)
    traffic_data_file = os.path.join(os.path.dirname(__file__), '..', 'uploads', 'traffic_data', traffic_data_file)

    logger.log_debug(f"City map file: {city_map_file}")
    logger.log_debug(f"Traffic data file: {traffic_data_file}")

     # Load the city map
    with open(city_map_file, 'r') as f:
        city_map = json.load(f)

    # Create a set of valid edge IDs from the city map
    valid_edge_ids = {edge['id'] for edge in city_map['edges']}  # optimized to use set

    # Load the traffic data
    traffic_data = pd.read_csv(traffic_data_file)

    flag = True

    # Check if all edge IDs in the traffic data are valid
    for edge_id in traffic_data['EdgeID']:
        if edge_id not in valid_edge_ids:
            logger.log_error(f"Invalid edge ID {edge_id} found in traffic data.")
            flag = False

    logger.log_info("Traffic data validation passed.")
    return flag

def validate_city_map(city_map_file = None):
    """
    City map is valid if  all edges are connected to valid nodes.
    """

    if city_map_file is None:
        logger.log_error("City map file not provided.")
        return False

    city_map_file = os.path.join(os.path.dirname(__file__), '..', 'uploads', 'maps', city_map_file)

    # Load the city map
    with open(city_map_file, 'r') as f:
        city_map = json.load(f)

    # Create a set of valid node IDs from the city map
    valid_node_ids = {node['id'] for node in city_map['nodes']}  # optimized to use set

    flag = True

    # Check if all edges are connected to valid nodes
    for edge in city_map['edges']:
        if edge['source'] not in valid_node_ids or edge['target'] not in valid_node_ids:
            logger.log_error(f"Invalid edge {edge['id']} found in city map.")
            flag = False

    logger.log_info("City map validation passed.")
    return flag