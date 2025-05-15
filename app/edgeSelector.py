import networkx
import json
import yaml
import os
import pandas as pd
import numpy as np
import tensorflow
from app.preprocessor import GraphHandler, TrafficDataHandler, TrashcanDataHandler
from app.logger import logger


####################### Load Config #######################
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Load parameters
city_map_url = config["uploads"]["map_upload_dir"]
trashcan_data_url = config["uploads"]["trash_data_dir"]
traffic_data_url = config["uploads"]["traffic_data_dir"]
##########################################################

####################### Setup Logger #######################
logger = logger()
logger.user = "EdgeSelector"
############################################################

class edgeSelector():
    def __init__(self, city_map_file, trashcan_data_file, traffic_data_file):

        # Setup Paths for respective files
        self.city_map_file = os.path.join(os.path.dirname(__file__),"..", "uploads", city_map_url, city_map_file)
        self.trashcan_data_file = os.path.join(os.path.dirname(__file__),"..", "uploads", trashcan_data_url, trashcan_data_file)
        self.traffic_data_file = os.path.join(os.path.dirname(__file__),"..", "uploads", traffic_data_url, traffic_data_file)
        
        # Setup the file handlers
        self.GraphHandler = GraphHandler(self.city_map_file)
        self.TrafficDataHandler = TrafficDataHandler(self.traffic_data_file)
        self.TrashcanDataHandler = TrashcanDataHandler(self.trashcan_data_file)


        logger.log_info(f"Edge Selector initialized with city map: {self.city_map_file}, trashcan data: {self.trashcan_data_file}, traffic data: {self.traffic_data_file}")
        # Setup models array
        self.trashcan_models = []
        self.traffic_models = []

    def select_trashcans(self, latest_trashcan_data):
        selected_trashcans = []

        # Append the latest data


        # Select the must visit trashcans


        # Initialize all models for each trashcan


        # predict the trashcan data


        # Select the trashcans that are not full but will get full in given days

        # Return the selection
        """
        selected_trashcans is an dictionary with the following keys:
        "trashcan_id": [0,1,2]

        -- 0 means dont visit at all
        -- 1 means visit the trashcan, not full, visit only if worth it
        -- 2 means must visit, full trashcan
        """

        return selected_trashcans
    
    def predict_traffic(self, latest_traffic_data):
        # Predict the traffic data using the models
        # Return the predicted traffic data as a dictionary

        return latest_traffic_data
    
    def get_path(self, start, end):
        
        path = []


        #  Retrieve the to be vsited trashcans
        selected_trashcans = self.select_trashcans()
        #  Retrieve the predicted traffic data
        traffic_forecast = self.predict_traffic()

        # Call the path planner class


        """
        Path is defined by an array of node names to follow, the system returns a cyclic path always
        """
        return path