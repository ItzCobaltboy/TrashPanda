import threading
import networkx
import json
import yaml
import os
import pandas as pd
import numpy as np
import tensorflow
from .preprocessor import GraphHandler, TrafficDataHandler, TrashcanDataHandler
from .logger import logger
from .trashcan_model import TrashcanModel
import threading
import time

####################### Load Config #######################
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Load parameters
city_map_url = config["uploads"]["map_upload_dir"]
trashcan_data_url = config["uploads"]["trash_data_dir"]
traffic_data_url = config["uploads"]["traffic_data_dir"]
batch_size = config["edge_selector"]["training_batch_size"]
trashcan_threshold_mandatory = config["edge_selector"]["trashcan_threshold_mandatory"]
trashcan_threshold_optional = config["edge_selector"]["trashcan_threshold_optional"]
trashcan_col = config["trashcan_model"]["trashcanID_coloumn_name"]
edge_col = config["trashcan_model"]["edgeID_coloumn_name"]
##########################################################

####################### Setup Logger #######################
logger = logger()
logger.user = "EdgeSelector"
############################################################

time = 0.0

class EdgeSelector():
    def __init__(self, city_map_file, trashcan_data_file):

        # Setup Paths for respective files
        self.city_map_file = os.path.join(os.path.dirname(__file__),"..", "uploads", city_map_url, city_map_file)
        self.trashcan_data_file = os.path.join(os.path.dirname(__file__),"..", "uploads", trashcan_data_url, trashcan_data_file)
        
        # Setup the file handlers
        self.GraphHandler = GraphHandler(self.city_map_file)
        self.TrashcanDataHandler = TrashcanDataHandler(self.trashcan_data_file)
        logger.log_info(f"Edge Selector initialized with city map: {self.city_map_file}, trashcan data: {self.trashcan_data_file}")
        # Setup models array
        self.edgeList = self.GraphHandler.get_all_edges_IDS()
        self.trashcan_models = {}
        self.are_models_trained = False

    def get_time():
        return time
    
    def initialize_trashcan_models(self):
        """
        Preprocess trashcan data and initialize a model for each trashcan.
        Also creates a mapping from trashcanID to edgeID.
        """
        df = self.TrashcanDataHandler.trashcan_data

        self.trashcan_to_edge = {}
        self.trashcan_models = {}

        for idx, row in df.iterrows(): # Initializing models for each trashcan and storing them inthe trashcan_models dict
            edge_id = row[edge_col]
            trashcan_id = row[trashcan_col]
            self.trashcan_to_edge[trashcan_id] = edge_id

            # Initialize model for this trashcan (assuming your model takes a timeseries array)
            model = TrashcanModel(trashcan_ID=trashcan_id, full_dataframe=df)   
            self.trashcan_models[trashcan_id] = model

            logger.log_debug(f"Initialized model for trashcan {trashcan_id} on edge {edge_id}.")
        
        logger.log_info(f"{len(self.trashcan_models)} trashcan models initialized successfully.")

    def train_models_parallel(self):
        """
        Train multiple models in parallel using tf.distribute strategy if GPU is available.

        Parameters:
        - models (list): list of model objects, each with a .train() method.
        - batch_size (int): number of models to train in parallel (X from config).
        """

        gpus = tensorflow.config.list_physical_devices('GPU')
        gpu_available = len(gpus) > 0

        if not gpu_available:
            logger.log_info("GPU not available. Training models sequentially.")
            for model in self.trashcan_models.values():
                model.train()
            return
        
        for gpu in gpus:
            tensorflow.config.experimental.set_memory_growth(gpu, True)

        strategy = tensorflow.distribute.MirroredStrategy()
        logger.log_info(f"GPU detected. Using MirroredStrategy with {strategy.num_replicas_in_sync} replicas.")

        logger.log_info(f"Training models in parallel with batch size {batch_size}")
        models_list = list(self.trashcan_models.values())
        
        for i in range(0, len(models_list), batch_size):
            batch_models = models_list[i:i+batch_size]
            threads = []
            
            for model in batch_models:
                thread = threading.Thread(target=model.train)
                thread.start()
                threads.append(thread)
            
            # Wait for all threads in this batch to complete
            for thread in threads:
                thread.join()
                
        logger.log_info("Parallel training completed for all models")

    def train_models(self):
        # Initialize all models for each trashcan
        start = time.time()
        self.initialize_trashcan_models()
        # Train all models in parallel
        self.train_models_parallel()
        self.are_models_trained = True
        end = time.time()

        logger.log_debug(f"All models trained in {end - start} seconds.")
        return True


    def update_trashcan_data(self):
        # update the models to last data
        for trashcan_id, model in self.trashcan_models.items():
            model.update_with_new_data(self.TrashcanDataHandler.trashcan_data)

    def validate_latest_data(self, latest_trashcan_data = dict):
        """
        Validate the latest trashcan data against the existing data.
        This function checks if the trashcan IDs in the latest data exist in the existing data.

        Parameters:
        latest_trashcan_data (dict): Dictionary containing trashcan IDs and their values.

        Returns:
        bool: True if all trashcan IDs are valid, False otherwise.
        """
        for trashcan_id, value in latest_trashcan_data.items():
            if trashcan_id not in self.TrashcanDataHandler.trashcan_data[trashcan_col].values:
                logger.log_error(f"Trashcan ID {trashcan_id} not found in the existing data.")
                return False
        logger.log_debug("All trashcan IDs are present in latest data.")
        return True

    def select_trashcans(self, latest_trashcan_data = dict):
        selected_trashcans = {}

        """
        latest_trashcan_data is a dict with "trashcanID": value to append
        """

        if not self.validate_latest_data(latest_trashcan_data):
            logger.log_error("Invalid trashcan data. Please check the IDs.")
            return None, None

        # Copy all trashcan IDs to output
        for trashcan_id in self.TrashcanDataHandler.trashcan_data[trashcan_col].values:
            selected_trashcans[trashcan_id] = 0

        # if u have GPU, deploy multiple for training
        # self.train_models_parallel(self.trashcan_models, batch_size)
        if self.are_models_trained == False:
            # Initialize all models for each trashcan
            start = time.time()
            self.initialize_trashcan_models()
            # Train all models in parallel
            self.train_models_parallel()
            self.are_models_trained = True
            end = time.time()

            global time 
            time = end - start
            logger.log_debug(f"All models trained in {end - start} seconds.")

        # Append the latest data
        self.TrashcanDataHandler.append(latest_trashcan_data, timestamp="DAY")

        # predict the trashcan data
        predicted_trash_values = {}
        for trashcan_id, model in self.trashcan_models.items():
            predicted_trash_values[trashcan_id] = model.predict()

        # update all the models
        self.update_trashcan_data()
        
        # Return the selection
        """
        selected_trashcans is an dictionary with the following keys:
        "trashcan_id": [0,1,2]

        -- 0 means dont visit at all
        -- 1 means visit the trashcan, not full, visit only if worth it
        -- 2 means must visit, full trashcan
        """

        for trashcan_id, value in latest_trashcan_data.items():
            if value > trashcan_threshold_mandatory:
                selected_trashcans[trashcan_id] = 2
            elif predicted_trash_values[trashcan_id] > trashcan_threshold_optional:
                selected_trashcans[trashcan_id] = 1
            else:
                selected_trashcans[trashcan_id] = 0


        return selected_trashcans, predicted_trash_values

    def select_edges(self, trashcan_actions: dict, trashcan_predictions: dict) -> tuple:
        """
        Given a dict of trashcan actions and predictions, return:
        1. A dict of edge actions: { edgeID: action_code } where action code is the max of all trashcans on that edge.
        2. A dict of edge rewards: { edgeID: reward } which is the sum of predicted values of trashcans on that edge,
        but only if the edge action is 1.

        Parameters:
        trashcan_actions (dict): { "trashcanID": action_code }
        trashcan_predictions (dict): { "trashcanID": predicted_value }

        Returns:
        tuple: (edge_actions: dict, edge_rewards: dict)
        """
        edge_actions = {edge_id: 0 for edge_id in self.edgeList}
        edge_rewards = {edge_id: 0 for edge_id in self.edgeList}

        # Step 1: Compute max action code per edge
        for trashcan_id, action_code in trashcan_actions.items():
            edge_id = self.trashcan_to_edge.get(trashcan_id)
            if edge_id is None:
                logger.log_warning(f"No edge found for trashcan ID: {trashcan_id}")
                continue
            edge_actions[edge_id] = max(edge_actions[edge_id], action_code)

        # Step 2: Compute rewards only for edges with action_code == 1
        for trashcan_id, predicted_value in trashcan_predictions.items():
            edge_id = self.trashcan_to_edge.get(trashcan_id)
            if edge_id is None:
                logger.log_warning(f"No edge found for trashcan ID: {trashcan_id}")
                continue
            if edge_actions[edge_id] == 1:
                edge_rewards[edge_id] += predicted_value

        return edge_actions, edge_rewards

