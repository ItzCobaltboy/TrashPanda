import networkx
import json
import yaml
import os
import pandas as pd
import numpy as np
import tensorflow
from app.preprocessor import GraphHandler, TrafficDataHandler, TrashcanDataHandler
from app.logger import logger
from trashcan_model import TrashcanModel


####################### Load Config #######################
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Load parameters
city_map_url = config["uploads"]["map_upload_dir"]
trashcan_data_url = config["uploads"]["trash_data_dir"]
traffic_data_url = config["uploads"]["traffic_data_dir"]

batch_size = config["edge_selector"]["training_batch_size"]
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
        self.TrashcanDataHandler = TrashcanDataHandler(self.trashcan_data_file)


        logger.log_info(f"Edge Selector initialized with city map: {self.city_map_file}, trashcan data: {self.trashcan_data_file}, traffic data: {self.traffic_data_file}")
        # Setup models array
        self.trashcan_models = {}

    def select_trashcans(self, latest_trashcan_data):
        selected_trashcans = {}

        """
        latest_trashcan_data is a dict with "trashcanID": value to append
        """

        for trashcan_id, value in latest_trashcan_data.items():
            if trashcan_id not in self.TrashcanDataHandler.trashcan_data["trashcanID"].values:
                logger.log_error(f"Trashcan ID {trashcan_id} not found in the existing data.")
                raise ValueError(f"Trashcan ID {trashcan_id} not found in the existing data.")
                continue
        # Append the latest data
        self.TrashcanDataHandler.append(latest_trashcan_data)


        # Copy all trashcan IDs to output
        for trashcan_id in self.TrashcanDataHandler.trashcan_data["trashcanID"].values:
            selected_trashcans[trashcan_id] = 0

        # Initialize all models for each trashcan
        self.initialize_trashcan_models()

        # if u have GPU, deploy multiple for training
        # predict the trashcan data


        # Select the trashcans that are not full but will get full in given days
        # Select Trashcans must be visited today


        # Return the selection
        """
        selected_trashcans is an dictionary with the following keys:
        "trashcan_id": [0,1,2]

        -- 0 means dont visit at all
        -- 1 means visit the trashcan, not full, visit only if worth it
        -- 2 means must visit, full trashcan
        """

        return selected_trashcans
    
    def initialize_trashcan_models(self):
        """
        Preprocess trashcan data and initialize a model for each trashcan.
        Also creates a mapping from trashcanID to edgeID.
        """
        df = self.TrashcanDataHandler.trashcan_data

        self.trashcan_to_edge = {}
        self.trashcan_models = {}

        for idx, row in df.iterrows():
            edge_id = row["edgeID"]
            trashcan_id = row["trashcanID"]
            self.trashcan_to_edge[trashcan_id] = edge_id

            # Initialize model for this trashcan (assuming your model takes a timeseries array)
            model = TrashcanModel(trashcan_id=trashcan_id, full_dataframe=df)   
            self.trashcan_models[trashcan_id] = model

            logger.log_debug(f"Initialized model for trashcan {trashcan_id} on edge {edge_id}.")
        
        logger.log_info(f"{len(self.trashcan_models)} trashcan models initialized successfully.")

    def train_models_parallel(self, models, batch_size):
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
            for model in models:
                model.train()
            return

        strategy = tensorflow.distribute.MirroredStrategy()
        logger.log_info(f"GPU detected. Using MirroredStrategy with {strategy.num_replicas_in_sync} replicas.")

        # Train models in batches of batch_size
        for i in range(0, len(models), batch_size):
            batch = models[i:i+batch_size]
            logger.log_info(f"Training batch of {len(batch)} models in parallel...")

            # Run training in the scope of distribution strategy
            with strategy.scope():
                # Start parallel training - here you can adapt depending on how model.train() is implemented
                # For example, if model.train() returns a tf.function, you can vectorize or map it
                # Otherwise, run them sequentially inside the scope but benefit from GPU parallelism
                for model in batch:
                    model.train()



    def select_edges(self, selected_trashcans):
        selected_edges = {}

        '''
        Returns a dictionary with all the edgeIDs marked as 0, 1, 2
        2 == must visit
        1 == visit if worth it
        0 == doesn't matter
        '''


        return selected_edges