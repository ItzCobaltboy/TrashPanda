import os
import yaml
import pandas as pd
import numpy as np
import json
import networkx as nx
from app.logger import logger
from app.preprocessor import GraphHandler, TrashcanDataHandler, TrafficDataHandler


# import config
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# configs go here (rn NONE)


# setup logger
logger = logger()
logger.user = "PathPlanner"
logger.log_info("PathPlanner initialized.")


class PathPlanner:
    def __init__(self, city_map_file=None, trashcan_data_file=None, traffic_data_file=None):
        self.city_map_file = city_map_file
        self.trashcan_data_file = trashcan_data_file
        self.traffic_data_file = traffic_data_file
        self.Graph = GraphHandler(self.city_map_file)
        self.trashcan_data = TrashcanDataHandler(self.trashcan_data_file)
        self.traffic_data = TrafficDataHandler(self.traffic_data_file)
        logger.log_info(f"Currently using {self.city_map_file}, {self.trashcan_data_file}, {self.traffic_data_file} for path planning.")

    