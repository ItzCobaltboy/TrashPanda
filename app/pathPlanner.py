import os
import yaml
import pandas as pd
import numpy as np
import json
import networkx as nx
from logger import logger
from preprocessor import GraphHandler, TrashcanDataHandler, TrafficDataHandler

import random

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
    def __init__(self, city_map=nx.graph):
        
        # Record the data, inputed to pathPlanner
        self.graph = city_map
        self.nodes = list(city_map.nodes)
        self.master_graph_hash = {}
        logger.log_info(f"Currently using {city_map}")


        # preprocess hash
        self.preprocess_map()

    def preprocess_map(self):
        # Generate all possible combinations of short node paths in map and store then

        for source in self.nodes:
            lengths, paths = nx.single_source_dijkstra(self.graph, source, weight='weight')
            for target, path in paths.items():
                self.master_graph_hash[(source, target)] = {
                    "path": path,
                    "cost": lengths[target]
                }
                logger.log_debug(f'Shortest Path for "{source}" to "{target}" added, cost = "{lengths[target]}".')
        

        logger.log_info("Hash for shortest routes created.")
        # make a dict of all possible shortest path combos for each node combination

    def get_shortest_path(self, start, goal):
        """
        Return cached shortest path and its cost as a dict:
        {
            "path": [...],
            "cost": ...
        }
        """
        return self.master_graph_hash.get((start, goal), {"path": None, "cost": float('inf')})



# Create graph with 6 nodes and 10 edges with random weights between 1 and 10
G = nx.Graph()
nodes = list(range(6))
G.add_nodes_from(nodes)

edges = [
    (0, 1), (0, 2), (0, 3),
    (1, 2), (1, 4), (2, 3),
    (2, 5), (3, 4), (4, 5),
    (1, 5)
]

for u, v in edges:
    G.add_edge(u, v, weight=random.randint(1, 10))

# Instantiate PathPlanner
planner = PathPlanner(G)

# Sample queries to test retrieval of cached shortest paths
test_pairs = [(0, 4), (1, 5), (3, 0), (5, 2)]

for start, goal in test_pairs:
    info = planner.get_shortest_path(start, goal)
    print(f"Shortest path from {start} to {goal}: {info['path']} with cost {info['cost']}")