import os
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import yaml
import time
from .logger import logger
from .preprocessor import GraphHandler, TrashcanDataHandler

# Load config
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Config parameters
city_map_url = config["uploads"]["map_upload_dir"]
trashcan_data_url = config["uploads"]["trash_data_dir"]
traffic_data_url = config["uploads"]["traffic_data_dir"]
batch_size = config["edge_selector"]["training_batch_size"]
trashcan_threshold_mandatory = config["edge_selector"]["trashcan_threshold_mandatory"]
trashcan_threshold_optional = config["edge_selector"]["trashcan_threshold_optional"]
trashcan_col = config["trashcan_model"]["trashcanID_coloumn_name"]
edge_col = config["trashcan_model"]["edgeID_coloumn_name"]
#############################

logger = logger()
logger.user = "EdgeSelector"

class EdgeSelector:
    def __init__(self, city_map_file, trashcan_data_file):
        self.trashcan_data_file = os.path.join(os.path.dirname(__file__), "..", "uploads", trashcan_data_url, trashcan_data_file)
        self.trashcan_data = pd.read_csv(self.trashcan_data_file)
        self.edgeList = self.trashcan_data[edge_col].unique().tolist()
        self.models = {}
        self.are_models_trained = False
        self.history_days = 3  # Number of days used for training

        self.GraphHandler = GraphHandler(city_map_file)

        logger.log_info(f"EdgeSelector initialized with trashcan data: {self.trashcan_data_file}")

    def initialize_models(self):
        for idx, row in self.trashcan_data.iterrows():
            trashcan_id = row[trashcan_col]
            fill_values = row.iloc[2:].values.astype(float)

            data = []
            labels = []

            for i in range(len(fill_values) - self.history_days - 1):
                X = fill_values[i:i+self.history_days]
                y_val = fill_values[i+self.history_days+1]
                y = 2 if y_val >= trashcan_threshold_mandatory else (1 if y_val >= trashcan_threshold_optional else 0)
                data.append(X)
                labels.append(y)

            if len(data) < 5:
                continue

            model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
            model.fit(np.array(data), np.array(labels))
            self.models[trashcan_id] = {
                "model": model,
                "latest_input": fill_values[-self.history_days:]  # Store last N days for prediction
            }
            logger.log_debug(f"Trained model for trashcan {trashcan_id}")

        self.are_models_trained = True
        logger.log_info(f"Trained {len(self.models)} models successfully.")

    def train_models(self):
        start = time.time()
        self.initialize_models()
        end = time.time()
        logger.log_debug(f"Training completed in {end - start:.2f} seconds.")
        return end - start

    def update_trashcan_data(self):
        self.trashcan_data = pd.read_csv(self.trashcan_data_file)

    def validate_latest_data(self, latest_trashcan_data=dict):
        existing_ids = self.trashcan_data[trashcan_col].values
        for tid in latest_trashcan_data:
            if tid not in existing_ids:
                logger.log_error(f"Trashcan ID {tid} not found.")
                return False
        return True

    def select_trashcans(self, latest_trashcan_data=dict, day_name=str):
        """
        Input: latest_trashcan_data = { trashcanId: fill_level }, e.g. { "TC1": 87 }
        Output: (trashcan_actions, predicted_values)
        """
        if not self.validate_latest_data(latest_trashcan_data):
            return None, None

        if not self.are_models_trained:
            self.train_models()

        trashcan_actions = {}
        predicted_values = {}

        for tid, data in self.models.items():
            if tid not in latest_trashcan_data:
                continue

            latest_fill = latest_trashcan_data[tid]
            latest_input = np.append(data["latest_input"][1:], latest_fill).astype(float)
            model = data["model"]
            pred_class = model.predict(latest_input.reshape(1, -1))[0]
            prob = model.predict_proba(latest_input.reshape(1, -1))[0][pred_class]

            trashcan_actions[tid] = int(pred_class)
            predicted_values[tid] = prob * 100  # confidence as %
            # Update stored input
            self.models[tid]["latest_input"] = latest_input

        return trashcan_actions, predicted_values

    def select_edges(self, trashcan_actions: dict, trashcan_predictions: dict) -> tuple:
        edge_actions = {edge_id: 0 for edge_id in self.edgeList}
        edge_rewards = {edge_id: 0 for edge_id in self.edgeList}

        # Map trashcans to edges
        edge_map = dict(zip(self.trashcan_data["trashcanId"], self.trashcan_data["edgeId"]))

        for tid, action in trashcan_actions.items():
            edge_id = edge_map.get(tid)
            if edge_id:
                edge_actions[edge_id] = max(edge_actions[edge_id], action)

        for tid, prob in trashcan_predictions.items():
            edge_id = edge_map.get(tid)
            if edge_id and edge_actions[edge_id] == 1:
                edge_rewards[edge_id] += prob

        return edge_actions, edge_rewards
