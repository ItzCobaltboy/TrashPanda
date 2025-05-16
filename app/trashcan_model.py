import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential, load_model
import os
import yaml

from logger import logger

######################## Load Config #######################
config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

trash_file_url = config["uploads"]["trash_data_dir"]
window_size = config["trashcan_model"]["training_window_size"]
epochs = config["trashcan_model"]["epochs"]
############################################################

####################### Setup Logger #######################
logger = logger()
logger.user = "Trashcan_Model"
############################################################

class TrashcanModel():
    def __init__(self, trashcan_ID, trashcan_data_file):
        # Set trashcan name
        self.trashcan_ID = trashcan_ID
        logger.user = f'TC_{self.trashcan_ID}'
        
        # retrieve its specific data and edgeID it is present on
        self.master_data_file = os.path.join(os.path.dirname(__file__), "..", "uploads", trash_file_url, trashcan_data_file)
        self.trashcan_data, self.edgeID = self.load_trashcan_data()
        
        # set the model
        self.model = None

    def load_trashcan_data(self):
        # Load the trashcan data
        try:
            master_data = pd.read_csv(self.master_data_file)
            trashcan_data = master_data[master_data['trashcan_ID'] == self.trashcan_ID]
            edgeID = trashcan_data['edgeID'].values[0]
            trashcan_data = trashcan_data.drop(columns=['trashcan_ID', 'edgeID'])

            logger.log_info(f"Trashcan data loaded successfully for {self.trashcan_ID} : EdgeID = {edgeID}")
            return trashcan_data, edgeID
        except Exception as e:
            logger.log_error(f"Error loading trashcan data: {e}")
            return None
        
    def preprocess_data(self):
        # Preprocess the data
        try:
            # Convert to numpy array
            data = self.trashcan_data.to_numpy()
            data = data[0]

            # Normalize the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

            # Your time sequence
            sequence = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

            # Create input-output pairs
            X, y = [], []
            for i in range(len(sequence) - window_size):
                X.append(sequence[i:i + window_size])  # window as input
                y.append(sequence[i + window_size])    # next value as output

            # Convert to numpy arrays
            X = np.array(X)
            y = np.array(y)

            # Reshape X for LSTM input: (samples, time_steps, features)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            logger.log_info(f"Data preprocessed successfully for {self.trashcan_ID}")
            return X, y, scaler
        except Exception as e:
            logger.log_error(f"Error preprocessing data: {e}")
            return None
        
    def train(self, input_shape, x_train, y_train):

        model = Sequential([
            LSTM(50, return_sequences= True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(x_train, y_train, epochs=epochs, verbose=0)
