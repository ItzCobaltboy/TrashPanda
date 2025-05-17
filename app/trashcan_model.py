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
debug_mode = config["logging"]["debug"]
trashcan_col = config["trashcan_model"]["trashcanID_coloumn_name"]
edge_col = config["trashcan_model"]["edgeID_coloumn_name"]
############################################################

####################### Setup Logger #######################
logger = logger()
logger.user = "Trashcan_Model"
############################################################

import os
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

class TrashcanModel:
    def __init__(self, trashcan_ID, full_dataframe):
        self.trashcan_ID = trashcan_ID
        logger.user = f'TC_{self.trashcan_ID}'

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

        self.trashcan_data, self.edgeID = self._extract_trashcan_data(full_dataframe)

    def _extract_trashcan_data(self, df):
        try:
            filtered = df[df[trashcan_col] == self.trashcan_ID]
            if filtered.empty:
                raise ValueError("Trashcan ID not found in provided data.")

            edgeID = filtered[edge_col].values[0]
            trashcan_series = filtered.drop(columns=[trashcan_col, edge_col])
            logger.log_debug(f"Trashcan data extracted for {self.trashcan_ID} : EdgeID = {edgeID}")
            return trashcan_series, edgeID
        except Exception as e:
            logger.log_error(f"Error extracting trashcan data: {e}")
            return None, None

    def create_training_dataset(self):
        values = self.trashcan_data.values.flatten()
        values = self.scaler.fit_transform(values.reshape(-1, 1))

        X, y = [], []
        for i in range(len(values) - window_size):
            X.append(values[i:i + window_size])
            y.append(values[i + window_size])

        return np.array(X), np.array(y).reshape(-1, 1)

    def train(self):
        x_train, y_train = self.create_training_dataset()

        model = Sequential([
            LSTM(50, return_sequences=False, input_shape=(window_size, 1)),
            Dropout(0.2),
            Dense(25),
            Dropout(0.1),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(x_train, y_train, epochs=epochs, verbose=debug_mode)

        logger.log_info(f"Model trained successfully for {self.trashcan_ID}")
        self.model = model

    def predict(self):
        if self.model is None:
            raise ValueError("Model not trained yet.")

        values = self.trashcan_data.values.flatten()
        if len(values) < window_size:
            raise ValueError("Not enough data to make prediction.")

        scaled_seq = self.scaler.transform(values[-window_size:].reshape(-1, 1))
        input_seq = scaled_seq.reshape(1, window_size, 1)

        pred_scaled = self.model.predict(input_seq, verbose=debug_mode)[0][0]
        return self.scaler.inverse_transform([[pred_scaled]])[0][0]
    
    def update_with_new_data(self, updated_full_df):
        """
        Re-extracts trashcan data from updated DataFrame,
        and fine-tunes model using last (window_size+1) days.
        """
        try:
            self.trashcan_data, _ = self._extract_trashcan_data(updated_full_df)
            data = self.trashcan_data.values.flatten()
            if len(data) < window_size + 1:
                logger.log_info(f"Not enough data to update model yet for {self.trashcan_ID}")
                return

            data = self.scaler.transform(data.reshape(-1, 1))
            x_new = data[-(window_size + 1):-1]
            y_new = data[-1]

            self.model.fit(x_new.reshape(1, window_size, 1), np.array([[y_new]]), epochs=1, verbose=debug_mode)
            logger.log_debug(f"Model updated with new row data for {self.trashcan_ID}")
        except Exception as e:
            logger.log_error(f"Error during online update: {e}")
