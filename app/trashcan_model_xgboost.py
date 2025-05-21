import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import os
import yaml

from .logger import logger

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

        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        self.trashcan_data, self.edgeID = self._extract_trashcan_data(full_dataframe)
        self.timestamps = pd.to_datetime(self.trashcan_data.columns)

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

    def _build_feature_frame(self):
        series = self.trashcan_data.values.flatten()
        df = pd.DataFrame({
            'timestamp': self.timestamps,
            'fill_level': series
        }).sort_values('timestamp').reset_index(drop=True)

        df['daily_fill_rate'] = df['fill_level'].diff()
        df['avg_fill_rate_3d'] = df['daily_fill_rate'].rolling(3).mean()
        df['days_to_full'] = (100 - df['fill_level']) / df['avg_fill_rate_3d']
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['month'] = df['timestamp'].dt.month
        df['future_fill'] = df['fill_level'].shift(-2)
        df['target'] = (df['future_fill'] >= 80).astype(int)

        return df.dropna().reset_index(drop=True)

    def create_training_dataset(self):
        df = self._build_feature_frame()

        features = ['fill_level', 'daily_fill_rate', 'avg_fill_rate_3d',
                    'days_to_full', 'day_of_week', 'is_weekend', 'month']
        X = df[features]
        y = df['target']
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, y.values

    def train(self):
        try:
            X, y = self.create_training_dataset()
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
            model.fit(X, y)
            self.model = model
            logger.log_info(f"Model trained successfully for {self.trashcan_ID}")
        except Exception as e:
            logger.log_error(f"Training failed for {self.trashcan_ID}: {e}")

    def predict(self):
        if self.model is None:
            raise ValueError("Model not trained yet.")

        try:
            df = self._build_feature_frame()
            latest_row = df.iloc[[-1]]

            features = ['fill_level', 'daily_fill_rate', 'avg_fill_rate_3d',
                        'days_to_full', 'day_of_week', 'is_weekend', 'month']
            X = self.scaler.transform(latest_row[features])
            pred = self.model.predict(X)[0]
            return bool(pred)
        except Exception as e:
            logger.log_error(f"Prediction error for {self.trashcan_ID}: {e}")
            return None

    def update_with_new_data(self, updated_full_df):
        try:
            self.trashcan_data, _ = self._extract_trashcan_data(updated_full_df)
            df = self._build_feature_frame()

            if len(df) < window_size + 1:
                logger.log_info(f"Not enough data to update model yet for {self.trashcan_ID}")
                return

            features = ['fill_level', 'daily_fill_rate', 'avg_fill_rate_3d',
                        'days_to_full', 'day_of_week', 'is_weekend', 'month']

            # Get last window_size rows for features
            X_window = df.iloc[-(window_size + 1):-1][features].values.flatten().reshape(1, -1)
            y_target = df.iloc[-1]['target']

            X_scaled = self.scaler.transform(X_window)
            y = np.array([y_target])

            self.model.fit(X_scaled, y)
            logger.log_debug(f"Model updated with new row data for {self.trashcan_ID}")

        except Exception as e:
            logger.log_error(f"Error during online update: {e}")
