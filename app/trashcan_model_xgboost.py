from xgboost import XGBClassifier
import pandas as pd
import numpy as np

class TrashcanModel:
    def __init__(self, trashcan_ID, full_dataframe, history_days=3, predict_threshold=80):
        self.trashcan_ID = trashcan_ID
        self.history_days = history_days
        self.predict_threshold = predict_threshold
        self.df = full_dataframe
        self.model = None
        self.X_latest = None  # Store the last known feature row for prediction

    def _prepare_data(self):
        """
        Prepare training data by transforming wide-format rows into rolling windows.
        """
        row = self.df[self.df['trashcanId'] == self.trashcan_ID].iloc[0]
        fill_values = row[2:].values.astype(float)  # Skip trashcanId, edgeId

        data = []
        labels = []

        for i in range(len(fill_values) - self.history_days - 1):
            X = fill_values[i:i + self.history_days]
            y_val = fill_values[i + self.history_days + 1]  # Predict 2 days ahead
            y = int(y_val >= self.predict_threshold)
            data.append(X)
            labels.append(y)

        if len(data) == 0:
            return None, None

        return np.array(data), np.array(labels)

    def train(self):
        X, y = self._prepare_data()
        if X is None:
            return
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.model.fit(X, y)

        # Store the last row for prediction
        self.X_latest = self.df[self.df['trashcanId'] == self.trashcan_ID].iloc[0][-(self.history_days):].values.astype(float)

    def predict(self):
        if self.model is None or self.X_latest is None:
            return 0.0  # Default to not full
        X = np.array(self.X_latest).reshape(1, -1)
        prob = self.model.predict_proba(X)[0][1]
        return prob * 100  # Return as percentage fill probability

    def update_with_new_data(self, updated_df):
        self.df = updated_df
        # Update the latest input
        self.X_latest = self.df[self.df['trashcanId'] == self.trashcan_ID].iloc[0][-(self.history_days):].values.astype(float)
