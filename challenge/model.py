import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from typing import Tuple, Union, List

class DelayModel:

    def __init__(
        self, random_state: int = 1, learning_rate: float = 0.01
    ):
        self._model = xgb.XGBClassifier(random_state=random_state, 
                                        learning_rate=learning_rate,
                                        scale_pos_weight = 4.4402380952380955) # Model should be saved in this attribute.

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = 'delay',
        train: bool = True
        
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        
        features = pd.concat([
        pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
        pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
        pd.get_dummies(data['MES'], prefix = 'MES')],
        axis = 1
        )
        top_10_features = [
                            "OPERA_Latin American Wings", 
                            "MES_7",
                            "MES_10",
                            "OPERA_Grupo LATAM",
                            "MES_12",
                            "TIPOVUELO_I",
                            "MES_4",
                            "MES_11",
                            "OPERA_Sky Airline",
                            "OPERA_Copa Air"
                        ]

        if train:
            features = features[top_10_features]
            data['min_diff'] = data.apply(self.get_min_diff, axis = 1)
            threshold_in_minutes = 15
            data['delay'] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
            
            target = data[target_column]
            return features, pd.DataFrame(target)

        features_pred = pd.DataFrame(columns=top_10_features)
        features_pred.loc[0] = [0] * len(top_10_features)
        for col in features.columns:
            if col in top_10_features:
                features_pred[col] = features[col]

        return features_pred

        

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        self._model.fit(features, target)
        return

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        prediction = self._model.predict(features)
        prediction = prediction.tolist()
        return prediction
    

    @staticmethod
    def get_min_diff(data):
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds())/60
        return min_diff
