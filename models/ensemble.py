import pandas as pd
import numpy as np

from models.utility import load_input, scale_with_minmax, get_dates_from_config, handle_outliers, inverse_transform_minmax, split_dataframe
from models.deeplearner import DeepLearner

from solutil import evaluations as ev
from solutil import feature_selection as fs

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, ParameterGrid


class Ensemble():

    def get_ensemble_input(self):
        pass


    def define_ensemble_model(self):
        pass


    def train_model(self, model, x_train, y_train):
        # Can be done with deeplearner > match case logic
        pass



