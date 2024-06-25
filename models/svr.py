import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from math import sin, cos, pi, log10
from calendar import monthrange
import pickle
import configparser
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Dropout, LSTM, Bidirectional
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from solutil import dbqueries as db
from models.utility import load_input, scale_with_minmax, generate_sequences, get_params_from_config

# Build SVR class
class SVReg():

    def __init__(self):
        # Load config from directory
        with open(Path("config/config.json"), "r") as jsonfile:
            self.config = json.load(jsonfile)

    # Arrange model input
    @staticmethod
    def build_model_input(df, target_var:str, str_model:str, n_offset:int=None, n_timestep:int=None):
        """
        Create new target variables for each timestep-SVR model. As SVR output is single-length, a full-day prediction
        requires the training of n_timestep models and thus n_timestep different target variables.
        :param df: Feature dataframe containing the target variable.
        :param target_var: Name of the target variable passed as string - used for column referencing.
        :param model_str: Name of model for which parameter n_timestep should be loaded from config.
        :param n_offset: Number of timesteps between training and first prediction.
        :param n_timestep: Number of intra-day timesteps within a day.
        :return df: Dataframe with target variables for n_timestep models.
        :return name_list: List with all label names.
        """
        # Retrieve config values
        if n_timestep is None:
            n_timestep = get_params_from_config('model_train', str_model)['n_timestep']

        if n_offset is None:
            n_offset = n_timestep

        # Create target variables for every model
        name_list = []

        for timestep in range(n_timestep):
            # Individual model parameters
            model_num = timestep + 1
            model_name = f"model{model_num}"
            lag = -(timestep + n_offset)
            y_name = f"y_{model_name}"
            name_list.append(y_name)

            # Lag target variable & remove starting NAs
            df[y_name] = df[target_var].shift(lag)
            df[y_name] = df[y_name].bfill()

        # Drop original label & remove trailing NAs
        df_save = df.copy().drop(columns=[target_var])
        df_save.dropna(inplace=True)

        return df_save, name_list


    def build_svr(self, label_names:list=None, model_dict:list=None):
        pass