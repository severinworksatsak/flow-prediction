import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from math import sin, cos, pi
from calendar import monthrange
import pickle
import configparser
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Dropout, LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from solutil import dbqueries as db
from models.utility import load_input, scale_with_minmax, generate_sequences

# Build LSTM class
class simpleLSTM():

    def __init__(self):
        # Load config from directory
        with open(Path("config/config.json"), "r") as jsonfile:
            self.config = json.load(jsonfile)

    def build_2layer_lstm(self, x_train, y_train, units_l1:int=200, activation_l1:str='sigmoid', dropout_l1:float=0.1,
                          units_l2:int=150, activation_l2:str='sigmoid', dropout_l2:float=0.1,
                          activation_dense:str='sigmoid', loss_f:str='mse', optimizer:str='Adam'):

        # Get dimensions
        n_timestep = x_train.shape[1]
        n_features = x_train.shape[2]
        n_ahead = y_train.shape[1]

        # Build Model
        model = Sequential()
        model.add(Input(shape=(n_timestep, n_features), dtype='float64'))
        model.add(LSTM(units=units_l1, activation=activation_l1, return_sequences=True))
        model.add(Dropout(dropout_l1))
        model.add(LSTM(units=units_l2, activation=activation_l2))
        model.add(Dropout(dropout_l2))
        model.add(Dense(units=n_ahead, activation=activation_dense))
        model.compile(loss=loss_f, optimizer=optimizer)
        print(model.summary())

        return model