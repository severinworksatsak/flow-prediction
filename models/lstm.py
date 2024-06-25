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
from keras.layers import Input, Dense, Dropout, LSTM, Bidirectional
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

    def build_2layer_lstm(self, x_train, y_train, units_l1:int=200, activation_l1:str='tanh', dropout_l1:float=0.1,
                          units_l2:int=150, activation_l2:str='tanh', dropout_l2:float=0.1,
                          activation_dense:str='sigmoid', loss_f:str='mse', optimizer:str='Adam',
                          print_summary:bool=True):
        """
        Build a two-layer LSTM neural net.

        :param x_train: (array) 3D-shaped training features with dimensions batch_size, time_steps, seq_len
        :param y_train: (array) 2D-shaped label array with dimensions batch_size, time_steps
        :param units_l1: (int) Number of LSTM "neurons" in layer 1. Default is 200.
        :param activation_l1: (str) Activation function to use in layer 1. Default is 'tanh'.
        :param dropout_l1: (float) Dropout share within (0,1) for layer 1. Default is 0.2.
        :param units_l2: (int) Number of LSTM "neurons" in layer 2. Default is 150.
        :param activation_l2: (str) Activation function to use in layer 2. Default is 'tanh'.
        :param dropout_l2: (float) Dropout share within (0,1) for layer 2. Default is 0.1.
        :param activation_dense: (str) Activation function to use in layer 2. Default is 'sigmoid' to ensure scaling
                                 between [0,1].
        :param loss_f: (str) Loss function to-be-used for weight optimization. Default is 'mse'.
        :param optimizer: (str) Optimizer to use. Default is 'Adam'.
        :param print_summary: (bool) Boolean indicator whether to print model summary after executing build_2layer_lstm.

        :return: Parameterized keras 2l-LSTM model.
        """
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
        if print_summary:
            print(model.summary())

        return model