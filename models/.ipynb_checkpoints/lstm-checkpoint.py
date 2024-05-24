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
        n_timestep = x.shape[1]
        n_features = x.shape[2]
        n_ahead = y.shape[1]

        # Build Model
        model = Sequential()
        model.add(Input(shape=(n_timestep, n_features), dtype='float64'))
        model.add(LSTM(units=units_l1, activation=activation_l1))
        model.add(Dropout(dropout_l1))
        model.add(LSTM(units=units_l2, activation=activation_l2))
        model.add(Dropout(dropout_l2))
        model.add(Dense(units=n_ahead, activation=activation_dense))
        model.compile(loss=loss_f, optimizer=optimizer)
        print(model.summary())

        return model

    def train_model(self, x_train, y_train, model, n_patience:int, n_epochs:int, n_batch_size:int, val_share:float=0.2):
        # Split off validation data
        x_val = train_test_split(x_train, test_size=val_share, shuffle=False)
        y_val = train_test_split(y_train, test_size=val_share, shuffle=False)

        # Fit model incl early stopping & Learning rate
        stop_condition = EarlyStopping(monitor='val_loss', mode='min', verbose=0,
                                       patience=n_patience, restore_best_weights=True)

        lrate_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=n_patience-1, min_lr=1e-7)

        training = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=n_epochs,
                             verbose=0, batch_size=n_batch_size, callbacks=[stop_condition, lrate_reduction])

        return training

    def predict_model(self, x_test, y_test, trained_model):
        # Prediction step
        y_pred = trained_model.predict(x_test)

        return y_pred
