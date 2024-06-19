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

class DeepLearner():

    def train_model(self, x_train, y_train, model, n_patience:int, n_epochs:int,
                    n_batch_size:int, val_share:float=0.2, verbose_int=0):
        # Split off validation data
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_share, shuffle=False)

        # Fit model incl early stopping & Learning rate
        stop_condition = EarlyStopping(monitor='val_loss', mode='min', verbose=0, # Change back to 0
                                       patience=n_patience, restore_best_weights=True)

        lrate_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=n_patience-1, min_lr=1e-7)

        training = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=n_epochs,
                             verbose=verbose_int, batch_size=n_batch_size, callbacks=[stop_condition, lrate_reduction])

        return model, training

    def predict_model(self, x_test, y_test, trained_model):
        # Prediction step
        y_pred = trained_model.predict(x_test)
        # TODO: Model evaluation
        # Evaluate model
        y_test
        return y_pred
