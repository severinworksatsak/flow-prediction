# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:12:50 2024

@author: LES
"""

from models.utility import load_input, get_dates_from_config, handle_outliers, get_params_from_config
from models.master_prediction import prepare_inputs, prepare_inputs_lstm, train_lstm, predict_lstm, predict_lstm_daywise


import configparser
from datetime import datetime, timedelta
import pandas as pd

import solutil.dbqueries as db
from datetime import datetime, timedelta

## Variable Assignment
str_model = 'Inlet1'

date_from = datetime.strptime('01.07.2024', '%d.%m.%Y')
date_to = datetime.now() + timedelta(days=8)
env_vars = db.get_env_variables(mandant='EPAG_ENERGIE')
n_timestep = 6

ts_id_1d = 11055778
ts_daily = db.get_timeseries_1d(ts_id_1d, date_from, date_to, **env_vars, str_table='meanvalues')

# Resampler tryout
ts_resampled = ts_daily.resample(timedelta(hours=24 // n_timestep)).ffill()

ts_resampled.iloc[:-1]
daterange = pd.date_range(start=date_from, end=date_to, freq=f'{24//n_timestep}h', tz='Etc/GMT-1')
ts_shift = ts_resampled.shift(8)

## Combination
data = load_input(str_model='inlet1_lstm', date_from=date_from, date_to=date_to)

outliers = handle_outliers(data)

dates = get_dates_from_config(str_model='inlet1_lstm', training=False)
get_params_from_config(function='get_label', str_model='inlet1_lstm')

inputs = prepare_inputs(str_model='inlet1_lstm', idx_train=False)
inputs_lstm = prepare_inputs_lstm(str_model='inlet1_lstm', idx_train=False)
train_lstm(str_model='inlet1_lstm')
pred_lstm = predict_lstm(str_model='inlet1_lstm')

pred_lstm_daywise = predict_lstm_daywise(str_model='inlet1_lstm', idx_train=False)



# df update tryout ############################################################

df_test = df_scaled.copy()

df_pred = pred_lstm.to_frame(name='ypred')
df_pred['base_1d_lag1'] = [pred_lstm.mean() for i in range(len(pred_lstm))]
series_pred = pd.Series(df_pred['base_1d_lag1'])


df_test.update(series_pred)













