import pandas as pd
from datetime import datetime, timedelta
from math import sin, cos, pi
from calendar import monthrange
from pytz import timezone
import pickle
import configparser
import os
from keras.layers import Input, Dense, SimpleRNN
from keras.models import Model, save_model, load_model
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

from tasks.Funktionen.models_keras.main import back_to_ts, rmae_func
from tasks.Funktionen.BelVis.get_timeseries import get_timeseries_15min, get_timeseries_1h, get_timeseries_1d
from tasks.Funktionen.Filesystem.DWH import write_DWH


def load_input(str_model, date_from, date_to, n_timestep, idx_train):

    # Config Parser
    config = configparser.ConfigParser()
    # config.read('tasks//Input//models_keras//model_config.ini')
    config.read('config//model_config.ini')

    # ---------------
    # lade Zeitreihen
    # ---------------
    # Tagessumme Abfluss (1d)
    ts_base_1d_id = config.get(str_model, 'base_1d')
    ts_base_1d = get_timeseries_1d('EPAG_ENERGIE', ts_base_1d_id, date_from - timedelta(days=1),
                                   date_to, 'meanvalues')

    # Tag verwenden (1d)
    ts_useday_1d_id = config.get(str_model, 'useday_1d')
    ts_useday_1d = get_timeseries_1d('EPAG_ENERGIE', ts_useday_1d_id, date_from,
                                     date_to + timedelta(days=1), 'meanvalues')

    # Temperatur (1h)
    ts_temp_1h_id = config.get(str_model, 'temp_1h')
    ts_temp_1h = get_timeseries_1h('EPAG_ENERGIE', ts_temp_1h_id, date_from - timedelta(days=1), date_to)

    # Globalstrahlung (1h)
    ts_glob_1h_id = config.get(str_model, 'glob_1h')
    ts_glob_1h = get_timeseries_1h('EPAG_ENERGIE', ts_glob_1h_id, date_from - timedelta(days=1), date_to)

    # Niederschlag (1h)
    ts_rain_1h_id = config.get(str_model, 'rain_1h')
    ts_rain_1h = get_timeseries_1h('EPAG_ENERGIE', ts_rain_1h_id, date_from - timedelta(days=1), date_to)

    # Bodenfeuchtigkeit (1h)
    ts_bf15_1h_id = config.get(str_model, 'bf15_1h')
    ts_bf15_1h = get_timeseries_1h('EPAG_ENERGIE', ts_bf15_1h_id, date_from - timedelta(days=1), date_to)

    # Schneeschmelzmenge (1h)
    ts_schmelz_1h_id = config.get(str_model, 'schmelz_1h')
    ts_schmelz_1h = get_timeseries_1h('EPAG_ENERGIE', ts_schmelz_1h_id, date_from - timedelta(days=1), date_to)

    # --------------------
    # setze Input zusammen
    # --------------------
    # Tag des Jahres Sinus
    input_1_s = ts_useday_1d.resample(timedelta(hours=24 // n_timestep)).pad()
    input_1_s = input_1_s.iloc[:-1, :]
    input_1_s['value'] = [sin(element.dayofyear / (monthrange(element.year, 2)[1] + 337) * 2 * pi) for element in
                          input_1_s.index]
    input_1_s.columns = ['yearday_sin']

    # Tag des Jahres Cosinus
    input_1_c = ts_useday_1d.resample(timedelta(hours=24 // n_timestep)).pad()
    input_1_c = input_1_c.iloc[:-1, :]
    input_1_c['value'] = [cos(element.dayofyear / (monthrange(element.year, 2)[1] + 337) * 2 * pi) for element in
                          input_1_c.index]
    input_1_c.columns = ['yearday_cos']

    # Tag für Training verwenden
    input_3 = ts_useday_1d.resample(timedelta(hours=24 // n_timestep)).pad()
    input_3 = input_3.iloc[:-1, :]
    input_3.columns = ['useday']

    # Abfluss Vortag
    input_4 = ts_base_1d.resample(timedelta(hours=24 // n_timestep)).pad()
    input_4 = input_4.iloc[:-1, :]
    input_4.columns = ['base_day_-1']
    input_4.index = input_1_s.index
    input_4 = input_4.fillna(0)

    # Temperatur -1
    input_5 = ts_temp_1h.resample(timedelta(hours=24 // n_timestep)).mean()
    input_5 = input_5.iloc[n_timestep - 1:-1, :]
    input_5.index = input_1_s.index
    input_5.columns = ['temp-1']

    # Globalstrahlung -1
    input_6 = ts_glob_1h.resample(timedelta(hours=24 // n_timestep)).mean()
    input_6 = input_6.iloc[n_timestep - 1:-1, :]
    input_6.index = input_1_s.index
    input_6.columns = ['glob-1']

    # Globalstrahlung -2
    input_6a = ts_glob_1h.resample(timedelta(hours=24 // n_timestep)).mean()
    input_6a = input_6a.iloc[n_timestep - 2:-2, :]
    input_6a.index = input_1_s.index
    input_6a.columns = ['glob-2']

    # Globalstrahlung -3
    input_6b = ts_glob_1h.resample(timedelta(hours=24 // n_timestep)).mean()
    input_6b = input_6b.iloc[n_timestep - 3:-3, :]
    input_6b.index = input_1_s.index
    input_6b.columns = ['glob-3']

    # Niederschlag -1
    input_7 = ts_rain_1h.resample(timedelta(hours=24 // n_timestep)).mean()
    input_7 = input_7.iloc[n_timestep - 1:-1, :]
    input_7.index = input_1_s.index
    input_7.columns = ['rain-1']

    # Niederschlag -2
    input_7a = ts_rain_1h.resample(timedelta(hours=24 // n_timestep)).mean()
    input_7a = input_7a.iloc[n_timestep - 2:-2, :]
    input_7a.index = input_1_s.index
    input_7a.columns = ['rain-2']

    # Niederschlag -3
    input_7b = ts_rain_1h.resample(timedelta(hours=24 // n_timestep)).mean()
    input_7b = input_7b.iloc[n_timestep - 3:-3, :]
    input_7b.index = input_1_s.index
    input_7b.columns = ['rain-3']

    # Bodenfeuchtigkeit15
    input_8 = ts_bf15_1h.resample(timedelta(hours=24 // n_timestep)).mean()
    input_8 = input_8.iloc[n_timestep:, :]
    input_8.index = input_1_s.index
    input_8.columns = ['bf15']

    # Bodenfeuchtigkeit15 -1
    input_8a = ts_bf15_1h.resample(timedelta(hours=24 // n_timestep)).mean()
    input_8a = input_8a.iloc[n_timestep - 1:-1, :]
    input_8a.index = input_1_s.index
    input_8a.columns = ['bf15-1']

    # Bodenfeuchtigkeit15 -2
    input_8b = ts_bf15_1h.resample(timedelta(hours=24 // n_timestep)).mean()
    input_8b = input_8b.iloc[n_timestep - 2:-2, :]
    input_8b.index = input_1_s.index
    input_8b.columns = ['bf15-2']

    # Bodenfeuchtigkeit15 -3
    input_8c = ts_bf15_1h.resample(timedelta(hours=24 // n_timestep)).mean()
    input_8c = input_8c.iloc[n_timestep - 3:-3, :]
    input_8c.index = input_1_s.index
    input_8c.columns = ['bf15-3']

    # Schneeschmelzmenge -1
    input_9 = ts_schmelz_1h.resample(timedelta(hours=24 // n_timestep)).mean()
    input_9 = input_9.iloc[n_timestep - 1:-1, :]
    input_9.index = input_1_s.index
    input_9.columns = ['schmelz-1']

    # Schneeschmelzmenge -2
    input_9a = ts_schmelz_1h.resample(timedelta(hours=24 // n_timestep)).mean()
    input_9a = input_9a.iloc[n_timestep - 2:-2, :]
    input_9a.index = input_1_s.index
    input_9a.columns = ['schmelz-2']

    # Schneeschmelzmenge -3
    input_9b = ts_schmelz_1h.resample(timedelta(hours=24 // n_timestep)).mean()
    input_9b = input_9b.iloc[n_timestep - 3:-3, :]
    input_9b.index = input_1_s.index
    input_9b.columns = ['schmelz-3']

    # Normalisiere Input (im Prognosemodus hole min, max aus Training)
    input_data = pd.concat([input_1_s, input_1_c, input_4, input_5, input_6, input_6a, input_6b, input_7, input_7a,
                            input_7b, input_8, input_8a, input_8b, input_8c, input_9, input_9a, input_9b],
                           axis=1, join="inner")
    if idx_train:
        min_col = input_data.min()
        max_col = input_data.max()
        with open(f'tasks//Input//models_keras//zufluss//{str_model}//min_col.pkl', 'wb') as file:
            pickle.dump(min_col, file)
        with open(f'tasks//Input//models_keras//zufluss//{str_model}//max_col.pkl', 'wb') as file:
            pickle.dump(max_col, file)
    else:
        with open(f'tasks//Input//models_keras//zufluss//{str_model}//min_col.pkl', 'rb') as file:
            min_col = pickle.load(file)
        with open(f'tasks//Input//models_keras//zufluss//{str_model}//max_col.pkl', 'rb') as file:
            max_col = pickle.load(file)

    input_data = (input_data - min_col) / (max_col - min_col)
    n_input = input_data.shape[1]

    # entferne ungewollte Trainingstage
    if idx_train:
        return input_data[(input_3['useday'] > 0)].to_numpy().reshape(
            [-1, n_timestep, n_input]), input_data.columns.get_loc('base_day_-1')
    else:
        return input_data.to_numpy().reshape([-1, n_timestep, n_input]), input_data.columns.get_loc('base_day_-1')


def model_train_zufluss(str_model, date_from, date_to):

    # Config Parser
    config = configparser.ConfigParser()
    config.read('tasks//Input//models_keras//model_config.ini')

    # Modell-Spezifikationen
    n_timestep = int(config.get(str_model, 'n_timestep'))
    n_hidden_neurons = int(config.get(str_model, 'n_hidden_neurons'))
    n_valid = int(config.get(str_model, 'n_valid'))
    n_patience = int(config.get(str_model, 'n_patience'))
    n_epochs = int(config.get(str_model, 'n_epochs'))
    n_batch_size = int(config.get(str_model, 'n_batch_size'))

    # ----------
    # lade Input
    # ----------
    x, idx_day = load_input(str_model, date_from, date_to, n_timestep, True)
    n_input = x.shape[2]

    # -----------------
    # formatiere Output
    # -----------------
    # lade Basiszeitreihe
    ts_base_id = config.get(str_model, 'base')
    ts_base = get_timeseries_1h('EPAG_ENERGIE', ts_base_id, date_from, date_to).resample(timedelta(hours=24 // n_timestep)).mean()

    # Tag verwenden (1d)
    ts_useday_1d_id = config.get(str_model, 'useday_1d')
    ts_useday_1d = get_timeseries_1d('EPAG_ENERGIE', ts_useday_1d_id, date_from, date_to, 'meanvalues')

    # Normalisiere Zielgrösse
    base_data = ts_base[(ts_base.index.hour == 0)]
    for i_col in range(1, n_timestep):
        data_add = ts_base[(ts_base.index.hour == range(0, 24, 24//n_timestep)[i_col])]
        data_add.index = base_data.index
        base_data = pd.concat([base_data, data_add], axis=1, join='inner')
    base_data.columns = [f'value{i}' for i in range(n_timestep)]
    base_data_min = min(base_data.min())
    base_data -= base_data_min
    base_data_max = max(base_data.max())
    base_data /= base_data_max
    # speichere Min und Max für Prognose
    with open(f'tasks//Input//models_keras//zufluss//{str_model}//base_data_min.pkl', 'wb') as file:
        pickle.dump(base_data_min, file)
    with open(f'tasks//Input//models_keras//zufluss//{str_model}//base_data_max.pkl', 'wb') as file:
        pickle.dump(base_data_max, file)

    # ---------------------------
    # Modell bauen und trainieren
    # ---------------------------
    # Modell bauen
    input_layer = Input(shape=(n_timestep, n_input), dtype='float64')
    hidden_layer = SimpleRNN(n_hidden_neurons, activation='sigmoid')(input_layer)
    output_layer = Dense(n_timestep, activation='sigmoid')(hidden_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='mean_squared_error', optimizer='Adam')
    print(model.summary())

    # Daten vorbereiten
    y = base_data[ts_useday_1d['value'] > 0]
    x_train = x[0:-1 * n_valid, :, :]
    y_train = y.iloc[0:-1 * n_valid, :]
    x_valid = x[-1 * n_valid:, :, :]
    y_valid = y.iloc[-1 * n_valid:, :]

    # definiere Callbacks
    stop_condition = EarlyStopping(monitor='val_loss', mode='min', verbose=0,
                                   patience=n_patience, restore_best_weights=True)

    # Modell trainieren
    training = model.fit(x=x_train, y=y_train, validation_data=(x_valid, y_valid), epochs=n_epochs,
                         verbose=0, batch_size=n_batch_size, callbacks=stop_condition)

    # ----------
    # Auswertung
    # ----------
    # Zeitreihen berechnen
    ts_y_train = back_to_ts(y_train, n_timestep)
    ts_y_valid = back_to_ts(y_valid, n_timestep)
    y_pred = y_valid.copy()
    y_pred.iloc[:, :] = model.predict(x_valid)
    ts_y_pred = back_to_ts(y_pred, n_timestep)
    y_pred_train = y_train.copy()
    y_pred_train.iloc[:, :] = model.predict(x_train)
    ts_y_pred_train = back_to_ts(y_pred_train, n_timestep)

    # Plots
    plt.figure(figsize=[16, 8])
    plt.plot(ts_y_train.index, ts_y_train['value'])
    plt.plot(ts_y_pred_train.index, ts_y_pred_train['value'])
    plt.xlabel(f"Mape Training: {'{:.3f}'.format(rmae_func(ts_y_pred_train, ts_y_train) * 100)} %")
    plt.savefig(f"tasks//Input//models_keras//zufluss//{str_model}//Training.png")
    plt.figure(figsize=[16, 8])
    plt.plot(ts_y_valid.index, ts_y_valid['value'])
    plt.plot(ts_y_pred.index, ts_y_pred['value'])
    plt.xlabel(f"Mape Validierung: {'{:.3f}'.format(rmae_func(ts_y_pred, ts_y_valid) * 100)} %")
    plt.savefig(f"tasks//Input//models_keras//zufluss//{str_model}//Validierung.png")
    plt.close('all')

    # speichere Modell
    save_model(model, f'tasks//Input//models_keras//zufluss//{str_model}//model')


def model_calc_zufluss(str_model, date_from, date_to):

    # Prognosezeitraum (Vortag dazunehmen für erste Stunde in Sommerzeit)
    tz_summer = timezone('CET')
    tz_winter = timezone('Etc/GMT-1')
    date_from_cut = tz_summer.localize(date_from).astimezone(tz_winter)
    date_from -= timedelta(days=1)
    date_to += timedelta(days=1)
    date_to_cut = tz_summer.localize(date_to).astimezone(tz_winter)

    # lade Modell
    model = load_model(f'tasks//Input//models_keras//zufluss//{str_model}//model')
    n_timestep = model.layers[0].output_shape[0][1]

    # lade Input
    x, idx_day = load_input(str_model, date_from, date_to + timedelta(days=1), n_timestep, False)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    idx_act_day = max(0, int((today - date_from) / timedelta(days=1)))

    # Prognose (skalieren mit Min und Max aus Training)
    with open(f'tasks//Input//models_keras//zufluss//{str_model}//base_data_min.pkl', 'rb') as file:
        base_data_min = pickle.load(file)
    with open(f'tasks//Input//models_keras//zufluss//{str_model}//base_data_max.pkl', 'rb') as file:
        base_data_max = pickle.load(file)
    with open(f'tasks//Input//models_keras//zufluss//{str_model}//max_col.pkl', 'rb') as file:
        max_col = pickle.load(file)
    with open(f'tasks//Input//models_keras//zufluss//{str_model}//min_col.pkl', 'rb') as file:
        min_col = pickle.load(file)
    model_output = pd.DataFrame(data=model.predict(x), index=pd.date_range(start=date_from, end=date_to, freq='1d',
                                                                           tz='Etc/GMT-1'))
    # Schrittweise Prognose anpassen für Horizont > 1 Tag
    while idx_act_day + 1 < len(model_output):
        day_mean = model_output.iloc[idx_act_day].mean() * base_data_max + base_data_min
        day_mean = (day_mean - min_col.iloc[idx_day]) / (max_col.iloc[idx_day] - min_col.iloc[idx_day])
        idx_act_day += 1
        x[idx_act_day, :, idx_day] = day_mean
        model_output.iloc[idx_act_day] = model.predict(x[idx_act_day:idx_act_day + 1])
    model_output = back_to_ts(model_output, n_timestep) * base_data_max + base_data_min
    model_output = model_output.resample('1h').asfreq().shift(12 // n_timestep).interpolate('cubic')
    model_output = model_output[(model_output.index >= date_from_cut) & (model_output.index < date_to_cut)]

    write_DWH(os.path.join(r'\\srvedm11', 'Import', 'Messdaten', 'EPAG_Energie', 'DWH_EX_60'),
              f'{str_model}_Prediction_EPAG', 'Python', 'm3/s', model_output)

    d_next = (timezone('CET').localize(datetime.now()) + timedelta(days=1))\
        .replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone('Etc/GMT-1'))
    write_DWH(os.path.join(r'\\srvedm11', 'Import', 'Messdaten', 'EPAG_Energie', 'DWH_EX_60'),
              f'{str_model}_Prediction_EPAG', 'Python_d+1', 'm3/s', model_output.loc[model_output.index >= d_next])


############################################$


# convert model-output to timeseries
import pandas as pd
from statistics import mean
from datetime import timedelta


def back_to_ts(frame, n_bits):
    data_out = pd.DataFrame()
    for i in range(0, len(frame)):
        index_ts = pd.date_range(start=frame.index[i], end=frame.index[i] + timedelta(hours=24 - 24 / n_bits),
                                 freq=timedelta(hours=24 / n_bits), tz='Etc/GMT-1')
        data_app = pd.DataFrame(index=index_ts, columns=['value'])
        data_out = data_out.append(data_app)
    data_out.index.name = 'Timestamp'
    data_out['value'] = [frame.iloc[i // n_bits, i % n_bits] for i in range(0, len(data_out))]
    return data_out


def mape_func(df1, df2):
    return mean(abs((df1 - df2) / df2).iloc[:, 0])


def rmae_func(df1, df2):
    return mean(abs(df1 - df2).iloc[:, 0]) / mean(df2.iloc[:, 0])




#######################


import pandas as pd
from datetime import datetime
from os import path
import csv
from shutil import copy

def write_DWH(str_path: str, str_tsname: str, str_property: str, str_unit: str,
              df_timeseries: pd.DataFrame, filename=None):

    bool_copy = False

    if str_path.find('EPAG_PFM') >= 1 and str_path.find('srvedm11') >= 1:
        str_path_old = str_path.replace('srvedm11', 'srvedm11')
        bool_copy = True

    # Winterzeit!

    col_name = df_timeseries.columns[0]
    if filename is None:
        datestr_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{datestr_now}_{str_tsname}_{str_property}_DWH.csv'

    with open(path.join(str_path, filename), 'wt', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=";")

        # Header
        csv_writer.writerow([f"Timeseries: '{str_tsname}'"])
        csv_writer.writerow([f"Property: {str_property}"])
        csv_writer.writerow([f"Unit: {str_unit}"])
        csv_writer.writerow([f"From: {df_timeseries.index[0].strftime('%d.%m.%Y %H:%M:%S')}"])
        csv_writer.writerow([f"To: {df_timeseries.index[len(df_timeseries.index) - 1].strftime('%d.%m.%Y %H:%M:%S')}"])
        csv_writer.writerow([])
        csv_writer.writerow(["AsOf", "Knowledge", "Value", "State", ""])

        # Zeilen
        for i_row in range(len(df_timeseries.index)):
            csv_writer.writerow([f"{df_timeseries.index[i_row].strftime('%d.%m.%Y %H:%M:%S')}",
                                 f"{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}",
                                 f"{df_timeseries[col_name][i_row]}", "-", ""])

    if bool_copy == True:
        with open(path.join(str_path_old, filename), 'wt', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=";")

            # Header
            csv_writer.writerow([f"Timeseries: '{str_tsname}'"])
            csv_writer.writerow([f"Property: {str_property}"])
            csv_writer.writerow([f"Unit: {str_unit}"])
            csv_writer.writerow([f"From: {df_timeseries.index[0].strftime('%d.%m.%Y %H:%M:%S')}"])
            csv_writer.writerow(
                [f"To: {df_timeseries.index[len(df_timeseries.index) - 1].strftime('%d.%m.%Y %H:%M:%S')}"])
            csv_writer.writerow([])
            csv_writer.writerow(["AsOf", "Knowledge", "Value", "State", ""])

            # Zeilen
            for i_row in range(len(df_timeseries.index)):
                csv_writer.writerow([f"{df_timeseries.index[i_row].strftime('%d.%m.%Y %H:%M:%S')}",
                                     f"{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}",
                                     f"{df_timeseries[col_name][i_row]}", "-", ""])

    idx_success = True
    return idx_success


def write_DWH_datum_am_ende(str_path: str, str_tsname: str, str_property: str, str_unit: str,
              df_timeseries: pd.DataFrame, filename=None):

    bool_copy = False

    if str_path.find('EPAG_PFM') >= 1 and str_path.find('srvedm11') >= 1:
        str_path_old = str_path.replace('srvedm11', 'srvedm11')
        bool_copy = True

    # Winterzeit!

    col_name = df_timeseries.columns[0]
    if filename is None:
        datestr_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{str_tsname}_{str_property}_{datestr_now}.csv'

    with open(path.join(str_path, filename), 'wt', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=";")

        # Header
        csv_writer.writerow([f"Timeseries: '{str_tsname}'"])
        csv_writer.writerow([f"Property: {str_property}"])
        csv_writer.writerow([f"Unit: {str_unit}"])
        csv_writer.writerow([f"From: {df_timeseries.index[0].strftime('%d.%m.%Y %H:%M:%S')}"])
        csv_writer.writerow([f"To: {df_timeseries.index[len(df_timeseries.index) - 1].strftime('%d.%m.%Y %H:%M:%S')}"])
        csv_writer.writerow([])
        csv_writer.writerow(["AsOf", "Knowledge", "Value", "State", ""])

        # Zeilen
        for i_row in range(len(df_timeseries.index)):
            csv_writer.writerow([f"{df_timeseries.index[i_row].strftime('%d.%m.%Y %H:%M:%S')}",
                                 f"{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}",
                                 f"{df_timeseries[col_name][i_row]}", "-", ""])

    if bool_copy == True:
        with open(path.join(str_path_old, filename), 'wt', newline='') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=";")

            # Header
            csv_writer.writerow([f"Timeseries: '{str_tsname}'"])
            csv_writer.writerow([f"Property: {str_property}"])
            csv_writer.writerow([f"Unit: {str_unit}"])
            csv_writer.writerow([f"From: {df_timeseries.index[0].strftime('%d.%m.%Y %H:%M:%S')}"])
            csv_writer.writerow(
                [f"To: {df_timeseries.index[len(df_timeseries.index) - 1].strftime('%d.%m.%Y %H:%M:%S')}"])
            csv_writer.writerow([])
            csv_writer.writerow(["AsOf", "Knowledge", "Value", "State", ""])

            # Zeilen
            for i_row in range(len(df_timeseries.index)):
                csv_writer.writerow([f"{df_timeseries.index[i_row].strftime('%d.%m.%Y %H:%M:%S')}",
                                     f"{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}",
                                     f"{df_timeseries[col_name][i_row]}", "-", ""])

    idx_success = True
    return idx_success



