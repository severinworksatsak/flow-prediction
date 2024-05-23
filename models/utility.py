import pandas as pd
from datetime import datetime, timedelta
from math import sin, cos, pi
from calendar import monthrange
import pickle
import configparser
import json
from pathlib import Path

from solutil import dbqueries as db


def load_input(str_model: str, date_from, date_to, n_timestep: int = None, use_day: bool = True,
               use_day_var:str='useday_1d_lag0', mandant_user: str = None, mandant_pwd: str = None,
               mandant_addr: str = None):
    """
    Load feature matrix for machine learning models according to variable specification in config.json, including
    built-in time interval rescaling of timeseries.

    Parameters:
    :param str_model: (str) Model name as specified under 'models' section in json config.
    :param date_from: Start of daterange to-be-retrieved by the function. Can either be a datetime object
                      or string of the format '%d.%m.%Y'.
    :param date_to: End of daterange to-be-retrieved by the function. Can either be a datetime object
                    or string of the format '%d.%m.%Y'.
    :param n_timestep: (int) Number of intraday timestemps that should be passed onto the model's input layer. Loaded
                       from the config json.
    :param use_day: (bool) Flag variable to indicate whether use_day column is included in the parameter specification
                    in the json config. Default is True, in which case filtering will be performed using use_day_var.
    :param use_day_var: (str) Column name of the use_day classifier. Important: Must be final name as occurring in the
                        output dataframe. Hence, the name is '{var_name}_lag{n_lag}'. Default is useday_1d_lag0.
    :param mandant_user: (str)
    :param mandant_pwd: (str)
    :param mandant_addr: (str)

    Returns:
    :return: (pd.DataFrame) DataFrame containing feature matrix with Etc/GMT-1-timestamped index.

    Notes:
    :note: Function can only extract time series from the same Belvis mandant at the same time
    :note: For complete specification, each variable should have the following config parameterization:
            - ts_id: (int) Belvis ID of a specific timeseries.
            - lags: (list) List of lags for an individual variable.
            - freq: (str) Frequency of the timeseries; options are '1d', '1h' and '15min'.
            - time_shift: (str) Determines the direction of the timeshift; 'date_from' entails subtraction of 1 day
                          from date_from, while 'date_to' leads to addition of 1 day to date_to.
            - str_table: (str) Only for daily variables. Allows to control which Belvis table TS_{str_table} is queried.
    """

    # Load Environment Variables
    if mandant_user is None:
        env_vars = db.get_env_variables('EPAG_ENERGIE')
        mandant_user = env_vars['mandant_user']
        mandant_pwd = env_vars['mandant_pwd']
        mandant_addr = env_vars['mandant_addr']

    # Define config path
    config_file = "config/config.json"
    config_path = Path(config_file)

    # Load config from directory
    with open(config_path, "r") as jsonfile:
        config = json.load(jsonfile)

    # Save model config
    model_config = config['model'][str_model]

    # Get n_timestep from config
    if n_timestep is None:
        n_timestep = model_config['parameters']['architecture']['n_timestep']

    # Convert input dates to datetime
    date_from = datetime.strptime(date_from, '%d.%m.%Y') if isinstance(date_from, str) else date_from
    date_to = datetime.strptime(date_to, '%d.%m.%Y') if isinstance(date_to, str) else date_to

    # Recursively Load Timeseries and Lag Variables --------------------------------------------------------------------

    counter = 0

    for input_variable in model_config['inputs'].keys():
        # Extract input variable parameters
        ts_id_i = model_config['inputs'][input_variable]['ts_id']
        lags_i = model_config['inputs'][input_variable]['lags']

        # Get time ranges
        if model_config['inputs'][input_variable].get('time_shift') == 'date_from':
            date_from = date_from - timedelta(days=1)
        elif model_config['inputs'][input_variable].get('time_shift') == 'date_to':
            date_to = date_to + timedelta(days=1)

        # Create initial df from scratch -> Deliberately elongated to accommodate both time_shifts
        if counter < 1:
            date_index = pd.date_range(start=date_from - timedelta(days=1),
                                       end=date_to + timedelta(days=1),
                                       tz='Etc/GMT-1',
                                       freq=f'{24//n_timestep}h')
            df_collect = pd.DataFrame(index=date_index)

        # Conditional Data Import from Belvis
        match model_config['inputs'][input_variable]['freq']:
            case '1h':
                # Load daily data
                loaded_ts = db.get_timeseries_1h(ts_id_i, date_from, date_to,
                                                 mandant_user, mandant_pwd, mandant_addr)

                # Resample hourly data
                resampled_ts = loaded_ts.resample(timedelta(hours=24//n_timestep)).mean()
            case '1d':
                # Load daily data
                str_table = model_config['inputs'][input_variable]['str_table']
                loaded_ts = db.get_timeseries_1d(ts_id_i, date_from, date_to,
                                     mandant_user, mandant_pwd, mandant_addr, str_table)
                loaded_ts = loaded_ts['value'] # no dynamic col_name required because no option to change -> hardcopy 'value' ok

                # Resample daily data
                resampled_ts = loaded_ts.resample(timedelta(hours=24//n_timestep)).ffill()
            case '15min':
                # Load daily data
                loaded_ts = db.get_timeseries_15min(ts_id_i, date_from, date_to,
                                                 mandant_user, mandant_pwd, mandant_addr)

                # Resample hourly data
                resampled_ts = loaded_ts.resample(timedelta(hours=24 // n_timestep)).mean()
            case _:
                raise ValueError("Frequency neither of '1d', '1h', or '15min'!")

        counter += 1

        # Lag variables
        for lag in lags_i:
            # Lag variables and fill resulting nas
            lagged_ts = resampled_ts.shift(lag)
            lagged_ts.bfill(inplace=True)

            # Add lagged variable to collection dataframe
            variable_name = f'{input_variable}_lag{lag}'
            df_collect[variable_name] = lagged_ts

    # Clip df_collect
    df_collect.dropna(how='all', inplace=True)

    # Remove unwanted training days
    if use_day:
        df_collect = df_collect.query(f'useday_1d_lag0 > 0').copy()
        print('Unwanted days removed.')

    return df_collect


# Prepare Inputs for ML Model
def prepare_inputs(df_features, idx_train:bool=True):
    pass




























def load_input2(str_model:str, date_from, date_to, n_timestep:int=None, idx_train:bool=True,
               mandant_user:str=None, mandant_pwd:str=None, mandant_addr:str=None):

    # Load Environment Variables
    if mandant_user is None:
        env_vars = db.get_env_variables('EPAG_ENERGIE')
        mandant_user = env_vars['mandant_user']
        mandant_pwd = env_vars['mandant_pwd']
        mandant_addr = env_vars['mandant_addr']

    # Config Parser
    config = configparser.ConfigParser()
    config.read('config//model_config.ini')

    # Get n_timestep from config
    if n_timestep is None:
        n_timestep = int(config.get(str_model, 'n_timestep'))

    # Load Timeseries --------------------------------------------------------------------------------------------------

    # Tagessumme Abfluss (1d)
    ts_base_1d_id = int(config.get(str_model, 'base_1d'))
    ts_base_1d = db.get_timeseries_1d(ts_base_1d_id, date_from - timedelta(days=1), date_to,
                                      mandant_user, mandant_pwd, mandant_addr, str_table='meanvalues')

    # Tag verwenden (1d)
    ts_useday_1d_id = int(config.get(str_model, 'useday_1d'))
    ts_useday_1d = db.get_timeseries_1d(ts_useday_1d_id, date_from, date_to + timedelta(days=1),
                                        mandant_user, mandant_pwd, mandant_addr, str_table='meanvalues')

    # Temperatur (1h)
    ts_temp_1h_id = int(config.get(str_model, 'temp_1h'))
    ts_temp_1h = db.get_timeseries_1h(ts_temp_1h_id, date_from - timedelta(days=1), date_to,
                                      mandant_user, mandant_pwd, mandant_addr)

    # Globalstrahlung (1h)
    ts_glob_1h_id = int(config.get(str_model, 'glob_1h'))
    ts_glob_1h = db.get_timeseries_1h(ts_glob_1h_id, date_from - timedelta(days=1), date_to,
                                      mandant_user, mandant_pwd, mandant_addr)

    # Niederschlag (1h)
    ts_rain_1h_id = int(config.get(str_model, 'rain_1h'))
    ts_rain_1h = db.get_timeseries_1h(ts_rain_1h_id, date_from - timedelta(days=1), date_to,
                                      mandant_user, mandant_pwd, mandant_addr)

    # Bodenfeuchtigkeit (1h)
    ts_bf15_1h_id = int(config.get(str_model, 'bf15_1h'))
    ts_bf15_1h = db.get_timeseries_1h(ts_bf15_1h_id, date_from - timedelta(days=1), date_to,
                                      mandant_user, mandant_pwd, mandant_addr)

    # Schneeschmelzmenge (1h)
    ts_schmelz_1h_id = int(config.get(str_model, 'schmelz_1h'))
    ts_schmelz_1h = db.get_timeseries_1h(ts_schmelz_1h_id, date_from - timedelta(days=1), date_to,
                                      mandant_user, mandant_pwd, mandant_addr)

    # Concatenate Input Variables --------------------------------------------------------------------------------------

    # Tag des Jahres Sinus
    input_1_s = ts_useday_1d.resample(timedelta(hours=24 // n_timestep)).ffill()
    input_1_s = input_1_s.iloc[:-1]
    input_1_s['value'] = [sin(element.dayofyear / (monthrange(element.year, 2)[1] + 337) * 2 * pi) for element in
                          input_1_s.index]
    input_1_s.name = 'yearday_sin'

    # Tag des Jahres Cosinus
    input_1_c = ts_useday_1d.resample(timedelta(hours=24 // n_timestep)).ffill()
    input_1_c = input_1_c.iloc[:-1]
    input_1_c['value'] = [cos(element.dayofyear / (monthrange(element.year, 2)[1] + 337) * 2 * pi) for element in
                          input_1_c.index]
    input_1_c.name = 'yearday_cos'

    # Tag für Training verwenden
    input_3 = ts_useday_1d.resample(timedelta(hours=24 // n_timestep)).ffill()
    input_3 = input_3.iloc[:-1]
    input_3.columns = 'useday'

    # Abfluss Vortag
    input_4 = ts_base_1d.resample(timedelta(hours=24 // n_timestep)).ffill()
    input_4 = input_4.iloc[:-1]
    input_4.name = 'base_day_-1'
    input_4.index = input_1_s.index
    input_4 = input_4.fillna(0)

    # Temperatur -1
    input_5 = ts_temp_1h.resample(timedelta(hours=24 // n_timestep)).mean()
    input_5 = input_5.iloc[n_timestep - 1:-1]
    input_5.index = input_1_s.index
    input_5.name = 'temp-1'

    # Globalstrahlung -1
    input_6 = ts_glob_1h.resample(timedelta(hours=24 // n_timestep)).mean()
    input_6 = input_6.iloc[n_timestep - 1:-1]
    input_6.index = input_1_s.index
    input_6.name = 'glob-1'

    # Globalstrahlung -2
    input_6a = ts_glob_1h.resample(timedelta(hours=24 // n_timestep)).mean()
    input_6a = input_6a.iloc[n_timestep - 2:-2]
    input_6a.index = input_1_s.index
    input_6a.name = 'glob-2'

    # Globalstrahlung -3
    input_6b = ts_glob_1h.resample(timedelta(hours=24 // n_timestep)).mean()
    input_6b = input_6b.iloc[n_timestep - 3:-3]
    input_6b.index = input_1_s.index
    input_6b.name = 'glob-3'

    # Niederschlag -1
    input_7 = ts_rain_1h.resample(timedelta(hours=24 // n_timestep)).mean()
    input_7 = input_7.iloc[n_timestep - 1:-1]
    input_7.index = input_1_s.index
    input_7.name = 'rain-1'

    # Niederschlag -2
    input_7a = ts_rain_1h.resample(timedelta(hours=24 // n_timestep)).mean()
    input_7a = input_7a.iloc[n_timestep - 2:-2]
    input_7a.index = input_1_s.index
    input_7a.name = 'rain-2'

    # Niederschlag -3
    input_7b = ts_rain_1h.resample(timedelta(hours=24 // n_timestep)).mean()
    input_7b = input_7b.iloc[n_timestep - 3:-3]
    input_7b.index = input_1_s.index
    input_7b.name = 'rain-3'

    # Bodenfeuchtigkeit15
    input_8 = ts_bf15_1h.resample(timedelta(hours=24 // n_timestep)).mean()
    input_8 = input_8.iloc[n_timestep:]
    input_8.index = input_1_s.index
    input_8.name = 'bf15'

    # Bodenfeuchtigkeit15 -1
    input_8a = ts_bf15_1h.resample(timedelta(hours=24 // n_timestep)).mean()
    input_8a = input_8a.iloc[n_timestep - 1:-1]
    input_8a.index = input_1_s.index
    input_8a.name = 'bf15-1'

    # Bodenfeuchtigkeit15 -2
    input_8b = ts_bf15_1h.resample(timedelta(hours=24 // n_timestep)).mean()
    input_8b = input_8b.iloc[n_timestep - 2:-2]
    input_8b.index = input_1_s.index
    input_8b.name = 'bf15-2'

    # Bodenfeuchtigkeit15 -3
    input_8c = ts_bf15_1h.resample(timedelta(hours=24 // n_timestep)).mean()
    input_8c = input_8c.iloc[n_timestep - 3:-3]
    input_8c.index = input_1_s.index
    input_8c.name = 'bf15-3'

    # Schneeschmelzmenge -1
    input_9 = ts_schmelz_1h.resample(timedelta(hours=24 // n_timestep)).mean()
    input_9 = input_9.iloc[n_timestep - 1:-1]
    input_9.index = input_1_s.index
    input_9.name = 'schmelz-1'

    # Schneeschmelzmenge -2
    input_9a = ts_schmelz_1h.resample(timedelta(hours=24 // n_timestep)).mean()
    input_9a = input_9a.iloc[n_timestep - 2:-2]
    input_9a.index = input_1_s.index
    input_9a.name = 'schmelz-2'

    # Schneeschmelzmenge -3
    input_9b = ts_schmelz_1h.resample(timedelta(hours=24 // n_timestep)).mean()
    input_9b = input_9b.iloc[n_timestep - 3:-3]
    input_9b.index = input_1_s.index
    input_9b.name = 'schmelz-3'

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





