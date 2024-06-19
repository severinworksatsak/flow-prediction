import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pytz import timezone
from math import sin, cos, pi
from calendar import monthrange
import pickle
import configparser
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor

from solutil import dbqueries as db


def get_dates_from_config(str_model: str):
    """
    Method to load date parameters for load_input.
    :param str_model: Model name as used in section header of json config
    :return: date dict containing date_from and date_to parameters in datetime format.
    """
    # Load config from directory
    with open(Path("config/config.json"), "r") as jsonfile:
        config = json.load(jsonfile)

    # Get dates
    date_from_str = config['model'][str_model]['parameters']['training']['train_start']
    date_to_diff = int(config['model'][str_model]['parameters']['training']['last_day_train'])

    date_from = datetime.strptime(date_from_str, '%d.%m.%Y')
    date_to = datetime.now() + timedelta(days=date_to_diff)
    date_to_clean = date_to.replace(hour=0, minute=0, second=0, microsecond=0)

    date_dict = {'date_from': date_from,
                 'date_to': date_to_clean}

    return date_dict


def get_params_from_config(function: str, str_model: str):
    # Load config from directory
    with open(Path("config/config.json"), "r") as jsonfile:
        config = json.load(jsonfile)

    # Extract model config
    model_config = config['model'][str_model]

    # Extract parameters depending on function
    param_dict = {}
    match function:
        case 'dates':
            date_from_str = model_config['parameters']['training']['train_start']
            date_to_diff = int(model_config['parameters']['training']['last_day_train'])

            date_from = datetime.strptime(date_from_str, '%d.%m.%Y')
            date_to = datetime.now() + timedelta(days=date_to_diff)
            date_to_clean = date_to.replace(hour=0, minute=0, second=0, microsecond=0)

            param_dict['date_from'] = date_from
            param_dict['date_to'] = date_to_clean
        case 'model_train':
            param_dict['n_patience'] = model_config['parameters']['architecture']['n_patience']
            param_dict['n_epochs'] = model_config['parameters']['architecture']['n_epochs']
            param_dict['n_batch_size'] = model_config['parameters']['architecture']['n_batch_size']
        case _:
            raise ValueError('Provided function name is not available.')

    return param_dict


def load_input(str_model: str, date_from, date_to, n_timestep: int = None, use_day: bool = True,
               use_day_var: str = 'useday_1d_lag0', mandant_user: str = None, mandant_pwd: str = None,
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

        # Get modified time ranges -> evade circular updating of date variables
        if model_config['inputs'][input_variable].get('time_shift') == 'date_from':
            date_from_mod = date_from - timedelta(days=1)
            date_to_mod = date_to
        elif model_config['inputs'][input_variable].get('time_shift') == 'date_to':
            date_from_mod = date_from
            date_to_mod = date_to + timedelta(days=1)

        # Create initial df from scratch -> Deliberately elongated to accommodate both time_shifts
        if counter < 1:
            date_index = pd.date_range(start=date_from_mod - timedelta(days=1),
                                       end=date_to_mod + timedelta(days=1),
                                       tz='Etc/GMT-1',
                                       freq=f'{24 // n_timestep}h')
            df_collect = pd.DataFrame(index=date_index)

        # Conditional Data Import from Belvis
        match model_config['inputs'][input_variable]['freq']:
            case '1h':
                # Load daily data
                loaded_ts = db.get_timeseries_1h(ts_id_i, date_from_mod, date_to_mod,
                                                 mandant_user, mandant_pwd, mandant_addr)

                # Resample hourly data
                resampled_ts = loaded_ts.resample(timedelta(hours=24 // n_timestep)).mean()
            case '1d':
                # Load daily data
                str_table = model_config['inputs'][input_variable]['str_table']
                loaded_ts = db.get_timeseries_1d(ts_id_i, date_from_mod, date_to_mod,
                                                 mandant_user, mandant_pwd, mandant_addr, str_table)
                loaded_ts = loaded_ts[
                    'value']  # no dynamic col_name required because no option to change -> hardcopy 'value' ok

                # Resample daily data
                resampled_ts = loaded_ts.resample(timedelta(hours=24 // n_timestep)).ffill()
            case '15min':
                # Load daily data
                loaded_ts = db.get_timeseries_15min(ts_id_i, date_from_mod, date_to_mod,
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

    # # Clip df_collect
    # df_collect.dropna(how='all', inplace=True)

    # Remove unwanted training days
    if use_day:
        df_collect = df_collect.query(f"{use_day_var} > 0").copy()
        df_collect.drop(columns=[f'{use_day_var}'], inplace=True)

    # Handle NAs
    df_collect.dropna(how='all', inplace=True)
    df_collect.ffill(inplace=True)

    return df_collect


# Outlier Detection
def handle_outliers(df, contamination='auto', window_length: int = 6, alpha: float = None):
    """
    Detect outliers based on Local Outlier Factor replace these values with exponential
    weighted moving average of x previous values.
    :param df: Dataframe containing to-be-checked values. Algo will loop through all
               columns of df.
    :param contamination: Contamination mode for LocalOutlierFactor class. Default is 'auto'.
    :param window_length: EWM average window length for outlier imputation.
    :param alpha: Fading parameter for ewm computation. Defaults to 2/(x+1).
    :return: Dataframe with corrected outliers.
    """
    if alpha is None:
        alpha = 2 / (window_length + 1)  # Rule of thumb

    lof = LocalOutlierFactor(contamination=contamination)

    # Loop over all variables
    outliers_dict = {}

    for variable in df.columns:
        print(f'Variable {variable}')
        outlier_preds = lof.fit_predict(df[variable].values.reshape(-1, 1))
        mask = outlier_preds == -1
        variable_mask = df[variable][mask]

        # Impute outliers
        for outlier_i in variable_mask.index:
            # Get EMWA range index
            # date_index = df[variable].index[outlier_i]
            int_index = df[variable].index.get_loc(outlier_i)
            start_index = max(int_index - window_length, 0)

            # Compute EMWA and Assign to original Data
            df_subsample = df[variable].iloc[start_index:int_index]
            emwa_i = df_subsample.ewm(alpha=alpha).mean().iloc[-1]
            df.loc[outlier_i, variable] = emwa_i

    return df


# Prepare Inputs for ML Model
def scale_with_minmax(df_features, str_model: str, idx_train: bool = True):
    # Perform MinMax Scaling
    if idx_train:
        # Get scaling parameters from hist data
        min_col = df_features.min()
        max_col = df_features.max()
        scale_factor = max_col - min_col

        # Save attributes
        with open(f'models//attributes//{str_model}_min_col.pkl', 'wb') as file:
            pickle.dump(min_col, file)
        with open(f'models//attributes//{str_model}_scale_factor.pkl', 'wb') as file:
            pickle.dump(scale_factor, file)
    else:
        # Load attributes
        with open(f'models//attributes//{str_model}_min_col.pkl', 'rb') as file:
            min_col = pickle.load(file)
        with open(f'models//attributes//{str_model}_scale_factor.pkl', 'rb') as file:
            scale_factor = pickle.load(file)

    # Scale dataset
    df_features_scaled = (df_features - min_col) / scale_factor

    return df_features_scaled


# Inverse transform scaling
def inverse_transform_minmax(df_scaled, str_model: str, attributes):
    # Load attributes
    with open(f'models//attributes//{str_model}_min_col.pkl', 'rb') as file:
        min_col = pickle.load(file)
    with open(f'models//attributes//{str_model}_scale_factor.pkl', 'rb') as file:
        scale_factor = pickle.load(file)

    # Rescale dataframe
    if len(attributes) <= 1:
        df_rescaled = df_scaled * scale_factor[attributes].values[0] + min_col[attributes].values[0]
    else:
        df_rescaled = df_scaled * scale_factor[attributes] + min_col[attributes]

    return df_rescaled


# Split training / test set
def split_dataframe(df_features, target_var: str = None, train_share: float = 0.7, **kwargs):
    """
    Separate target variable from feature matrix and perform train-test-split.

    Parameters:
    :param df_features: (pd.DataFrame) Feature dataframe containing both target variable and features.
    :param target_var: (str) Name of the target variable, used for column indexing.
    :param train_share: (float) Share of training data from total dataset.
    :param kwargs: Input parameters for sklearn train_test_split function, e.g. shuffle.

    Returns:
    :return: x_train, x_test, y_train, y_test -> Dataframes with corresponding train / test data.
    """

    # Extract Target Variable & Split
    y_train, y_test = train_test_split(df_features[target_var], train_size=train_share, **kwargs)

    # Remove Target Variable & Split
    x_train, x_test = train_test_split(df_features.drop(columns=[target_var]), train_size=train_share, **kwargs)

    return x_train, x_test, y_train, y_test


# Sequencing for LSTM
def generate_sequences(df, target_var: str, n_lookback: int, n_ahead: int, n_offset:int=0,
                       drop_target: bool = True, continuous:bool=True, n_timestep:int=6,
                       train_share: float = 0.7):
    """
    Generate sequence arrays for LSTM and perform train-test-split.
    :param df: Dataframe containing x and y features.
    :param target_var: Column name of target variable y in df.
    :param n_lookback: Sequence length of features x.
    :param n_ahead: Sequence length of target variable y.
    :param n_offset: Offset interval between feature and target sequence as expressed in number of timesteps.
                     Defaults to 0.
    :param drop_target: Boolean indicator to drop target variable from x sample.
                        Defaults to True.
    :param continuous: Boolean indicator whether sequencing should be continuous or step-wise. Step-wise can
                       be used if the prediction should be only once a day for the entire day ahead.
    :param n_timestep: Number of timesteps in a day. Used to ensure only one sequence per day occurring.
    :param train_share: Train-test split threshold; train_share = % of dataset included
                        in train sample.
    :return: Tuple containing x_train, x_test, y_train, y_test.
    """
    # Separate target variable
    df_y = df[target_var]
    df_x = df.drop(columns=[target_var]) if drop_target else df

    # Prepare sequence variables
    x_list = []
    y_list = []
    df_x_numpy = df_x.to_numpy()
    df_y_numpy = df_y.to_numpy()

    # Create sequences
    for timestep in range(n_lookback, df_x_numpy.shape[0] - n_ahead - n_offset):
        # Slice df into series
        x_sequence_t = df_x_numpy[timestep - n_lookback:timestep]
        y_sequence_t = df_y_numpy[timestep + n_offset:timestep + n_ahead + n_offset]

        # Assign series to strawmans
        x_list.append(x_sequence_t)
        y_list.append(y_sequence_t)

    # Filter out in case of discontinuous sequences
    if not continuous:
        x_list = x_list[::n_timestep]
        y_list = y_list[::n_timestep]

    # Train Test Split & Numpy Conversion
    x_train, x_test, y_train, y_test = train_test_split(np.array(x_list), np.array(y_list),
                                                        train_size=train_share, shuffle=False)

    return x_train, x_test, y_train, y_test


def convert_seq_to_df(seq_array, start_date, n_timestep:int, daily:bool=False):
    # Ensure datetime object
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%d.%m.%Y %H:%M:%S')

    # Localize in Etc/GMT-1 timezone
    start_date = timezone('Etc/GMT-1').localize(start_date)

    # Create datetime index from date range
    date_index = pd.date_range(start=start_date,
                               periods=seq_array.shape[0],
                               tz='Etc/GMT-1',
                               freq=f'{24 // n_timestep}h')

    # Create dataframe
    df_seq = pd.DataFrame().from_records(seq_array)
    df_seq.index = date_index

    return df_seq


def dailydf_to_ts(daily_df, header:str='value'):
    # Extract values
    values = []
    for row in range(len(daily_df)):
        for column in range(len(daily_df.columns)):
            values.append(daily_df.iloc[row, column])

    # Set new date index
    date_index = pd.date_range(start=daily_df.index[0],
                               periods=len(values),
                               tz='Etc/GMT-1',
                               freq=f'{24 // len(daily_df.columns)}h')

    daily_ts = pd.Series(data=values, index=date_index)
    daily_ts.name = header

    return daily_ts




