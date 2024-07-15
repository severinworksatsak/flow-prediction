import pandas as pd
from datetime import datetime, timedelta
from pytz import timezone
from math import sin, cos, pi
from calendar import monthrange
import pickle
from os import path
import csv
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor

from models.solutil_func import get_timeseries_15min, get_timeseries_1h, get_timeseries_1d, get_eval_metrics, \
    get_act_vs_pred_plot, get_env_variables


def get_dates_from_config(str_model: str, training:bool=True):
    """
    Method to load date parameters for load_input.

    :param str_model: (str) Model name as used in section header of json config.
    :param training: (bool) Flag indicating whether in training or prediction mode. If training=True, output dict will
                     contain dates ranging from train_start to last_day_train. If training=False, prediction mode will
                     be entered and the dates range from first_day_calc to last_day_calc.

    :return: date dict containing date_from and date_to parameters in datetime format.
    """
    # Load config from directory
    with open(Path("config/config.json"), "r") as jsonfile:
        config = json.load(jsonfile)

    config_short = config['model'][str_model]['parameters']['training']

    # Get dates
    #TODO: Check if not conversion to Etc/GMT-1 necessary before .replace()
    if training:
        date_from_str = config_short['train_start']
        date_to_diff = int(config_short['last_day_train'])

        date_from = datetime.strptime(date_from_str, '%d.%m.%Y')
        date_to = datetime.now() + timedelta(days=date_to_diff)
        date_to_clean = date_to.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        date_from_int = config_short['first_day_calc']
        date_from = datetime.now() + timedelta(days=date_from_int)
        date_from = date_from.replace(hour=0, minute=0, second=0, microsecond=0)
        date_to_int = config_short['last_day_calc']
        date_to_clean = datetime.now() + timedelta(days=date_to_int)
        date_to_clean = date_to_clean.replace(hour=0, minute=0, second=0, microsecond=0)

    date_dict = {'date_from': date_from,
                 'date_to': date_to_clean}

    #print(f"date_dict from get_dates_from_config: {date_dict}")

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

        case 'build_svr':
            param_dict['hyperparameters'] = model_config['parameters']['architecture']['hyperparameters']
            param_dict['n_offset'] = model_config['parameters']['architecture']['n_offset']

        case 'lstm_sequence':
            param_dict['n_lookback'] = model_config['parameters']['architecture']['n_lookback']
            param_dict['n_ahead'] = model_config['parameters']['architecture']['n_ahead']
            param_dict['n_offset'] = model_config['parameters']['architecture']['n_offset']

        case 'build_lstm':
            param_dict['n_valid'] = model_config['parameters']['architecture']['n_valid']
            param_dict['hyperparameters'] = model_config['parameters']['architecture']['hyperparameters']

        case 'get_n_timestep':
            param_dict['n_timestep'] = model_config['parameters']['architecture']['n_timestep']

        case 'get_label':
            iter_dict = model_config['inputs']

            for var_key, var_value in iter_dict.items():
                if isinstance(var_value, dict) and var_value.get('is_label'):
                    lag = iter_dict[var_key]['lags'][0]
                    label_name = f'{var_key}_lag{lag}'
                    var_name = var_key
            if 'label_name' not in locals():
                raise NameError("No 'is_label' key found in config variable specification.")

            param_dict['label'] = label_name
            param_dict['var_name'] = var_name

        case 'get_labelmean':
            iter_dict = model_config['inputs']

            for var_key, var_value in iter_dict.items():
                if isinstance(var_value, dict) and var_value.get('is_labelmean'):
                    lag = iter_dict[var_key]['lags'][0]
                    label_name = f'{var_key}_lag{lag}'
                    var_name = var_key
            if 'label_name' not in locals():
                raise NameError("No 'is_labelmean' key found in config variable specification.")

            param_dict['label_mean'] = label_name
            param_dict['meanvar_name'] = var_name

        case 'get_ensemble_labels':
            iter_dict = model_config['inputs']

            for var_key, var_value in iter_dict.items():
                if var_key != 'include_doy':
                    lag = iter_dict[var_key]['lags'][0]
                    label_name = f'{var_key}_lag{lag}'
                    param_dict[var_key] = label_name

        case 'get_exogenous_labels':
            iter_dict = model_config['inputs']
            label_list = []

            for var_key, var_value in iter_dict.items():
                if isinstance(var_value, dict) and var_value.get('is_exogenous'):
                    for i_lag in var_value['lags']:
                        label_name = f'{var_key}_lag{i_lag}'
                        label_list.append(label_name)

            param_dict['exog_labels'] = label_list

        case 'get_doyflag':
            param_dict['doy_flag'] = model_config['inputs']['include_doy']

        case 'build_ensemble':
            param_dict['hyperparameters'] = model_config['parameters']['architecture']['hyperparameters']

        case 'get_rnn_baseid':
            param_dict['rnn_id'] = model_config['inputs']['base']['ts_id']

        case _:
            raise ValueError('Provided function name is not available.')

    return param_dict


def load_input(str_model: str, date_from, date_to, n_timestep: int = None, use_day: bool = True,
               use_day_var: str = 'useday_1d_lag0', mandant_user: str = None, mandant_pwd: str = None,
               mandant_addr: str = None, idx_train:bool=True, verbose:int=0):
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
    :param idx_train: (bool) Boolean indicator whether in training or prediction mode. Default is True. If False, label
                      variable will be skipped as it is not required for the prediction in productive mode.
    :param verbose: (int) Indicator whether to print data load progress. Prints if verbose > 0. Default is 0.

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
        env_vars = get_env_variables('EPAG_ENERGIE')
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

    # Exclude doy from loadable inputs as not coming from database
    clean_variables = [key for key in model_config['inputs'].keys() if key != 'include_doy']

    # Remove label from data if not in training (productive)
    if not idx_train:
        var_name = get_params_from_config(function='get_label', str_model=str_model)['var_name']
        clean_variables.remove(var_name)

    for input_variable in clean_variables:
        if verbose >= 1:
            print(input_variable)
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

        # Get modified time ranges -> evade circular updating of time variables & ensure
        date_from_mod = (date_from - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        date_to_mod = (date_to + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

        # Create initial df from scratch -> Deliberately elongated to accommodate both time_shifts
        if counter < 1:
            date_index = pd.date_range(start=date_from_mod, #- timedelta(days=1),
                                       end=date_to_mod, #+ timedelta(days=1),
                                       tz='Etc/GMT-1',
                                       freq=f'{24 // n_timestep}h')
            df_collect = pd.DataFrame(index=date_index)

        # Conditional Data Import from Belvis
        match model_config['inputs'][input_variable]['freq']:
            case '1h':
                # Load daily data
                loaded_ts = get_timeseries_1h(ts_id_i, date_from_mod, date_to_mod,
                                              mandant_user, mandant_pwd, mandant_addr)

                # Resample hourly data
                resampled_ts = loaded_ts.resample(timedelta(hours=24 // n_timestep)).mean()
            case '1d':
                # Load daily data
                str_table = model_config['inputs'][input_variable]['str_table']
                loaded_ts = get_timeseries_1d(ts_id_i, date_from_mod, date_to_mod,
                                              mandant_user, mandant_pwd, mandant_addr, str_table)

                # Resample daily data
                resampled_ts = loaded_ts.resample(timedelta(hours=24 // n_timestep)).ffill(limit=n_timestep)
                # print(f"Loaded_ts {input_variable} Head: {loaded_ts.head(5)}")
                # print(f"{input_variable} Head: {resampled_ts.head(5)}")
                # print(f"{input_variable} Tail: {resampled_ts.tail(5)}")
            case '15min':
                # Load daily data
                loaded_ts = get_timeseries_15min(ts_id_i, date_from_mod, date_to_mod,
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

    # Remove unwanted training days
    if use_day:
        df_collect = df_collect.query(f"{use_day_var} > 0").copy()
        df_collect.drop(columns=[f'{use_day_var}'], inplace=True)

    # Handle NAs
    df_collect.dropna(how='all', inplace=True)
    df_collect.ffill(inplace=True)

    # Clip df on daterange
    date_from = timezone('Etc/GMT-1').localize(date_from)
    date_to = timezone('Etc/GMT-1').localize(date_to)
    df_collect = df_collect.loc[date_from:date_to,]

    return df_collect


# Day of Year Function
def transform_dayofyear(df):
    """
    Get day of year in sinus / cosinus transformed format.

    :param df: Input dataframe with datetime index on which the transformation is performed.

    :return: Input pd.DataFrame with amended doy_sin and doy_cos features.
    """
    df = df.copy()
    date_index = df.index

    # Get sinus wave
    sin_wave = [sin((timestamp.dayofyear + timestamp.hour / 24) /
                    (monthrange(timestamp.year, 2)[1] + 337) * 2 * pi)
                for timestamp in date_index]

    # Get cosinus wave
    cos_wave = [cos((timestamp.dayofyear + timestamp.hour / 24) /
                    (monthrange(timestamp.year, 2)[1] + 337) * 2 * pi)
                for timestamp in date_index]

    # Assign wave series to df
    df['yearday_sin'] = sin_wave
    df['yearday_cos'] = cos_wave

    return df


# Outlier Detection
def handle_outliers(df_outlier, contamination='auto', window_length: int = 6, alpha: float = None, verbose:int=0):
    """
    Detect outliers based on Local Outlier Factor replace these values with exponential
    weighted moving average of x previous values.

    :param df: Dataframe containing to-be-checked values. Algo will loop through all
               columns of df.
    :param contamination: Contamination mode for LocalOutlierFactor class. Default is 'auto'.
    :param window_length: EWM average window length for outlier imputation.
    :param alpha: Fading parameter for ewm computation. Defaults to 2/(x+1).
    :param verbose: (int) Indicator whether to print data load progress. Prints if verbose > 0. Default is 0.

    :return: Dataframe with corrected outliers.
    """
    if alpha is None:
        alpha = 2 / (window_length + 1)  # Rule of thumb

    lof = LocalOutlierFactor(contamination=contamination)

    # Loop over all variables
    df = df_outlier.copy()

    for variable in df.columns:
        if verbose >= 1:
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

            # Skip outlier if subsample is emtpy -> occurs if outlier is in first row of subsample
            if not df_subsample.empty:
                emwa_i = df_subsample.ewm(alpha=alpha).mean().iloc[-1]
                df.loc[outlier_i, variable] = emwa_i

    return df


# Prepare Inputs for ML Model
def scale_with_minmax(df_features, str_model: str, idx_train: bool = True, verbose:int=0):
    """
    Scale dataframe with Min Max Normalization and save weights as .pkl file.

    :param df_features: Dataframe containing to-be-scaled variables.
    :param str_model: Name of prediction model as occurring in config, e.g. inlet1_lstm.
    :param idx_train: Boolean indicator for training / prediction mode.
    :param verbose: Detail level of output print. 0 means no output print, 1 returns mins and scale weights.

    :return: Scaled dataframe and normalization weights in pkl file saved in attributes folder.
    """
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

    if verbose == 1:
        print(f"Minima per Variable:")
        print(f"{min_col}")
        print(f"Scale Factors per Variable:")
        print(f"{scale_factor}")

    # Scale dataset
    df_features_scaled = (df_features.copy() - min_col) / scale_factor

    return df_features_scaled


# Inverse transform scaling
def inverse_transform_minmax(df_scaled, str_model: str, attributes:list, verbose:int=0):
    """
    Reverse min-max scaling of data sets based on saved weights.

    :param df_scaled: Scaled dataframe, e.g. output of ML model.
    :param str_model: Name of prediction model as occurring in config, e.g. inlet1_lstm.
    :param attributes: List of variable names of to-be-rescaled features.
    :param verbose: Detail level of output print. 0 means no output print, 1 returns mins and scale weights.

    :return: Rescaled dataframe.
    """
    # Load attributes
    with open(f'models//attributes//{str_model}_min_col.pkl', 'rb') as file:
        min_col = pickle.load(file)
    with open(f'models//attributes//{str_model}_scale_factor.pkl', 'rb') as file:
        scale_factor = pickle.load(file)

    if verbose == 1:
        print(f"Minima per variable: {min_col}")
        print(f"Scale Factors per variable: {scale_factor}")

    # Rescale dataframe
    if len(attributes) <= 1:
        df_rescaled = df_scaled.copy() * scale_factor[attributes].values[0] + min_col[attributes].values[0]
    else:
        df_rescaled = df_scaled.copy() * scale_factor[attributes] + min_col[attributes]

    return df_rescaled


# Split training / test set
def split_dataframe(df_features, target_var = None, productive:bool=False, train_share: float = 0.7, shuffle:bool=False,
                    **kwargs):
    """
    Separate target variable from feature matrix and perform train-test-split.

    Parameters:
    :param df_features: (pd.DataFrame) Feature dataframe containing both target variable and features.
    :param target_var: (str, list) Single name or list of names of the target variable(s), used for column indexing.
    :param productive: (bool) Indicator whether method to be used in productive environment. Default is False. If true,
                       features and labels will be separated without creating train and test sets. If false, function
                       returns x_train, y_train, x_test, y_test.
    :param train_share: (float) Share of training data from total dataset.
    :param shuffle: (bool) Indicator whether to shuffle observations before making the split. Default is False.
    :param kwargs: Input parameters for sklearn train_test_split function, e.g. shuffle.

    Returns:
    :return: x_train, x_test, y_train, y_test -> Dataframes with corresponding train / test data.
    """
    # Input Check
    if isinstance(target_var, str) or isinstance(target_var, list):
        features = df_features.drop(columns=target_var)
        label = df_features[target_var]
    else:
        raise ValueError("Type of target_var must be either str or list. Change input accordingly.")

    # Remove Target Variable & Split
    if not productive:
        x_train, x_test, y_train, y_test = train_test_split(features,
                                                            label,
                                                            train_size=train_share,
                                                            shuffle=shuffle,
                                                            **kwargs
                                                            )

    else: # In productive environment
        x_train = features
        y_train = label
        x_test = None
        y_test = None

    return x_train, x_test, y_train, y_test


def dailydf_to_ts(daily_df, header:str='value'):
    """
    Concatenate rows of dataframe into single-column format to generate pd.Series-like structure.

    :param daily_df: (pd.DataFrame) Dataframe with row-wise observation entries and datetime index,
                     e.g. all observations of the same day in the same row.
    :param header: (str) Name of the newly created dataframe column.

    :return: (pd.Series) Series of concatenated sequences with datetime index.
    """
    # Extract values
    values = []
    # for row in range(len(daily_df)):
    #     for column in range(len(daily_df.columns)):
    #         values.append(daily_df.iloc[row, column])

    values = daily_df.stack().tolist()

    # Set new date index
    date_index = pd.date_range(start=daily_df.index[0],
                               periods=len(values),
                               tz='Etc/GMT-1',
                               freq=f'{24 // len(daily_df.columns)}h')

    daily_ts = pd.Series(data=values, index=date_index)
    daily_ts.name = header

    return daily_ts


def write_DWH(str_path:str, str_tsname:str, str_property:str, str_unit:str,
              df_timeseries:pd.DataFrame, filename=None):
    """
    Write time series to Belvis Datawarehouse via CSV Export.

    Parameters:
    :param str_path:
    :param str_tsname:
    :param str_property:
    :param str_unit:
    :param df_timeseries:
    :param filename:

    Returns:
    :return:
    """
    bool_copy = False

    if str_path.find('EPAG_PFM') >= 1 and str_path.find('srvedm11') >= 1:
        str_path_old = str_path.replace('srvedm11', 'srvedm11')
        bool_copy = True

    # Winterzeit!

    col_name = df_timeseries.columns[0]
    if filename is None:
        datestr_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        if str_property is None:
            filename = f'{datestr_now}_{str_tsname}_DWH.csv'
        else:
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
                                 f"{df_timeseries[col_name].iloc[i_row]}", "-", ""]
                                )

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
                                     f"{df_timeseries[col_name].iloc[i_row]}", "-", ""]
                                    )

    idx_success = True
    return idx_success







