from models.utility import load_input, scale_with_minmax, get_dates_from_config, handle_outliers, \
    get_params_from_config, dailydf_to_ts, inverse_transform_minmax, transform_dayofyear, \
    split_dataframe, write_DWH
from models.lstm import simpleLSTM
from models.deeplearner import DeepLearner
from models.svr import SVReg
from models.solutil_func import get_timeseries_1h

from sklearn.ensemble import RandomForestRegressor
from keras.models import save_model, load_model
import pickle
import pandas as pd
from datetime import datetime, timedelta
import os
from pytz import timezone


def prepare_inputs(str_model:str, idx_train:bool, date_dict:dict=None):
    """
    Base function for model dispatch. Retrieves feature and labels in one data array as
    specified in the 'inputs' section of the config file.

    Parameters:
    :param str_model: (str) Model name as specified under 'models' section in json config.
    :param idx_train: (bool) Boolean indicator for training / prediction mode. Idx_train affects
                      various nested methods:
                      - get_dates_from_config(): If idx_train = True, the date range between
                        first_ & last_day_calc is returned and the training window otherwise.
                      - load_inputs(): If idx_train = True, the label timeseries will be omitted
                        from loading, otherwise the label is loaded to facilitate benchmarking.
                      - scale_with_minmax(): If idx_train = True, scaling factors are calculated and
                        newly saved. If false, the factors are merely reloaded from the existing pkl file.
    :param date_dict: (dict) Dictionary containing 'date_from' and 'date_to' variables either as string or
                      datetime object. Default is None, in which case the dates are retrieved from the config.

    Returns:
    :return df_scaled: (df) Dataframe with sanitized and scaled features and labels.
    """
    # Get Parameters from Config
    doy_flag = get_params_from_config(function='get_doyflag', str_model=str_model)['doy_flag']
    if date_dict is None:
        date_dict = get_dates_from_config(str_model, training=idx_train)

    # Load Input Data
    df_variables = load_input(str_model=str_model, idx_train=idx_train, **date_dict)

    # Day of Year Transformation (if commanded)
    if doy_flag:
        df_variables = transform_dayofyear(df_variables)

    # Outlier Detection & Clearing
    df_handled = handle_outliers(df_variables)

    # Scaling with MinMax (idx_train vs. predict)
    df_scaled = scale_with_minmax(df_features=df_handled, str_model=str_model, idx_train=idx_train, verbose=0)

    return df_scaled


# Prepare Inputs for each Model ----------------------------------------------------------------------------------------

def prepare_inputs_lstm(str_model:str, idx_train:bool, date_dict:dict=None, replace_data=None):
    """
    Prepare inputs for LSTM models by retrieving features / labels, sanitizing and scaling, generating sequences,
    and splitting into train/test & feature/label frames. Moreover, original input data can be replaced by passing a
    dict as argument.

    Parameters:
    :param str_model: (str) Model name as specified under 'models' section in json config.
    :param idx_train: (bool) Boolean indicator for training / prediction mode. Idx_train affects
                      various nested methods:
                      - get_dates_from_config(): If idx_train = True, the date range between
                        first_ & last_day_calc is returned and the training window otherwise.
                      - load_inputs(): If idx_train = True, the label timeseries will be omitted
                        from loading, otherwise the label is loaded to facilitate benchmarking.
                      - scale_with_minmax(): If idx_train = True, scaling factors are calculated and
                        newly saved. If false, the factors are merely reloaded from the existing pkl file.
    :param date_dict: (dict) Dictionary containing 'date_from' and 'date_to' variables either as string or
                      datetime object. Default is None, in which case the dates are retrieved from the config.
    :param replace_data: (pd.Series / pd.DataFrame / dict) Iterable object containing replacement subsets, with which
                         existing data can be overridden, e.g. to iteratively insert 1d-mean values for multi-day
                         predictions.

    Returns:
    :return: (array) Sequence arrays x_train, x_test, y_train, y_test. If in productive version, x_test and y_test will
             be empty.
    """
    # Instantiate Objects
    lstm = simpleLSTM()

    # Get config parameters
    sequence_params = get_params_from_config(function='lstm_sequence', str_model=str_model)
    target_var = get_params_from_config(function='get_label', str_model=str_model)['label']
    n_timestep = get_params_from_config(function='get_n_timestep', str_model=str_model)['n_timestep']
    n_lookback = sequence_params['n_lookback']
    n_offset = sequence_params['n_offset']

    # Preparation up to Scaling
    if date_dict is None:
        date_dict = get_dates_from_config(str_model, training=idx_train)
        date_dict['date_from'] = date_dict['date_from'] - timedelta(hours=(n_lookback + n_offset) * (24 // n_timestep))

    df_scaled = prepare_inputs(str_model=str_model, idx_train=idx_train, date_dict=date_dict)

    if replace_data is not None:
        df_scaled.update(replace_data)

    # Generate Sequences
    n_lookback = sequence_params['n_lookback']
    n_ahead = sequence_params['n_ahead']
    n_offset = sequence_params['n_offset']

    '''
    Test data size is set to zero since i) training process no longer needs testing data, instead relying on validation 
    sets directly sliced in the DeepLearner module, and ii) holistic usage of prepared inputs for prediction purposes.
    '''
    df_seq = lstm.generate_sequences(df=df_scaled,
                                     target_var=target_var,
                                     n_lookback=n_lookback,
                                     n_ahead=n_ahead,
                                     n_offset=n_offset,
                                     continuous=False,
                                     productive=True
                                     )

    return df_seq


def prepare_inputs_svr(str_model:str, idx_train:bool, date_dict:dict=None, replace_data=None):
    """
    Prepare inputs for SVR models by retrieving features / labels, sanitizing and scaling, and splitting into
    train/test & feature/label frames. Important: As SVRs are single-output models, labels for n different models have
    to be created.

    Parameters:
    :param str_model: (str) Model name as specified under 'models' section in json config.
    :param idx_train: (bool) Boolean indicator for training / prediction mode. Idx_train affects
                      various nested methods:
                      - get_dates_from_config(): If idx_train = True, the date range between
                        first_ & last_day_calc is returned and the training window otherwise.
                      - load_inputs(): If idx_train = True, the label timeseries will be omitted
                        from loading, otherwise the label is loaded to facilitate benchmarking.
                      - scale_with_minmax(): If idx_train = True, scaling factors are calculated and
                        newly saved. If false, the factors are merely reloaded from the existing pkl file.
    :param date_dict: (dict) Dictionary containing 'date_from' and 'date_to' variables either as string or
                      datetime object. Default is None, in which case the dates are retrieved from the config.
    :param replace_data: (pd.Series / pd.DataFrame / dict) Iterable object containing replacement subsets, with which
                         existing data can be overridden, e.g. to iteratively insert 1d-mean values for multi-day
                         predictions.

    Returns:
    :return df_label: (df) Dataframe containing features and lagged labels, with lags ranging from 1 to n_timestep,
                      representing 1- to n-ahead predictions.
    :return model_names: (list) Names of the n models and labels - their naming is equivalent for simplified reference
                         in subsequent methods.
    """
    # Get config parameters
    target_var = get_params_from_config(function='get_label', str_model=str_model)['label']
    n_offset = get_params_from_config(function='build_svr', str_model=str_model)['n_offset']

    # Preparation up to Scaling
    df_scaled = prepare_inputs(str_model=str_model, idx_train=idx_train, date_dict=date_dict)

    if replace_data is not None:
        df_scaled.update(replace_data)

    # Build SVR Model Input
    svr = SVReg()
    df_label, model_names = svr.build_model_input(df=df_scaled,
                                                  target_var=target_var,
                                                  str_model=str_model,
                                                  n_offset=n_offset,
                                                  idx_train=idx_train
                                                  )

    return df_label, model_names



# Train each Model -----------------------------------------------------------------------------------------------------

def train_lstm(str_model:str, idx_train:bool=True):
    """
    Train LSTM model based on hyperparameters specified in config file. This method prepares inputs, creates
    and parameterizes the model instance and trains it using x_train and y_train.

    Parameters:
    :param str_model: (str) Model name as specified under 'models' section in json config.
    :param idx_train: (bool) Boolean indicator for training / prediction mode. Idx_train affects
                      various nested methods:
                      - get_dates_from_config(): If idx_train = True, the date range between
                        first_ & last_day_calc is returned and the training window otherwise.
                      - load_inputs(): If idx_train = True, the label timeseries will be omitted
                        from loading, otherwise the label is loaded to facilitate benchmarking.
                      - scale_with_minmax(): If idx_train = True, scaling factors are calculated and
                        newly saved. If false, the factors are merely reloaded from the existing pkl file.

    Returns:
    :return trained_model: (pkl) Trained model saved as pkl file under models//attributes.
    :return history: (pkl) Training history saved as pkl file under models//attributes.
    """
    # Instantiate Objects
    lstm = simpleLSTM()
    deepl = DeepLearner()

    # Prepare inputs
    df_seq = prepare_inputs_lstm(str_model=str_model, idx_train=idx_train)
    x_train = df_seq[0]
    y_train = df_seq[2]

    # Load hyperparameters
    train_params = get_params_from_config(function='model_train', str_model=str_model)
    param_dict = get_params_from_config(function='build_lstm', str_model=str_model)
    n_valid = param_dict['n_valid']
    lstm_hyperparams = param_dict['hyperparameters']

    val_share = n_valid / x_train.shape[0]

    # Build model
    lstm_2l = lstm.build_2layer_lstm(x_train, y_train, **lstm_hyperparams)

    # Train model
    trained_lstm = deepl.train_model(x_train=x_train,
                                     y_train=y_train,
                                     model=lstm_2l,
                                     val_share=val_share,
                                     **train_params
                                     )

    # Save model and training history / params
    save_model(trained_lstm[0],f'models//attributes//{str_model}_trained_model.keras')
    with open(f'models//attributes//{str_model}_history.pkl', 'wb') as file:
        pickle.dump(trained_lstm[1].history, file)
    with open(f'models//attributes//{str_model}_params.pkl', 'wb') as file:
        pickle.dump(trained_lstm[1].params, file)


def train_svr(str_model:str, idx_train:bool=True):
    """
        Train SVR model based on hyperparameters specified in config file. This method prepares inputs, creates
        and parameterizes the model instances and trains them using x_train and y_train.

        Parameters:
        :param str_model: (str) Model name as specified under 'models' section in json config.
        :param idx_train: (bool) Boolean indicator for training / prediction mode. Idx_train affects
                          various nested methods:
                          - get_dates_from_config(): If idx_train = True, the date range between
                            first_ & last_day_calc is returned and the training window otherwise.
                          - load_inputs(): If idx_train = True, the label timeseries will be omitted
                            from loading, otherwise the label is loaded to facilitate benchmarking.
                          - scale_with_minmax(): If idx_train = True, scaling factors are calculated and
                            newly saved. If false, the factors are merely reloaded from the existing pkl file.

        Returns:
        :return trained_models: (dict) Dictionary of n_timestep rained models saved as pkl file
                                under models//attributes.
        """
    # Get SVR inputs
    df_label, model_names = prepare_inputs_svr(str_model=str_model, idx_train=idx_train)

    # Split features and labels
    x_train = split_dataframe(df_features=df_label, target_var=model_names, productive=True)[0]
    y_train = split_dataframe(df_features=df_label, target_var=model_names, productive=True)[2]

    # Generate and Parameterize Models
    svr = SVReg()
    models = svr.build_svr(str_model='inlet1_svr')

    # Train models
    trained_models = svr.train_svr(svr_dict=models, x_train=x_train, y_train=y_train)

    # Save models as pickle files
    with open(f'models//attributes//{str_model}_trained_dict.pkl', 'wb') as file:
        pickle.dump(trained_models, file)
    with open(f'models//attributes//{str_model}_feature_order.pkl', 'wb') as file:
        pickle.dump(x_train.columns, file) # Necessary because features in prediction need to be in same order

    return trained_models


def train_ensemble(str_model:str, idx_train:bool, date_dict:dict=None):

    # Get Config Parameters
    target_var = get_params_from_config(function='get_label', str_model=str_model)['label']
    hyperparameters = get_params_from_config(function='build_ensemble', str_model=str_model)['hyperparameters']

    # Prepare Inputs
    df_scaled = prepare_inputs(str_model=str_model, idx_train=idx_train, date_dict=date_dict)

    # Split Dataframe
    x_train = split_dataframe(df_features=df_scaled, target_var=target_var, productive=True)[0]
    y_train = split_dataframe(df_features=df_scaled, target_var=target_var, productive=True)[2]

    # Build Random Forest Regressor
    rfr = RandomForestRegressor(**hyperparameters)

    # Train Random Forest Regressor
    trained_rfr = rfr.fit(x_train, y_train)

    # Save model & feature order
    with open(f'models//attributes//{str_model}_trained_model.pkl', 'wb') as file:
        pickle.dump(trained_rfr, file)
    with open(f'models//attributes//{str_model}_feature_order.pkl', 'wb') as file:
        pickle.dump(x_train.columns, file) # Necessary because features in prediction need to be in same order

    return trained_rfr



# Predict each model ---------------------------------------------------------------------------------------------------

def predict_lstm(str_model:str, idx_train:bool=False, date_dict:dict=None, writeDWH:bool=False, replace_data=None):
    """
    Make predictions with pre-trained LSTM model, whose weights have been restored from pkl file.
    Method combines input preparation, data sanitation, sequence generation of x_test, and prediction
    for the time window specified in the config. Output is then written to data warehouse.

    Parameters:
    :param str_model: (str) Model name as specified under 'models' section in json config.
    :param idx_train: (bool) Boolean indicator for training / prediction mode. Idx_train affects
                      various nested methods:
                      - get_dates_from_config(): If idx_train = True, the date range between
                        first_ & last_day_calc is returned and the training window otherwise.
                      - load_inputs(): If idx_train = True, the label timeseries will be omitted
                        from loading, otherwise the label is loaded to facilitate benchmarking.
                      - scale_with_minmax(): If idx_train = True, scaling factors are calculated and
                        newly saved. If false, the factors are merely reloaded from the existing pkl file.
    :param date_dict: (dict) Dictionary containing 'date_from' and 'date_to' variables either as string or
                      datetime object. Default is None, in which case the dates are retrieved from the config.
    :param writeDWH: (bool) Boolean indicator for export to Data Warehouse. Defaults to False. If True, CSV file is
                     generated and uploaded to Belvis import folder.
    :param replace_data: (pd.Series / pd.DataFrame / dict) Iterable object containing replacement subsets, with which
                         existing data can be overridden, e.g. to iteratively insert 1d-mean values for multi-day
                         predictions.

    Returns:
    :return ts_ypred: (Series) Series of forecasted values.
    :return writeDWH: (csv) CSV export to data warehouse for merging into specified Belvis time series.
    """
    # Instantiate Objects
    lstm = simpleLSTM()

    # Get config parameters
    label_name = get_params_from_config(function='get_label', str_model=str_model)['label']
    n_timestep = get_params_from_config(function='get_n_timestep', str_model=str_model)['n_timestep']
    sequence_params = get_params_from_config(function='lstm_sequence', str_model=str_model)
    n_lookback = sequence_params['n_lookback']
    n_offset = sequence_params['n_offset']

    # Get modified dates -> elongate x_test so that first prediction will be for first_calc_day
    if date_dict is None:
        date_dict = get_dates_from_config(str_model=str_model, training=idx_train)

    date_dict['date_from'] = date_dict['date_from'] - timedelta(hours=(n_lookback + n_offset) * (24 // n_timestep))

    # Load prediction parameters & model
    model = load_model(f'models//attributes//{str_model}_trained_model.keras')

    # Load prediction input
    df_seq = prepare_inputs_lstm(str_model=str_model, idx_train=idx_train, date_dict=date_dict,
                                 replace_data=replace_data)
    x_pred = df_seq[0]
    lstm.ytest_startdate = df_seq[4]

    # Predict with LSTM
    y_pred = model.predict(x_pred)

    # Rescale predictions
    y_pred_rescaled = inverse_transform_minmax(df_scaled=y_pred, str_model=str_model, attributes=[label_name])

    # Convert back to ts
    df_ypred = lstm.convert_seq_to_df(seq_array=y_pred_rescaled, n_timestep=None, start_date=None)
    ts_ypred = dailydf_to_ts(df_ypred)

    # Resample & interpolate predictions
    if writeDWH:
        # Resample to 1h
        ts_ypred_1h = ts_ypred.resample('1h').interpolate()
        df_ypred_1h = ts_ypred_1h.to_frame()

        # Write normal predictions to DWH
        inlet_n = str_model.split('_')[0].capitalize()
        str_algo = str_model.split('_')[1].upper()
        str_pred = 'Prediction'
        ts_name = '_'.join([inlet_n, str_pred, str_algo])

        write_DWH(str_path=os.path.join(r'\\srvedm11', 'Import', 'Messdaten', 'EPAG_Energie', 'DWH_EX_60'),
                  str_tsname=ts_name,
                  str_property='Python',
                  str_unit='m/3',
                  df_timeseries=df_ypred_1h
                  )

        # D+1 Prediction to DWH
        d_next = (timezone('CET').localize(datetime.now()) + timedelta(days=1)) \
            .replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone('Etc/GMT-1'))

        write_DWH(str_path=os.path.join(r'\\srvedm11', 'Import', 'Messdaten', 'EPAG_Energie', 'DWH_EX_60'),
                  str_tsname=ts_name,
                  str_property='Python_d+1',
                  str_unit='m3/s',
                  df_timeseries=df_ypred_1h.loc[df_ypred_1h.index >= d_next]
                  )

    return ts_ypred


def predict_lstm_daywise(str_model:str, idx_train:bool=False, writeDWH:bool=False):

    # Instantiate Objects
    lstm = simpleLSTM()

    # Get config parameters
    label_name = get_params_from_config(function='get_label', str_model=str_model)['label']
    labelmean_name = get_params_from_config(function='get_labelmean', str_model=str_model)['meanvar_name']
    n_timestep = get_params_from_config(function='get_n_timestep', str_model=str_model)['n_timestep']
    sequence_params = get_params_from_config(function='lstm_sequence', str_model=str_model)
    n_lookback = sequence_params['n_lookback']
    n_offset = sequence_params['n_offset']

    # Get modified dates -> elongate x_test so that first prediction will be for first_calc_day
    dates = get_dates_from_config(str_model=str_model, training=idx_train)
    pred_dayrange = range((dates['date_to'] - dates['date_from']).days)
    pred_startdate = dates['date_from']

    # Load prediction parameters & model
    model = load_model(f'models//attributes//{str_model}_trained_model.keras')

    # Per-Day Predictions and calculation of daily average
    daymean_replace = None
    ypred_series = pd.Series()
    series_list = []

    for pred_day in pred_dayrange:

        # Create strawman date dict -> evade circular updating of date ranges
        mod_dates = {}

        # Get daily slice
        mod_dates['date_to'] = pred_startdate + timedelta(days=pred_day + 1)
        mod_dates['date_from'] = dates['date_from'] - timedelta(
            hours=(n_lookback + n_offset) * (24 // n_timestep)) + timedelta(days=pred_day)

        # Load prediction input
        df_seq = prepare_inputs_lstm(str_model=str_model,
                                     idx_train=idx_train,
                                     date_dict=mod_dates,
                                     replace_data=daymean_replace
                                     )
        x_pred = df_seq[0]
        lstm.ytest_startdate = df_seq[4]

        # Predict with LSTM
        y_pred = model.predict(x_pred)

        # Calculate day means of label and collect values to replace old daymeans in prepare_inputs_lstm
        df_ymean = lstm.convert_seq_to_df(seq_array=y_pred, n_timestep=None, start_date=None)
        ts_ymean = dailydf_to_ts(df_ymean)
        ts_ymean[labelmean_name] = [ts_ymean.mean() for i in range(len(ts_ymean))]
        daymean_replace = ts_ymean[labelmean_name]

        # Rescale predictions
        y_pred_rescaled = inverse_transform_minmax(df_scaled=y_pred, str_model=str_model, attributes=[label_name])

        # Convert predictions back to ts
        df_ypred = lstm.convert_seq_to_df(seq_array=y_pred_rescaled, n_timestep=None, start_date=None)
        ts_ypred = dailydf_to_ts(df_ypred)
        series_list.append(ts_ypred)

    ypred_series = pd.concat(series_list)

    # Write predictions to DWH
    if writeDWH:
        # Resample to 1h
        ts_ypred_1h = ypred_series.resample('1h').interpolate()
        df_ypred_1h = ts_ypred_1h.to_frame()

        # Write normal predictions to DWH
        inlet_n = str_model.split('_')[0].capitalize()
        str_algo = str_model.split('_')[1].upper()
        str_pred = 'Prediction'
        ts_name = '_'.join([inlet_n, str_pred, str_algo])

        write_DWH(str_path=os.path.join(r'\\srvedm11', 'Import', 'Messdaten', 'EPAG_Energie', 'DWH_EX_60'),
                  str_tsname=ts_name,
                  str_property='Python',
                  str_unit='m/3',
                  df_timeseries=df_ypred_1h
                  )

        # D+1 Prediction to DWH
        d_next = (timezone('CET').localize(datetime.now()) + timedelta(days=1)) \
            .replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone('Etc/GMT-1'))
        write_DWH(str_path=os.path.join(r'\\srvedm11', 'Import', 'Messdaten', 'EPAG_Energie', 'DWH_EX_60'),
                  str_tsname=ts_name,
                  str_property='Python_d+1',
                  str_unit='m3/s',
                  df_timeseries=df_ypred_1h.loc[df_ypred_1h.index >= d_next]
                  )

        return ypred_series


def forecast_svr(str_model:str, idx_train:bool=False, date_dict:dict=None, writeDWH:bool=False,
                 replace_data:dict=None): # different naming due to svr.predict_svr

    # Get Config Parameters
    label_name = get_params_from_config(function='get_label', str_model=str_model)['label']

    # Get SVR inputs
    x_pred, model_names = prepare_inputs_svr(str_model=str_model, idx_train=idx_train, date_dict=date_dict,
                                             replace_data=replace_data)

    # Load models & feature order
    with open(f'models//attributes//{str_model}_trained_dict.pkl', 'rb') as file:
        trained_models = pickle.load(file)

    with open(f'models//attributes//{str_model}_feature_order.pkl', 'rb') as file:
        feature_order = pickle.load(file)

    date_index = x_pred.index

    # Ensure same column order as in training -> requirement of sklearn SVR object
    x_pred_reorder = x_pred.reindex(columns=feature_order)

    # Instantiate SVR object
    svr = SVReg()
    y_pred = svr.predict_svr(trained_svr=trained_models, x_test=x_pred_reorder, str_model=str_model)

    # Rescale Predictions -> with rescale = False, the input for ensemble prediction is generated
    y_pred_rescaled = inverse_transform_minmax(df_scaled=y_pred, str_model=str_model, attributes=[label_name])

    y_pred_rescaled.index = date_index

    # Write prediction to DWH
    if writeDWH:
        # Resample to 1h
        ts_ypred_1h = y_pred_rescaled.resample('1h').interpolate()
        df_ypred_1h = ts_ypred_1h.to_frame()

        # Write normal predictions to DWH
        inlet_n = str_model.split('_')[0].capitalize()
        str_algo = str_model.split('_')[1].upper()
        str_pred = 'Prediction'
        ts_name = '_'.join([inlet_n, str_pred, str_algo])

        write_DWH(str_path=os.path.join(r'\\srvedm11', 'Import', 'Messdaten', 'EPAG_Energie', 'DWH_EX_60'),
                  str_tsname=ts_name,
                  str_property='Python',
                  str_unit='m/3',
                  df_timeseries=df_ypred_1h
                  )

        # Write d+1 predictions
        d_next = (timezone('CET').localize(datetime.now()) + timedelta(days=1)) \
            .replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone('Etc/GMT-1'))

        write_DWH(str_path=os.path.join(r'\\srvedm11', 'Import', 'Messdaten', 'EPAG_Energie', 'DWH_EX_60'),
                  str_tsname=ts_name,
                  str_property='Python_d+1',
                  str_unit='m3/s',
                  df_timeseries=df_ypred_1h.loc[df_ypred_1h.index >= d_next]
                  )

    return y_pred_rescaled


def predict_ensemble(str_model:str, idx_train=False, date_dict:dict=None, writeDWH:bool=False):

    # Load model & feature_order
    with open(f'models//attributes//{str_model}_trained_model.pkl', 'rb') as file:
        trained_model = pickle.load(file)
    with open(f'models//attributes//{str_model}_feature_order.pkl', 'rb') as file:
        feature_order = pickle.load(file)

    # Get Parameters from Config
    target_var = get_params_from_config(function='get_label', str_model=str_model)['label']

    # Prepare inputs
    df_scaled = prepare_inputs(str_model=str_model, idx_train=idx_train, date_dict=date_dict)
    date_index = df_scaled.index
    x_pred = split_dataframe(df_features=df_scaled, target_var=target_var, productive=True)[0]

    # Ensure same column order as in training -> requirement of sklearn RFR object
    x_pred_reorder = x_pred.reindex(columns=feature_order)

    # Predict Ensemble
    y_pred = trained_model.predict(x_pred_reorder)

    # Rescale Predictions
    y_pred_rescaled = inverse_transform_minmax(df_scaled=y_pred, str_model=str_model, attributes=[target_var])
    ts_ypred = pd.Series(y_pred_rescaled, index=date_index, name='pred_ensemble')

    # if writeDWH:
    #     # Resample to 1h
    #     ts_ypred_1h = ts_ypred.resample('1h').interpolate()
    #     df_ypred_1h = ts_ypred_1h.to_frame()
    #
    #     # Write normal predictions to DWH
    #     inlet_n = str_model.split('_')[0].capitalize()
    #     str_algo = str_model.split('_')[1].capitalize()
    #     str_pred = 'Prediction'
    #     ts_name = '_'.join([inlet_n, str_pred, str_algo])
    #
    #     write_DWH(str_path=os.path.join(r'\\srvedm11', 'Import', 'Messdaten', 'EPAG_Energie', 'DWH_EX_60'),
    #               str_tsname=ts_name,
    #               str_property='Python',
    #               str_unit='m/3',
    #               df_timeseries=df_ypred_1h
    #               )
    #
    #     # D+1 Prediction to DWH

    return ts_ypred


def predict_zufluss(str_inlet:str, writeDWH:bool=False):
    """
    Predict ensemble model by iteratively making a forecast for LSTM, SVR and RNN for each prediction day and passing
    these outputs to the ensemble model as inputs. If desired, function directly exports predictions of LSTM, SVR and
    daily prediction means to Belvis Data Warehouse.

    :param str_inlet: (str)
    :param writeDWH: (bool)

    :return: (pd.Series) Ensemble model prediction time series over the time range specified in config.json.
    """

    # Create str_models
    str_lstm = str_inlet + '_lstm'
    str_svr = str_inlet + '_svr'
    str_rnn = str_inlet + '_rnn'
    str_ensemble = str_inlet + '_ensemble'

    # Get config parameters
    dates = get_dates_from_config(str_model=str_ensemble, training=False)
    date_from = dates['date_from']
    date_to = dates['date_to']
    n_timestep = get_params_from_config(function='get_n_timestep', str_model=str_ensemble)['n_timestep']
    x_labels = get_params_from_config(function='get_ensemble_labels', str_model=str_ensemble)
    exog_feat = get_params_from_config(function='get_exogenous_labels', str_model=str_ensemble)
    label_name = get_params_from_config(function='get_label', str_model=str_ensemble)['label']

    # Get pickle inputs
    with open(f'models/attributes/{str_ensemble}_min_col.pkl', 'rb') as file:
        scale_mins = pickle.load(file)
    with open(f'models/attributes/{str_ensemble}_scale_factor.pkl', 'rb') as file:
        scale_factors = pickle.load(file)
    with open(f'models/attributes/{str_ensemble}_feature_order.pkl', 'rb') as file:
        feature_order = pickle.load(file)
    with open(f'models/attributes/{str_ensemble}_trained_model.pkl', 'rb') as file:
        ensemble_model = pickle.load(file)

    # Create strawmen
    date_index = pd.date_range(start=date_from, end=date_to, freq=f'{24 // n_timestep}h', tz=timezone('Etc/GMT-1'),
                               inclusive='left') # inclusive='left'

    y_pred_all = pd.Series(index=date_index)
    y_pred_all_lstm = pd.Series(index=date_index)
    y_pred_all_svr = pd.Series(index=date_index)
    dailymeans_all = pd.Series(index=date_index)

    pred_days = (y_pred_all[::n_timestep].index).tz_localize(None) # Necessary due to load_input not taking tz-aware inputs
    replace_data = None

    # Loop through each prediction day -> iterative prediction due to circular input of daily mean inflow
    for pred_day in pred_days: # removed pred_days[:-1]

        # Define truncated date subsets
        date_from_sub = pred_day
        date_to_sub = date_from_sub + timedelta(days=1)
        date_dict = {'date_from': date_from_sub,
                     'date_to': date_to_sub}

        # Predict LSTM
        y_pred_lstm = predict_lstm(str_model=str_lstm, idx_train=False, date_dict=date_dict, writeDWH=False,
                                   replace_data=replace_data)
        y_pred_lstm = (y_pred_lstm - scale_mins[x_labels['lstm']]) / scale_factors[x_labels['lstm']]
        print(f"Pred day {pred_day}: {y_pred_lstm}")

        # Predict SVR
        y_pred_svr = forecast_svr(str_model=str_svr, idx_train=False, date_dict=date_dict, writeDWH=False)
        y_pred_svr = (y_pred_svr - scale_mins[x_labels['svr']]) / scale_factors[x_labels['svr']]

        # Load RNN
        '''
        Load from Belvis is only required as long as RNN is predicted using legacy infrastructure. Should it be decided
        at some point that the RNN training / prediction is integrated in this new FE infrastructure, the data source
        used below would need to be updated. 
        '''
        y_pred_rnn = load_input(str_model=str_rnn, date_from=date_from_sub, date_to=date_to_sub)['base_lag0']
        y_pred_rnn = (y_pred_rnn - scale_mins[x_labels['rnn']]) / scale_factors[x_labels['rnn']]

        # Concatenate Predictions
        x_pred = pd.DataFrame(data={x_labels['lstm']: y_pred_lstm, x_labels['svr']: y_pred_svr,
                                    x_labels['rnn']: y_pred_rnn})
        x_pred.dropna(inplace=True)

        # Load & Scale Exogenous Features
        df_exog = prepare_inputs(str_model=str_ensemble,
                                 idx_train=False,
                                 date_dict=date_dict
                                 )[exog_feat['exog_labels']]
        df_exog_scaled = ((df_exog - scale_mins) / scale_factors).dropna(axis=1, how='all')
        x_pred[exog_feat['exog_labels']] = df_exog_scaled

        # Predict Ensemble & Calculate Day Mean
        x_pred_reorder = x_pred.reindex(columns=feature_order)
        y_pred = ensemble_model.predict(x_pred_reorder)
        ts_ypred = pd.Series(y_pred, index=x_pred.index, name='ypred_ensemble')

        replace_name = get_params_from_config(function='get_labelmean', str_model='inlet1_lstm')['label_mean'] # base_1d lags should be identical across all models
        replace_data = pd.Series(data=[ts_ypred.mean() for timeofday in ts_ypred.index],
                                 index=ts_ypred.index,
                                 name=replace_name)

        # Add Prediction to Strawman
        y_pred_all.update(ts_ypred)
        y_pred_all_lstm.update(y_pred_lstm)
        y_pred_all_svr.update(y_pred_svr)
        dailymeans_all.update(replace_data)

    # Create Prediction Dictionary
    pred_data = {
        'lstm': y_pred_all_lstm,
        'svr': y_pred_all_svr,
        # 'ensemble': y_pred_all     -> Uncomment once Ensemble Belvis Time series is created
    }

    # Rescale predictions
    y_pred_rescaled = inverse_transform_minmax(df_scaled=y_pred_all, str_model=str_ensemble, attributes=[label_name]) # Delete when y_pred_all included in dict
    dailymeans_all = inverse_transform_minmax(df_scaled=dailymeans_all, str_model=str_ensemble, attributes=[label_name])

    for model in pred_data.keys():
        pred_data[model] = inverse_transform_minmax(df_scaled=pred_data[model],
                                                    str_model=str_ensemble,
                                                    attributes=[label_name]
                                                    )

    # Write DWH
    if writeDWH:
        writeDWH_zufluss(str_inlet=str_inlet, pred_data=pred_data, daymean_data=None) # Change to dailymeans_all after go-live

    return y_pred_rescaled, pred_data, dailymeans_all # Remove pred_data, dailymeans_all after go-live, rewrite y_pred_rescaled to pred_data['ensemble']


def writeDWH_zufluss(str_inlet:str, pred_data:dict, daymean_data:pd.Series=None):
    """
    Iteratively write predictions stemming from predict_zufluss or other functions to Belvis Data Warehouse, including
    daily means of predicted model.

    :param str_inlet: (str) Inlet name, is used for referencing in config.json and as identifier for Belvis import.
    :param pred_data: (dict) Dictionary containing algorithm names and prediction data as key value pairs. For example,
                      such a dict can be structured as follows: {'lstm', ypred_lstm, 'svr': ypred_svr,
                      'ensemble': ypred_ensemble}. Since RNN is currently predicted differently than newer models, it
                      does not need to be included in this dict.
    :param daymean_data: (pd.Series) Pandas series containing daily means of predicted inflow. Default is None, in which
                         case no mean data is written to DWH. This is due to the fact that the daily data time series is
                         a productive series used elsewhere, which motivates parsimonious updating.

    :return: CSV Export / Belvis Import for each predicted model specified in the pred_data dictionary plus, if
             applicable, upload of daily means.
    """

    # Define global function variables
    inlet_n = str_inlet.capitalize()

    # Model Predictions ################################################################################################

    for model in pred_data.keys():
        # Prepare inputs
        df_model_1h = pred_data[model].resample('1h').interpolate().to_frame()
        str_algo = model.upper()
        ts_name = '_'.join([inlet_n, 'Prediction', str_algo])

        # Write LSTM Prediction
        write_DWH(str_path=os.path.join(r'\\srvedm11', 'Import', 'Messdaten', 'EPAG_Energie', 'DWH_EX_60'),
                  str_tsname=ts_name,
                  str_property='Python',
                  str_unit='m/3',
                  df_timeseries=df_model_1h
                  )

        # Write LSTM d+1 Prediction
        d_next = (timezone('CET').localize(datetime.now()) + timedelta(days=1)) \
            .replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone('Etc/GMT-1'))

        write_DWH(str_path=os.path.join(r'\\srvedm11', 'Import', 'Messdaten', 'EPAG_Energie', 'DWH_EX_60'),
                  str_tsname=ts_name,
                  str_property='Python_d+1',
                  str_unit='m3/s',
                  df_timeseries=df_model_1h.loc[df_model_1h.index >= d_next]
                  )

        # Day Means ####################################################################################################

        if daymean_data is not None:
            # Prepare inputs
            df_mean_1h = daymean_data.resample('1h').interpolate().to_frame()
            ts_name = '_'.join([str_inlet, 'Prediction', 'EPAG', 'day'])

            # Write Day Means
            write_DWH(str_path=os.path.join(r'\\srvedm11', 'Import', 'Messdaten', 'EPAG_Energie', 'DWH_EX_60'),
                      str_tsname=ts_name,
                      str_property='Python',
                      str_unit='m/3',
                      df_timeseries=df_mean_1h
                      )
























