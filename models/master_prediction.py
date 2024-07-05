from models.utility import load_input, scale_with_minmax, get_dates_from_config, handle_outliers, \
    get_params_from_config, dailydf_to_ts, inverse_transform_minmax, transform_dayofyear
from models.lstm import simpleLSTM
from models.deeplearner import DeepLearner
from models.svr import SVReg

from keras.models import save_model, load_model
import pickle
import pandas as pd
from datetime import datetime, timedelta


def prepare_inputs(str_model:str, idx_train:bool, date_dict:dict=None):

    # Get Parameters from Config
    doy_flag = get_params_from_config(function='get_doyflag', str_model=str_model)['doy_flag']
    if date_dict is None:
        date_dict = get_dates_from_config(str_model, training=idx_train)
    dates = date_dict

    # Load Input Data
    df_variables = load_input(str_model=str_model, idx_train=idx_train, **dates)

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

    # Instantiate Objects
    lstm = simpleLSTM()

    # Get config parameters
    sequence_params = get_params_from_config(function='lstm_sequence', str_model=str_model)
    target_var = get_params_from_config(function='get_label', str_model=str_model)['label']

    # Preparation up to Scaling
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


def prepare_inputs_svr(str_model:str, idx_train:bool):

    # Get config parameters
    target_var = get_params_from_config(function='get_label', str_model=str_model)['label']
    n_offset = get_params_from_config(function='build_svr', str_model=str_model)['n_offset']
    svr_hyperparams = get_params_from_config(function='build_svr', str_model=str_model)['hyperparameters']

    # Preparation up to Scaling
    df_scaled = prepare_inputs(str_model=str_model, idx_train=idx_train)

    # Build SVR Model Input
    svr = SVReg()
    df_label, model_names = svr.build_model_input(df=df_scaled,
                                                  target_var=target_var,
                                                  str_model=str_model,
                                                  n_offset=n_offset)


    return df_label, model_names


# Train each Model -----------------------------------------------------------------------------------------------------

def train_lstm(str_model:str, idx_train:bool=True):
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
    save_model(trained_lstm[0], f'models//attributes//model_{str_model}.keras')
    with open(f'models//attributes//history_{str_model}.pkl', 'wb') as file:
        pickle.dump(trained_lstm[1].history, file)
    with open(f'models//attributes//params_{str_model}.pkl', 'wb') as file:
        pickle.dump(trained_lstm[1].params, file)


def train_svr():
    pass


# Predict each model ---------------------------------------------------------------------------------------------------

def predict_lstm(str_model:str, idx_train:bool=False):

    # Instantiate Objects
    lstm = simpleLSTM()

    # Get config parameters
    label_name = get_params_from_config(function='get_label', str_model=str_model)['label']
    n_timestep = get_params_from_config(function='get_n_timestep', str_model=str_model)['n_timestep']
    sequence_params = get_params_from_config(function='lstm_sequence', str_model=str_model)
    n_lookback = sequence_params['n_lookback']
    n_offset = sequence_params['n_offset']

    # Get modified dates -> elongate x_test so that first prediction will be for first_calc_day
    dates = get_dates_from_config(str_model=str_model, training=idx_train)
    dates['date_from'] = dates['date_from'] - timedelta(hours=(n_lookback + n_offset) * (24 // n_timestep))
    print(f"Date_from: {dates['date_from']}, Date_to: {dates['date_to']}")

    # Load prediction parameters & model
    model = load_model(f'models//attributes//model_{str_model}.keras')

    # Load prediction input
    df_seq = prepare_inputs_lstm(str_model=str_model, idx_train=idx_train, date_dict=dates)
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
    return ts_ypred

    # Write predictions to DWH


def predict_lstm_daywise(str_model:str, idx_train:bool=False):

    # Instantiate Objects
    lstm = simpleLSTM()

    # Get config parameters
    label_name = get_params_from_config(function='get_label', str_model=str_model)['label']
    labelmean_name = get_params_from_config(function='get_labelmean', str_model=str_model)
    n_timestep = get_params_from_config(function='get_n_timestep', str_model=str_model)['n_timestep']
    sequence_params = get_params_from_config(function='lstm_sequence', str_model=str_model)
    n_lookback = sequence_params['n_lookback']
    n_offset = sequence_params['n_offset']

    # Get modified dates -> elongate x_test so that first prediction will be for first_calc_day
    dates = get_dates_from_config(str_model=str_model, training=idx_train)
    pred_dayrange = range((dates['date_to'] - dates['date_from']).days)
    pred_startdate = dates['date_from']

    # Load prediction parameters & model
    model = load_model(f'models//attributes//model_{str_model}.keras')

    # Per-Day Predictions and calculation of daily average
    daymean_replace = None
    ypred_series = pd.Series(dtype=float)

    for pred_day in pred_dayrange:
        # Get daily slice
        dates['date_to'] = pred_startdate + timedelta(days=pred_day)
        dates['date_from'] = dates['date_from'] - timedelta(
            hours=(n_lookback + n_offset) * (24 // n_timestep)) + timedelta(days=pred_day)
        print(f"Date_from: {dates['date_from']}, Date_to: {dates['date_to']}")


        # Load prediction input
        df_seq = prepare_inputs_lstm(str_model=str_model,
                                     idx_train=idx_train,
                                     date_dict=dates,
                                     replace_data=daymean_replace
                                     )
        x_pred = df_seq[0]
        lstm.ytest_startdate = df_seq[4]

        # Predict with LSTM
        y_pred = model.predict(x_pred)

        # Calculate day means of label and collect values to replace old daymeans in prepare_inputs_lstm
        df_ymean = lstm.convert_seq_to_df(seq_array=y_pred, n_timestep=None, start_date=None)
        ts_ymean = dailydf_to_ts(df_ymean)
        df_ymean[labelmean_name] = [ts_ymean.mean() for i in range(len(ts_ymean))]
        daymean_replace = df_ymean[labelmean_name]

        # Rescale predictions
        y_pred_rescaled = inverse_transform_minmax(df_scaled=y_pred, str_model=str_model, attributes=[label_name])

        # Convert predictions back to ts
        df_ypred = lstm.convert_seq_to_df(seq_array=y_pred_rescaled, n_timestep=None, start_date=None)
        ts_ypred = dailydf_to_ts(df_ypred)
        ypred_series = ypred_series.add(ts_ypred)

    # Resample & interpolate predictions
    return ypred_series

    # Write predictions to DWH
