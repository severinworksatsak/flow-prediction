from models.utility import load_input, scale_with_minmax, get_dates_from_config, handle_outliers, \
    get_params_from_config, dailydf_to_ts, inverse_transform_minmax, transform_dayofyear
from models.lstm import simpleLSTM
from models.deeplearner import DeepLearner
from models.svr import SVReg


def prepare_inputs(str_model:str, idx_train:bool):

    # Get Parameters from Config
    dates = get_dates_from_config(str_model)
    doy_flag = get_params_from_config(function='get_doyflag', str_model=str_model)['doy_flag']

    # Load Input Data
    df_variables = load_input(str_model=str_model, **dates)

    # Day of Year Transformation (if commanded)
    if doy_flag:
        df_variables = transform_dayofyear(df_variables)

    # Outlier Detection & Clearing
    df_handled = handle_outliers(df_variables)

    # Scaling with MinMax (idx_train vs. predict)
    df_scaled = scale_with_minmax(df_features=df_handled, str_model=str_model, idx_train=idx_train, verbose=0)

    return df_scaled


def prepare_inputs_lstm(str_model:str, idx_train:bool):

    # Instantiate Objects
    lstm = simpleLSTM()

    # Get config parameters
    sequence_params = get_params_from_config(function='lstm_sequence', str_model=str_model)
    target_var = get_params_from_config(function='get_label', str_model=str_model)['label']

    # Preparation up to Scaling
    df_scaled = prepare_inputs(str_model=str_model, idx_train=idx_train)

    # Generate Sequences
    n_lookback = sequence_params['n_lookback']
    n_ahead = sequence_params['n_ahead']
    n_offset = sequence_params['n_offset']

    '''
    Test size is set to zero since i) training process no longer needs testing data, instead relying on validation 
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


    # Split dataframe (Train, validation set)
    pass