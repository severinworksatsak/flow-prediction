from models.utility import load_input, scale_with_minmax, get_dates_from_config, handle_outliers, \
    get_params_from_config, dailydf_to_ts, inverse_transform_minmax, transform_dayofyear, \
    split_dataframe
from models.lstm import simpleLSTM
from models.deeplearner import DeepLearner
from models.svr import SVReg

from keras.models import save_model, load_model
import pickle
import pandas as pd
from datetime import datetime, timedelta


class ModelDispatcher:

    def __init__(self):
        # Initiate variables
        self.svr_feature_order = None









    def prepare_inputs(self, str_model:str, idx_train:bool, date_dict:dict=None):
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
        print(date_dict)

        # Load Input Data
        df_variables = load_input(str_model=str_model, idx_train=idx_train, **date_dict)

        # Day of Year Transformation (if commanded)
        if doy_flag:
            df_variables = transform_dayofyear(df_variables)

        # Outlier Detection & Clearing
        df_handled = handle_outliers(df_variables)

        # Scaling with MinMax (idx_train vs. predict)
        df_scaled = scale_with_minmax(df_features=df_handled, str_model=str_model, idx_train=idx_train, verbose=0)
        print(f"df_scaled: {df_scaled.head()}")

        return df_scaled


    # Prepare Inputs for each Model ----------------------------------------------------------------------------------------

    def prepare_inputs_lstm(self, str_model:str, idx_train:bool, date_dict:dict=None, replace_data=None):
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
        :param replace_data: (dict) Dictionary containing replacement subsets, with which existing data can be overridden,
                             e.g. to iteratively insert 1d-mean values for multi-day predictions.

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

        df_scaled = self.prepare_inputs(str_model=str_model, idx_train=idx_train, date_dict=date_dict)

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
        # print(f"df_scaled head: {df_scaled.head()}")
        # print(f"df_scaled tail: {df_scaled.tail()}")
        df_seq = lstm.generate_sequences(df=df_scaled,
                                         target_var=target_var,
                                         n_lookback=n_lookback,
                                         n_ahead=n_ahead,
                                         n_offset=n_offset,
                                         continuous=False,
                                         productive=True
                                         )

        return df_seq


    def prepare_inputs_svr(self, str_model:str, idx_train:bool, date_dict:dict=None):
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

        Returns:
        :return df_label: (df) Dataframe containing features and lagged labels, with lags ranging from 1 to n_timestep,
                          representing 1- to n-ahead predictions.
        :return model_names: (list) Names of the n models and labels - their naming is equivalent for simplified reference
                             in subsequent methods.
        """
        # Get config parameters
        target_var = get_params_from_config(function='get_label', str_model=str_model)['label']
        n_offset = get_params_from_config(function='build_svr', str_model=str_model)['n_offset']
        svr_hyperparams = get_params_from_config(function='build_svr', str_model=str_model)['hyperparameters']

        # Preparation up to Scaling
        df_scaled = self.prepare_inputs(str_model=str_model, idx_train=idx_train, date_dict=date_dict)
        # print(f"df_scaled prepare_inputs_svr: {df_scaled.head()}")
        print(f"features df_scaled: {df_scaled.columns}")

        # Build SVR Model Input
        svr = SVReg()
        df_label, model_names = svr.build_model_input(df=df_scaled,
                                                      target_var=target_var,
                                                      str_model=str_model,
                                                      n_offset=n_offset,
                                                      idx_train=idx_train
                                                      )
        print(f"features df_label: {df_label.columns}")


        return df_label, model_names


    # Train each Model -----------------------------------------------------------------------------------------------------

    def train_lstm(self, str_model:str, idx_train:bool=True):
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
        df_seq = self.prepare_inputs_lstm(str_model=str_model, idx_train=idx_train)
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


    def train_svr(self, str_model:str, idx_train:bool=True):
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
        df_label, model_names = self.prepare_inputs_svr(str_model=str_model, idx_train=idx_train)

        # Split features and labels
        x_train = split_dataframe(df_features=df_label, target_var=model_names, productive=True)[0]
        y_train = split_dataframe(df_features=df_label, target_var=model_names, productive=True)[2]

        print(f"x_train: {x_train.head()}")
        print(f"features: {x_train.columns}")
        print(f"y_train: {y_train.head()}")

        # Generate and Parameterize Models
        svr = SVReg()
        models = svr.build_svr(str_model='inlet1_svr')

        # Train models
        trained_models = svr.train_svr(svr_dict=models, x_train=x_train, y_train=y_train)

        # Save models as pickle file
        with open(f'models//attributes//trained_dict_{str_model}.pkl', 'wb') as file:
            pickle.dump(trained_models, file)

        return trained_models


    # Predict each model ---------------------------------------------------------------------------------------------------

    def predict_lstm(self, str_model:str, idx_train:bool=False, date_dict:dict=None):
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
        print(f"Date_from: {date_dict['date_from']}, Date_to: {date_dict['date_to']}")

        # Load prediction parameters & model
        model = load_model(f'models//attributes//model_{str_model}.keras')

        # Load prediction input
        df_seq = self.prepare_inputs_lstm(str_model=str_model, idx_train=idx_train, date_dict=date_dict)
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


    def predict_lstm_daywise(self, str_model:str, idx_train:bool=False):

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
        model = load_model(f'models//attributes//model_{str_model}.keras')

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
            print(f"Prediction day: {pred_day}")
            print(f"Date_from: {mod_dates['date_from']}, Date_to: {mod_dates['date_to']}")


            # Load prediction input
            df_seq = self.prepare_inputs_lstm(str_model=str_model,
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

        # Resample & interpolate predictions
        return ypred_series

        # Write predictions to DWH


    def forecast_svr(self, str_model:str, idx_train:bool=False, date_dict:dict=None): # different naming due to svr.predict_svr

        # Get Config Parameters
        label_name = get_params_from_config(function='get_label', str_model=str_model)['label']

        # Get SVR inputs
        df_label, model_names = self.prepare_inputs_svr(str_model=str_model, idx_train=idx_train)
        print(f"df_label features: {df_label.columns}")
        # print(f"model_names: {model_names}")

        # Split features and labels
        x_pred = df_label.copy() #split_dataframe(df_features=df_label, target_var=model_names, productive=True)[0]

        # Load models
        with open(f'models//attributes//trained_dict_{str_model}.pkl', 'rb') as file:
            trained_models = pickle.load(file)

        # print(f"trained models: {trained_models}")
        print(f"x_pred: {x_pred.head()}")
        print(f"features: {x_pred.columns}")

        # Instantiate SVR object
        svr = SVReg()
        y_pred = svr.predict_svr(trained_svr=trained_models, x_test=x_pred, str_model=str_model)

        # Rescale Predictions
        y_pred_rescaled = inverse_transform_minmax(df_scaled=y_pred, str_model=str_model, attributes=[label_name])

        return y_pred_rescaled

