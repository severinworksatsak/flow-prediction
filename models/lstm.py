import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Dropout, LSTM, Bidirectional
from keras.models import Sequential
from models.utility import get_params_from_config
from pytz import timezone

# Build LSTM class
class simpleLSTM():

    def __init__(self):
        # Load config from directory
        with open(Path("config/config.json"), "r") as jsonfile:
            self.config = json.load(jsonfile)

    def build_2layer_lstm(self, x_train, y_train, units_l1:int=200, activation_l1:str='tanh', dropout_l1:float=0.1,
                          units_l2:int=150, activation_l2:str='tanh', dropout_l2:float=0.1,
                          activation_dense:str='sigmoid', loss_f:str='mse', optimizer:str='Adam',
                          print_summary:bool=True):
        """
        Build a two-layer LSTM neural net.

        :param x_train: (array) 3D-shaped training features with dimensions batch_size, time_steps, seq_len
        :param y_train: (array) 2D-shaped label array with dimensions batch_size, time_steps
        :param units_l1: (int) Number of LSTM "neurons" in layer 1. Default is 200.
        :param activation_l1: (str) Activation function to use in layer 1. Default is 'tanh'.
        :param dropout_l1: (float) Dropout share within (0,1) for layer 1. Default is 0.2.
        :param units_l2: (int) Number of LSTM "neurons" in layer 2. Default is 150.
        :param activation_l2: (str) Activation function to use in layer 2. Default is 'tanh'.
        :param dropout_l2: (float) Dropout share within (0,1) for layer 2. Default is 0.1.
        :param activation_dense: (str) Activation function to use in layer 2. Default is 'sigmoid' to ensure scaling
                                 between [0,1].
        :param loss_f: (str) Loss function to-be-used for weight optimization. Default is 'mse'.
        :param optimizer: (str) Optimizer to use. Default is 'Adam'.
        :param print_summary: (bool) Boolean indicator whether to print model summary after executing build_2layer_lstm.

        :return: Parameterized keras 2l-LSTM model.
        """
        # Get dimensions
        n_timestep = x_train.shape[1]
        n_features = x_train.shape[2]
        n_ahead = y_train.shape[1]

        # Build Model
        model = Sequential()
        model.add(Input(shape=(n_timestep, n_features), dtype='float64'))
        model.add(LSTM(units=units_l1, activation=activation_l1, return_sequences=True))
        model.add(Dropout(dropout_l1))
        model.add(LSTM(units=units_l2, activation=activation_l2))
        model.add(Dropout(dropout_l2))
        model.add(Dense(units=n_ahead, activation=activation_dense))
        model.compile(loss=loss_f, optimizer=optimizer)
        if print_summary:
            print(model.summary())

        return model


    def build_2layer_bidirectional_lstm(self, x_train, y_train, units_l1:int=200, activation_l1:str='tanh',
                                        dropout_l1:float=0.1, units_l2:int=150, activation_l2:str='tanh',
                                        dropout_l2:float=0.1, activation_dense:str='sigmoid', loss_f:str='mse',
                                        optimizer:str='Adam', print_summary:bool=True):
        """
        Build a two-layer LSTM neural net.

        :param x_train: (array) 3D-shaped training features with dimensions batch_size, time_steps, seq_len
        :param y_train: (array) 2D-shaped label array with dimensions batch_size, time_steps
        :param units_l1: (int) Number of LSTM "neurons" in layer 1. Default is 200.
        :param activation_l1: (str) Activation function to use in layer 1. Default is 'tanh'.
        :param dropout_l1: (float) Dropout share within (0,1) for layer 1. Default is 0.2.
        :param units_l2: (int) Number of LSTM "neurons" in layer 2. Default is 150.
        :param activation_l2: (str) Activation function to use in layer 2. Default is 'tanh'.
        :param dropout_l2: (float) Dropout share within (0,1) for layer 2. Default is 0.1.
        :param activation_dense: (str) Activation function to use in layer 2. Default is 'sigmoid' to ensure scaling
                                 between [0,1].
        :param loss_f: (str) Loss function to-be-used for weight optimization. Default is 'mse'.
        :param optimizer: (str) Optimizer to use. Default is 'Adam'.
        :param print_summary: (bool) Boolean indicator whether to print model summary after executing build_2layer_lstm.

        :return: Parameterized keras 2l-LSTM model.
        """
        # Get dimensions
        n_timestep = x_train.shape[1]
        n_features = x_train.shape[2]
        n_ahead = y_train.shape[1]

        # Build Model
        model = Sequential()
        model.add(Input(shape=(n_timestep, n_features), dtype='float64'))
        model.add(Bidirectional(LSTM(units=units_l1, activation=activation_l1, return_sequences=True)))
        model.add(Dropout(dropout_l1))
        model.add(Bidirectional(LSTM(units=units_l2, activation=activation_l2)))
        model.add(Dropout(dropout_l2))
        model.add(Dense(units=n_ahead, activation=activation_dense))
        model.compile(loss=loss_f, optimizer=optimizer)
        if print_summary:
            print(model.summary())

        return model

    # Sequencing for LSTM
    def generate_sequences(self, df, target_var: str, n_lookback: int, n_ahead: int, n_offset: int = 0,
                           drop_target: bool = True, continuous: bool = True, n_timestep: int = 6,
                           productive:bool=False, train_share: float = 0.7):
        """
        Generate sequence arrays for LSTM and perform train-test-split.
        :param df: Dataframe containing x and y features with datetime index.
        :param target_var: (str) Column name of target variable y in df.
        :param n_lookback: (int) Sequence length of features x.
        :param n_ahead: (int) Sequence length of target variable y.
        :param n_offset: (int) Offset interval between feature and target sequence as expressed in number of timesteps.
                         Defaults to 0.
        :param drop_target: (bool) Boolean indicator to drop target variable from x sample.
                            Defaults to True.
        :param continuous: (bool) Boolean indicator whether sequencing should be continuous or step-wise. Step-wise can
                           be used if the prediction should be only once a day for the entire day ahead.
        :param n_timestep: (int) Number of timesteps in a day. Used to ensure only one sequence per day occurring.
                           Default is 6.
        :param train_share: (float) Train-test split threshold; train_share = % of dataset included
                            in train sample.
        :return: Variable sequence arrays x_train, x_test, y_train, y_test.
        """
        # Separate target variable
        df_y = df[target_var]
        df_x = df.drop(columns=[target_var]) if drop_target else df

        # Get timestamps
        df_timestamps = df.index.tz_convert('UTC').astype(np.int64) // 10**9
        date_freq = pd.infer_freq(df.index)

        # Prepare sequence variables
        x_list = []
        y_list = []
        timestamps = []
        df_x_numpy = df_x.to_numpy()
        df_y_numpy = df_y.to_numpy()
        timestamps_numpy = df_timestamps.to_numpy()

        # Create sequences
        for timestep in range(n_lookback, df_x_numpy.shape[0] - n_ahead - n_offset):
            # Slice df into series
            x_sequence_t = df_x_numpy[timestep - n_lookback:timestep]
            y_sequence_t = df_y_numpy[timestep + n_offset:timestep + n_ahead + n_offset]
            timestamps_t = timestamps_numpy[timestep + n_offset:timestep + n_ahead + n_offset]

            # Assign series to strawmans
            x_list.append(x_sequence_t)
            y_list.append(y_sequence_t)
            timestamps.append(timestamps_t)

        # Filter out observations in case of discontinuous sequences (i.e. keep only daily sequences) such that
        # sequence length corresponds to exactly one day
        if not continuous:
            x_list = x_list[::n_timestep]
            y_list = y_list[::n_timestep]
            timestamps = timestamps[::n_timestep]

        '''
        Split into train & test datasets only if not in productive version -> in productive version, full loaded input
        is used for training. Validation set is directly calculated in deepl.train_model() method. 
        '''
        if productive:
            # Return full input
            x_train = np.array(x_list)
            y_train = np.array(y_list)
            x_test = None
            y_test = None

            return x_train, y_train, x_test, y_test

        else:
            # Train Test Split & Numpy Conversion
            x_train, x_test, y_train, y_test = train_test_split(np.array(x_list), np.array(y_list),
                                                                train_size=train_share, shuffle=False)

            timestamp_train, timestamp_test = train_test_split(np.array(timestamps), train_size=train_share,
                                                               shuffle=False)

            # Save start date of y_test

            min_time = np.min(timestamp_test)
            min_test_date = pd.to_datetime(min_time, unit='s', utc=True).tz_convert('Etc/GMT-1')
            self.ytest_startdate = min_test_date

            return x_train, y_train, x_test, y_test


    def convert_seq_to_df(self, seq_array, n_timestep:int=None, start_date=None, str_model:str='inlet1_lstm'):
        """
        Convert LSTM sequence into dataframe by concatenating subsequent sequences. Function is built for
        non-overlapping sequences!

        :param seq_array: (array) Numpy array of sequences to-be-converted.
        :param start_date: (str, datetime) First timestamp of datetime index, which is added to the converted sequence.
        :param n_timestep: (int) Number of intraday timestemps that should be passed onto the model's input layer. Loaded
                           from the config json.

        :return: Dataframe with concatenated sequences; each new sequence fills a new row.

        :note: For further use, the resulting dataframe often has to be converted to resemble pd.Series structure. Also
               see dailydf_to_ts method.
        """
        # Ensure datetime object
        if start_date is None:
            start_date = self.ytest_startdate
        elif isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%d.%m.%Y %H:%M:%S')
        else:
            raise ValueError("Start Date must be either datetime object or 'str' with format '%d.%m.%Y %H:%M:%S'")

        # Retrieve config values
        if n_timestep is None:
            n_timestep = get_params_from_config('get_n_timestep', str_model)['n_timestep']

        # Localize in Etc/GMT-1 timezone
        tz = timezone('Etc/GMT-1')
        if start_date.tzinfo is None:
            start_date = tz.localize(start_date)
        elif start_date.tzinfo != tz:
            start_date = start_date.astimezone(tz)

        # Create datetime index from date range
        date_index = pd.date_range(start=start_date,
                                   periods=seq_array.shape[0],
                                   tz='Etc/GMT-1',
                                   freq=f'{24 // n_timestep}h')

        # Create dataframe
        df_seq = pd.DataFrame().from_records(seq_array)
        df_seq.index = date_index

        return df_seq
