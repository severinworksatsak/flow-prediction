import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from models.lstm import simpleLSTM
from tqdm import tqdm


class DeepLearner():

    def train_model(self, x_train, y_train, model, n_patience:int, n_epochs:int,
                    n_batch_size:int, val_share:float=0.2, verbose_int=0):
        """
        Train a keras model on training data with an additional validation holdout set and early stopping and dynamic
        learning rate reduction.

        :param x_train: (array) Numpy array or pd.DataFrame with training features.
        :param y_train: (array) Numpy array or pd.Series of training labels.
        :param model: (keras object) Fully parameterized Keras model.
        :param n_patience: (int) Patience parameter for early stopping and learning rate reduction. The higher the
                                 patience, the more epochs without error reduction the model tolerates during training.
        :param n_epochs: (int) Number of epochs, i.e. number of reiterations through the whole dataset during training.
        :param n_batch_size: (int) Batch size for training. Setting n_batch_size = 1 leads to Stochastic Gradient
                             Descent; setting 1 < n_batch_size < # samples leads to Mini Batch Gradient Descent; setting
                             n_batch_size = # samples leads to Batch Gradient Descent.
        :param val_share: (float) Share of validation set expressed as number from (0,1)
        :param verbose_int: (int) Verbose level to control extent of printed output.

        :return: Trained keras/sklearn model.
        :return: Training history.
        """
        # Split off validation data
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_share, shuffle=False)

        # Fit model incl early stopping & Learning rate
        stop_condition = EarlyStopping(monitor='val_loss', mode='min', verbose=0, # Change back to 0
                                       patience=n_patience, restore_best_weights=True)

        lrate_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=n_patience-1, min_lr=1e-7)

        training = model.fit(x=x_train, y=y_train, validation_data=(x_val, y_val), epochs=n_epochs,
                             verbose=verbose_int, batch_size=n_batch_size, callbacks=[stop_condition, lrate_reduction])

        return model, training

    def predict_model(self, x_test, y_test, trained_model):
        """
        Predict keras model based on pre-trained model.

        :param x_test: (array) Input feature array for test period.
        :param y_test: (array) Label array for test period.
        :param trained_model: (keras object) Pre-trained keras model.

        :return: Model prediction array in model-defined dimensions.
        """
        # Prediction step
        return trained_model.predict(x_test)

    def get_grid_train_dict(self, lookback_list:list, df, target_var:str, n_ahead:int=6, **kwargs):
        """
        Construct training sequence dictionary for LSTM features and label vectors.

        :param lookback_list: (list) List of lookback periods, i.e. selection of sequence lengths of features x.
        :param df: (pd.DataFrame) DataFrame containing both input and label variables -
        :param target_var: (str) Column name of target variable y in df.
        :param n_ahead: (int) Sequence length of target variable y.
        :param kwargs: Keyword arguments compatible with generate_sequences() method's inputs.

        :return: Dictionary with lookback period as keys and x_train, y_train sequence arrays as values.
        """
        lstm = simpleLSTM()
        input_dict = {}

        # Get x_train and y_train for all specified lengths
        for n_lookback in lookback_list:
            sequence_array = lstm.generate_sequences(df, target_var, n_lookback, n_ahead, **kwargs)
            x_train = sequence_array[0]
            y_train = sequence_array[2]
            input_dict[n_lookback] = {
                'x_train': x_train,
                'y_train': y_train
            }

        return input_dict

    def grid_search_lstm(self, input_dict, params_dict, n_patience=5, n_epochs=50, n_batch_size=32):
        """
        Run Grid Search for LSTM to find optimal hyperparameters.

        :param input_dict: (dict) Dictionary containing training features and labels for different lookback periods.
                           Generated with get_grid_train() method.
        :param params_dict: (dict) Dictionary with hyperparameters to be used for each lookback period. Hyperparameter
                            space is limited to input parameters for lstm.build_2layer_lstm() method.
        :param n_patience: (int) Patience parameter for early stopping and learning rate reduction. The higher the
                                 patience, the more epochs without error reduction the model tolerates during training.
        :param n_epochs: (int) Number of epochs, i.e. number of reiterations through the whole dataset during training.
        :param n_batch_size: (int) Batch size for training. Setting n_batch_size = 1 leads to Stochastic Gradient
                             Descent; setting 1 < n_batch_size < # samples leads to Mini Batch Gradient Descent; setting
                             n_batch_size = # samples leads to Batch Gradient Descent.
        :return:
        """
        # Initiate objects
        sl = simpleLSTM()

        # Prepare Data
        mse = []
        val_loss = []
        lookback_window = []
        param_grid = ParameterGrid(params_dict)

        # Loop over all window lengths
        for n_lookback in input_dict.keys():
            # Define training data sample
            x_train = input_dict[n_lookback]['x_train']
            y_train = input_dict[n_lookback]['y_train']

            # Train all combinations
            for count, configuration in tqdm(enumerate(param_grid)):
                # Instantiate model
                model = sl.build_2layer_lstm(x_train, y_train, **configuration, print_summary=False)

                # Train individual model
                trained_model, history = self.train_model(x_train, y_train, model, n_epochs, n_patience, n_batch_size)

                # Add evaluation metrics to list
                mse.append(history.history['loss'][-1])
                val_loss.append(history.history['val_loss'][-1])
                lookback_window.append(n_lookback)

        # Extract best model
        metrics = pd.DataFrame({'mse': mse, 'val_loss': val_loss, 'lookback': lookback_window})
        best_model = param_grid[(metrics['mse'].rank() + metrics['val_loss'].rank()).idxmin()]
        # best_loss = metrics[(metrics['mse'].rank() + metrics['val_loss'].rank()).idxmin()]

        return metrics, best_model
