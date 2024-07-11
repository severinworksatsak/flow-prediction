import pandas as pd
from datetime import datetime
import json
from pathlib import Path
from sklearn.svm import SVR

from models.utility import get_params_from_config

# Build SVR class
class SVReg():

    def __init__(self):
        # Load config from directory
        with open(Path("config/config.json"), "r") as jsonfile:
            self.config = json.load(jsonfile)
            self.model_config = self.config['model']

    # Arrange model input
    @staticmethod
    def build_model_input(df, target_var:str, str_model:str, n_offset:int=None, n_timestep:int=None,
                          idx_train:bool=True):
        """
        Create new target variables for each timestep-SVR model. As SVR output is single-length, a full-day prediction
        requires the training of n_timestep models and thus n_timestep different target variables.

        Parameters:
        :param df: Feature dataframe containing the target variable.
        :param target_var: Name of the target variable passed as string - used for column referencing.
        :param model_str: Name of model for which parameter n_timestep should be loaded from config.
        :param n_offset: Number of timesteps between training and first prediction.
        :param n_timestep: Number of intra-day timesteps within a day.

        Returns:
        :return df: Dataframe with target variables for n_timestep models.
        :return name_list: List with all label names.

        Notes:
        :note: If prediction is to be made for next timestep, offset parameter has to be set to 0. In other words, the
               default forecase period will be t+1. If inputs x ranging from t-(x+y) through t-x should be used to
               predict y(t-x), an offset of -1 can be used.
        """
        # Retrieve config values
        if n_timestep is None:
            n_timestep = get_params_from_config('get_n_timestep', str_model)['n_timestep']

        if n_offset is None:
            n_offset = n_timestep

        # Create target variables for every model
        name_list = []
        df = df.copy()

        for timestep in range(n_timestep):
            # Individual model parameters
            model_num = timestep + 1
            model_name = f"model{model_num}"
            lag = -(timestep + 1 + n_offset)
            y_name = f"y_{model_name}"
            name_list.append(y_name)

            if idx_train:
                # Lag target variable & remove starting NAs
                df[y_name] = df[target_var].shift(lag)
                df[y_name] = df[y_name].bfill()

        # Drop original label & remove trailing NAs
        df_save = df.drop(columns=[target_var])
        df_save.dropna(inplace=True)

        return df_save, name_list


    def build_svr(self, str_model:str='inlet1_svr', label_names:list=None, model_dict:dict=None):
        """
        Build and parameterize an SVR model for each specified model name.

        :param str_model: (str) Model name as used in section header of json config.
        :param label_names: (list) List of names of the individual models. Individual names should follow the
                            convention 'y_model_x' with x for the sequential model number.
        :param model_dict: (dict) Dictionary containing 'valid_for' and 'hyperparams' keys. If a model name appears
                           in the valid_for list, the model will be built on these hyperparameters.

        :return: (dict) Dictionary with model_name : configured sklearn SVR model pairs.
        """
        # Check None conditions
        if model_dict is None:
            model_dict = self.model_config[str_model]['parameters']['architecture']['hyperparameters']

        if label_names is None:
            label_names = []
            for key in model_dict.keys():
                label_names += model_dict[key]['valid_for']

        # Sequentially build models
        output_dict = {}

        for name_model in label_names:
            # Check which hyperparams to use for given model
            for key in model_dict.keys():
                if name_model in model_dict[key]['valid_for']:
                    param_set = model_dict[key]['hyperparams']
                    break

            # Create & Parameterize Model
            try:
                model_i = SVR(**param_set)
            except:
                raise ValueError(f"Model {name_model} not parameterized in config. Include in 'valid_for' "
                                 f"or add hyperparams.")

            # Assign Model
            output_dict[name_model] = model_i

        return output_dict


    def train_svr(self, svr_dict:dict, x_train, y_train):
        """
        Train each model from svr_dict individually.

        :param svr_dict: (dict) Dictionary with model_name : configured sklearn SVR model pairs. Can be taken from
                         build_svr() method.
        :param x_train: (array) Numpy array or pd.DataFrame with training features.
        :param y_train: (array) Numpy array or pd.Series of training labels. Important: One individual label series per
                        model is required - can be taken from output of build_model_input() method.

        :return: (dict) Dictionary containing trained sklearn SVR model objects.
        """
        # Train Models Individually
        output_dict = {}

        for model_i in svr_dict.keys():
            to_train = svr_dict[model_i]
            trained_model = to_train.fit(x_train, y_train[model_i])
            output_dict[model_i] = trained_model

        return output_dict

    def predict_svr(self, trained_svr:dict, x_test, date_from=None, freq:str=None, str_model:str='inlet1_svr',
                    n_timestep:int=None):
        """
        Predict timestep for each model individually.

        :param trained_svr: (dict) Dictionary containing pre-trained sklearn SVR objects.
        :param x_test: (array) Numpy array or pd.DataFrame containing test features.
        :param date_from: (str, datetime) Start of daterange of the output datetime index. Can either be a
                          datetime object or string of the format '%d.%m.%Y %H:%M:%S'. Timezone will be assumed to be
                          'Etc/GMT-1'.
        :param freq: (str) Frequency of datetime index of output series. Default is None. Options are equivalent to
                           pd.daterange method. Default is None.
        :param str_model: (str) Model name as used in section header of json config. Default is 'inlet1_svr'.
        :param n_timestep: (int) Number of intra-day timesteps within a day. Default is None, in which case parameter
                           is loaded from the config.

        :return: (pd.Series) Prediction time series.
        """
        # None Check
        if n_timestep is None:
            n_timestep = get_params_from_config('get_n_timestep', str_model)['n_timestep']

        if freq is None:
            freq = f'{24 // n_timestep}h'

        if date_from is None:
            date_index = x_test.index
        else:
            start = datetime.strptime(date_from, '%d.%m.%Y %H:%M:%S') if isinstance(date_from, str) else date_from
            date_index = pd.date_range(start, periods=len(x_test), freq=freq, tz='Etc/GMT-1')

        # Predict step-wise with each model
        output_df = pd.DataFrame()

        for count, model_i in enumerate(trained_svr.keys()):
            # Shifted test features for each model
            x_test_i = x_test.iloc[count::n_timestep]

            # Predict with Model
            output_df[model_i] = pd.Series(trained_svr[model_i].predict(x_test_i))

        # Combine all predictions
        output_series = output_df.stack()
        output_series.reset_index(drop=True, inplace=True)
        output_series.index = date_index

        return output_series
