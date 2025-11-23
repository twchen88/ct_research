import numbers

import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional, Type
from sklearn.model_selection import train_test_split

from ct.utils.logger import get_logger
logger = get_logger(__name__)

"""
This module contains the abstract base class for all predictors. Modified from NeuroAI Predictor contract.
"""

class Predictor(ABC):
    """
    This class contains the contract that any predictor
    must implement.
    """

    def __init__(self,
                 config: Dict[str, Any],
                 data_df: pd.DataFrame,
                 cao_mapping: Dict[str, List[str]],
                 data_split: Dict[str, float],
                 model_params: Dict[str, Any] = dict(),
                 metadata: Dict[str, Any] = dict()):
        """
        Initializes a predictor, its parameters, and the metadata.

        Parameters:
            config (Dict[str, Any]): Configuration dictionary for the predictor. Should include learning rate,
            random seed, weight decay, and other hyperparameters.
            data_df (pd.DataFrame): Dataframe containt all processed data
            cao_mapping (Dict[str, List[str]]): a dictionary with `context`, `actions` and `outcomes`
            keys where each key returns a List of the selected column names as strings.
            data_split (Dict[str, float]): Dictionary containing the training splits indexed
            by "train_pct" and "val_pct".
            model_params (Dict[str, Any], optional): Parameters of the model. Defaults to empty dictionary.
            metadata (Dict[str, Any], optional): Dictionary describing any other information
            that must be stored along with the model.
            This might help in uniquely identifying the model. Defaults to empty dictionary.
        """
        # store the parameters
        self.config = config
        self.df = data_df
        self.cao_mapping = cao_mapping
        self.data_split = data_split
        self.model_params = model_params
        self.metadata = metadata
        
        # Store context and action columns for prediction
        self.context_actions_columns = self.cao_mapping["context"] + self.cao_mapping["actions"]

        self.column_length = {} # keeps track of number of values encodes outcome
        if data_df is not None:
            train_df, val_df, test_df = self.generate_data_split(data_df, self.data_split)

            # Split the data between features (x) and labels(y)
            self.train_x_df, self.train_y_df = self.get_data_xy_split(train_df, cao_mapping)
            self.val_x_df, self.val_y_df = self.get_data_xy_split(val_df, cao_mapping)
            self.test_x_df, self.test_y_df = self.get_data_xy_split(test_df, cao_mapping)

            # make sure the train_y_df is not None
            if self.train_y_df is None:
                logger.error("train_y_df is None after splitting the data; cannot proceed.")
                raise ValueError("train_y_df is None after splitting the data; cannot proceed.")

            # Keep track of how many values are used to encode each outcome
            for column in self.cao_mapping["outcomes"]:
                first_value = self.train_y_df[column].head(1).values[0]
                if isinstance(first_value, numbers.Number):
                    # Value is a single scalar
                    self.column_length[column] = 1
                else:
                    # value is a one-hot encoded vector, i.e. a list. Get its size.
                    self.column_length[column] = len(self.train_y_df[column].head(1).values[0])
        else:
            # No data provided
            logger.error("No data_df provided to Predictor")
            raise ValueError("No data_df provided to Predictor")

        if model_params is None:
            model_params = {}
        self.model_params = model_params

        if metadata is None:
            metadata = {}
        self.metadata = metadata

        # Internal Parameters that are used to store the
        # latest state of the model.
        self._trained_model = None

    @abstractmethod
    def build_model(self, model_params: Dict):
        """
        This function must be overridden to build the model using the model
        parameters if desired and return a model.
        :param model_params: Dictionary containing the model parameters
        :return model: The built model.
        """

    @abstractmethod
    def train_model(self, model,
                    train_x: np.ndarray, train_y: np.ndarray,
                    val_x: Optional[np.ndarray], val_y: Optional[np.ndarray]) -> Type:
        """
        This function must be overridden to train the built model from the build_model step
        given the Data and must return the trained model.
        :param model: The model built in the build_model step
        :param train_x: numpy array containing the processed input features split for training
        :param train_y: numpy array containing the processed output features split for training
        :param val_x: Optional numpy array containing the processed input features split for validation
        :param val_y: Optional numpy array containing the processed output features split for validation

        :return trained_model
        """

    @abstractmethod
    def parse_config(self) -> Dict[str, Any]:
        """
        This function must be overridden to parse the config dictionary
        passed during initialization and return a dictionary of model parameters
        Return:
            model_params(Dict[str, Any]): Dictionary containing the model parameters
        """

    @abstractmethod
    def predict(self, encoded_context_actions_df: pd.DataFrame) -> pd.DataFrame:
        """
        This method uses the trained model to make a prediction for the passed Pandas DataFrame
        of context and actions. Returns the predicted outcomes in a Pandas DataFrame.
        :param encoded_context_actions_df: a Pandas DataFrame containing encoded rows of context and actions for
        which a prediction is requested. Categorical columns contain one-hot vectors, e.g. [1, 0, 0]. Which means
        a row can be a list of arrays (1 per column), e.g.: [1, 0, 0], [1,0].
        :return a Pandas DataFrame of the predicted outcomes for each context and actions row.
        """

    @abstractmethod
    def save_model(self, model_path: str, params_file_path: str) -> None:
        """
        Saves the trained model to the specified location
        :param file_path: the name and path of the file to persist the bytes to
        :return: nothing
        """
        
    @abstractmethod
    def load_saved_model(self, model_path: str, params_file_path: str) -> None:
        """
        Loads the trained model from the specified location
        :param file_path: the name and path of the file to load the bytes from
        :return: nothing
        """

    @abstractmethod
    def evaluate_model(self,
                       test_x: np.ndarray, test_y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates the trained model on the test data and returns evaluation metrics
        """

    def set_trained_model(self, trained_model) -> None:
        """
        Sets the underlying trained model to the passed one.
        :param trained_model: a trained model
        :return Nothing:
        """
        self._trained_model = trained_model

    def get_trained_model(self):
        """
        Returns the trained model if it has been set, None otherwise
        :return self._trained_model:
        """
        return self._trained_model

    @staticmethod
    def generate_data_split(data_df: pd.DataFrame,
                            data_split: Dict[str, Any]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
        """
        Splits the data between train, validation (optional) and test sets
        :param data_df: the full dataset as a Pandas DataFrame
        :param data_split: a dictionary with the
        :return: a tuple of Pandas DataFrame: one for train, one for validation (or None), and one for test
        """
        logger.info(f"Generating data splits with parameters: {data_split}")
        # First, split the data set in train and test sets.
        # Use the provided random_state, if any
        random_state = data_split.get("random_state")
        if random_state is None:
            random_state = 42
            # log that no random seed was provided and it was set to 42 as default
            logger.warning("No random seed provided in data_split; defaulting to 42")

        shuffle = data_split.get("shuffle")
        if shuffle is None:
            shuffle = True
            # log that no shuffle option was provided and it was set to True as default
            logger.warning("No shuffle option provided in data_split; defaulting to True")

        logger.info(f"Generating train and test data split with test_pct={data_split['test_pct']}, random_state={random_state} and shuffle={shuffle}")
        train_df, test_df = train_test_split(data_df,
                                             test_size=data_split["test_pct"],
                                             random_state=random_state,
                                             shuffle=shuffle)

        # If we also need a validation set, split the train set into train and validation sets.
        val_pct = data_split.get("val_pct")
        if val_pct is None:
            val_pct = 0
            logger.warning("No val_pct option provided in data_split; defaulting to 0")
        
        logger.info(f"Generating training and validation splits with val_pct={val_pct}")
        if val_pct > 0:
            train_df, val_df = train_test_split(train_df,
                                                test_size=val_pct,
                                                random_state=random_state,
                                                shuffle=shuffle)
        else:
            logger.warning("No validation set requested; skipping validation split. val_df set to None.")
            val_df = None
        return train_df, val_df, test_df

    @staticmethod
    def get_data_xy_split(data_df: Optional[pd.DataFrame],
                          cao_mapping: Dict) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        This function takes a dataframe and a dictionary mapping indices to context,
        action, or outcome. This then splits the dataframe into two dataframes based
        on it's CAO tagging.

        data_x: Context and Actions
        data_y: Outcomes

        :param data_df: a Pandas DataFrame with all the data
        :param cao_mapping: a dictionary with `context`, `actions` and `outcomes` keys where each key returns a List
         ofthe selected column names as strings.
        :return: A tuple containing two dataframes: data_x with the features, and data_y with the labels (outcomes)
        """
        if data_df is None:
            logger.warning("No data provided to get_data_xy_split; returning None for both x and y dataframes")
            return None, None

        data_x_df = data_df[cao_mapping["context"] + cao_mapping["actions"]]
        data_y_df = data_df[cao_mapping["outcomes"]]

        return data_x_df, data_y_df
