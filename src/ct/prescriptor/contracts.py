import pandas as pd

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from ct.predictor.contracts import Predictor


class Prescriptor(ABC):
    """Abstract base class for prescriptors."""
    def __init__(self, config: Dict[str, Any],
                 cao_mapping: Dict[str, List[str]],
                 model_params: Dict[str, Any],
                 predictors: List[Predictor],
                 data_encoder = None, ##TODO: specify type when that contract is done
                 evaluation_df: Optional[pd.DataFrame] = None) -> None:
        self.config = config
        self.cao_mapping = cao_mapping
        self.predictors = predictors
        self.data_encoder = data_encoder
        self.evaluation_df = evaluation_df
        self.model_params = model_params
        
    @abstractmethod
    def build_model(self) -> None:
        """Build the prescriptor model."""
        pass

    @abstractmethod
    def prescribe(self, context: pd.DataFrame) -> pd.DataFrame:
        """Given a context, prescribe an action.

        :param context: A dictionary representing the context.
        :return: A dictionary representing the prescribed action.
        """
        pass

    @abstractmethod
    def save_model(self, model_path: str) -> None:
        """Save the prescriptor model to the specified path.

        :param model_path: The path to save the model.
        """
        pass

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load the prescriptor model from the specified path.

        :param model_path: The path to load the model from.
        """
        pass

    def get_action_names(self) -> List[str]:
        """Get the action names from the CAO mapping.

        :param cao_mapping: The CAO mapping.
        :return: A list of action names.
        """
        return list(self.cao_mapping["actions"])
    
    def get_context_names(self) -> List[str]:
        """Get the context names from the CAO mapping.

        :param cao_mapping: The CAO mapping.
        :return: A list of context names.
        """
        return list(self.cao_mapping["contexts"])