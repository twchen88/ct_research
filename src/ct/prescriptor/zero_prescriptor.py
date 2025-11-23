import pandas as pd
import numpy as np

from typing import Dict, Any, List, Optional

from ct.predictor.contracts import Predictor
from ct.prescriptor.contracts import Prescriptor

from ct.utils.logger import get_logger
logger = get_logger(__name__)

class ZeroPrescriptor(Prescriptor):
    """Always prescribes 0 for each intervention (one per domain)."""
    def __init__(self,
                 config: Dict[str, Any],
                 cao_mapping: Dict[str, List[str]],
                 model_params: Dict[str, Any],
                 predictors: List[Predictor],
                 data_encoder = None,
                 evaluation_df: Optional[pd.DataFrame] = None) -> None:
        super().__init__(config, cao_mapping, model_params, predictors, data_encoder, evaluation_df)
        
        self.action_names = self.get_action_names()
        self.n_actions = len(self.action_names)
        logger.info(f"Initialized ZeroPrescriptor with {self.n_actions} actions.")

    def build_model(self) -> None:
        """Does nothing as there is no model to build."""
        return super().build_model()
    
    def prescribe(self, context: pd.DataFrame) -> pd.DataFrame:
        n = len(context)
        zeros = np.zeros((n, self.n_actions), dtype=int)
        return pd.DataFrame(zeros, columns=self.action_names)
    
    def save_model(self, model_path: str) -> None:
        """Does nothing as there is no model to save."""
        return super().save_model(model_path)
    
    def load_model(self, model_path: str) -> None:
        """Does nothing as there is no model to load."""
        return super().load_model(model_path)