import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ct.predictor.contracts import Predictor
from ct.utils.torch_layer_map import TORCH_LAYER_MAP as LAYER_MAP
from ct.utils.torch_layer_map import TORCH_ACTIVATION_MAP as ACTIVATION_MAP
from typing import Dict, Any, Optional, List

from ct.utils.logger import get_logger
logger = get_logger(__name__)

class NextStepPredictor(Predictor):
    """
    A simple neural network model for predicting target domains based on current scores and target domain encoding.
    The model consists of two linear layers with a sigmoid activation function in between.
    Can be set so that a Sigmoid activation is applied at the final output layer.
    """

    def __init__(self,
                config: Dict[str, Any],
                data_df: pd.DataFrame,
                cao_mapping: Dict[str, List[str]],
                data_split: Dict[str, float],
                model_params: List[Dict[str, Any]] = [dict()],
                metadata: Dict[str, Any] = dict()):
        
        super().__init__(
            config,
            data_df,
            cao_mapping,
            data_split,
            model_params,
            metadata
        )
        
        self.use_sigmoid_output = config.get("use_sigmoid_output", False)
        if not self.use_sigmoid_output:
            logger.info("Using Predictor model without Sigmoid at output layer.")
        else:
            logger.info("Using Predictor model WITH Sigmoid at output layer.")

        self.n_domains = config.get("n_domains", 14)
        logger.info(f"Number of target domains set to {self.n_domains}.")

    def _build_model_from_config(self, config: List[Dict[str, Any]]) -> nn.Sequential:
        layers = []

        for layer_spec in config:
            for layer_type, params in layer_spec.items():
                # Extract the activation
                activation = None
                if "activation" in params:
                    activation_name = params.pop("activation")
                    activation = ACTIVATION_MAP[activation_name]()  # activation class instance

                # Create the layer using remaining params
                layer_class = LAYER_MAP[layer_type]
                layer = layer_class(**params)
                layers.append(layer)

                # Add activation after layer if present
                if activation is not None:
                    layers.append(activation)

        return nn.Sequential(*layers)


    def build_model(self, model_params: Dict) -> nn.Sequential: #type: ignore
        model = self._build_model_from_config(model_params["layers"])
        return model
    
    def train_model(self, model, train_x, train_y, val_x, val_y): #type: ignore
        pass

    def predict(self, encoded_context_actions_df: pd.DataFrame) -> pd.DataFrame:
        if self._trained_model is None:
            logger.error("Model has not been trained yet or trained model hasn't been set yet.")
            raise ValueError("Model has not been trained yet or trained model hasn't been set yet.")
        self._trained_model.eval()