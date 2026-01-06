# src/ct/config/schema.py
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any, List

class PullRawConfig(BaseModel):
    metadata: dict
    source: dict
    output: dict
    snapshot: dict = {}
    logging: dict = {}


### PLACEHOLDERS FOR FULL CONFIG SCHEMA ###
class RunCfg(BaseModel):
    name: str
    raw_snapshot: str = "latest"
    seed: int = 0
    artifact_root: str = "artifacts"
    run_root: str = "runs"
    log_level: str = "INFO"

class DatasetStageCfg(BaseModel):
    name: str
    filtering: Dict[str, Any] = {}
    aggregation: Dict[str, Any] = {}
    encoding: Dict[str, Any] = {}

class TrainStageCfg(BaseModel):
    name: str
    family: Literal["predictor", "prescriptor"] = "predictor"
    model: Dict[str, Any]
    training: Dict[str, Any] = {}

class AnalysisStageCfg(BaseModel):
    name: str
    simulation: Dict[str, Any] = {}

class ResolvedConfig(BaseModel):
    run: RunCfg
    stages: Dict[str, Any]
    inputs: Dict[str, Any] = {}