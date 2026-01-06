# src/ct/config/validate.py
from ct.config.schema import ResolvedConfig

def validate_schema(resolved: dict) -> ResolvedConfig:
    return ResolvedConfig.model_validate(resolved)