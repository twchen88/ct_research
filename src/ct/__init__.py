# ct/__init__.py
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ct")
except PackageNotFoundError:
    # Fallback when running from source without installation
    __version__ = "0.0.0"

# Optional: expose subpackages at the top level
from . import datasets, experiments, predictor, utils, viz, config, pipeline, io

__all__ = ["datasets", "experiments", "predictor", "utils", "viz", "config", "pipeline", "io"]