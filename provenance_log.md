# Provenance Log for Version 2

This file documents the origin, purpose, and status of source code, notebooks, and datasets.

---
## top-level files
- `.gitignore`
- `CAHNGELOG.md`
- `provenance_log.md`
- `pyproject.toml`
- `README.md`

## artifacts/

## configs/

### connection/
Purpose: store database information, credentials, and SSH key (gitignore due to sensitive information)

### run/

### stages/


## data/
Notes: CSV files should be saved without index column

## legacy/ (.gitignored)
- Legacy configs, outputs, scripts, and code from v1. Should be left till v2 refactoring is done, and documenting zip file has been created.

## notebooks/
Numbered by create date.

## reference/

## runs/

## sandbox/

## scripts/

## sql/
- `predictor_data_query_*.sql`: preliminary data query with basic filtering information, created based on Claire Cordella's queries. Date in MMDDYY.


## src/ct/
- `cli.py`

### config/

### datasets/
- `db_utils.py`: contains functions that connects to SQL database and runs SQL queries as well as saving metadata related to database access
- `data_io.py`: contains functions that loads and writes session data in CSV format
- `preprocessing.py`: contains functions that filters and reformat the dataset so that it is suitable for training
- `encoding.py`: contains functions that create missing indicators and encode target vectors

### experiments/
- `aggregate_average.py`: contains functions that are specific to aggregate average experiments
- `file_io.py`: 
- `shared.py`: 
- `trajectory.py`: 

### io/

### pipeline/

### predictor/

### prescriptor/

### utils/
- `config_loading.py`: loads json or yaml files
- `logger.py`: implements logger functionality
- `reproducibility.py`: sets global seed for reproducbility
- `metadata.py`: helper functions for metadata information

### viz/
- `aggregate_average.py`:
- `data.py`:  
- `training.py`: plots for model training
- `trajectory.py`: 

## tests/