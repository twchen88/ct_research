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
Purpose: store database information, credentials, and SSH key (gitignored due to sensitive information)

### run/

### stages/
#### pull/
- `test.yaml`: Config for testing purposes.

### templates/
- `raw.yaml`: Config template for pulling raw data. Used for either scripts/pull_raw.py or in the pipeline.

## data/
Notes: CSV files should be saved without index column

## legacy/ (gitignored)
- Legacy configs, outputs, scripts, and code from v1. Should be left till v2 refactoring is done, and documenting zip file has been created.

## notebooks/


## reference/
- `project_Constant_Therapy_ESP_Toy_Project_experiment_run_for_notebook_and_others_run_4522_notebook.ipynb`: reference for a simple ESP run that is template for the actual Constant Therapy data

## runs/

## sandbox/

## scripts/

## sql/
- `predictor_data_query_*.sql`: preliminary data query with basic filtering information, created based on Claire Cordella's queries. Date in MMDDYY.


## src/ct/
- `cli.py`

### config/

### datasets/
- `featuriation.py`
- `filtering.py`
- `history.py`


### experiments/
- `aggregate_average.py`: contains functions that are specific to aggregate average experiments
- `file_io.py`: 
- `shared.py`: 
- `trajectory.py`: 

### io/
Data IO functions as well as 
- `data_io.py`
- `db_utils.py`: functions that load SQL queries/files and create connection to external SQL systems
- `pull_raw.py`: defines public entry point for pulling and saving raw data
- `snapshots.py`: snapshots-related functions such as creating snapshot ID, metadata, and pointing to latest snapshot

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
- `unit/`
- `integration/`