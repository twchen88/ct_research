# Provenance Log for Version 2

This file documents the origin, purpose, and status of source code, notebooks, and datasets that has been refactored from version 1. This version contains all the code from completed section of proposal.

---
## top-level scripts
**`00_pull_data.py`**: Takes in an SQL query (specified in config file in config/pull/), pulls the data, and stores data in data/raw_data/.
- `python 00_pull_data.py --config [config_file_path]`
**`01_preprocess_data.py`**: Takes in filtering parameters and raw data (specified in config files in config/preprocess/), preprocess data accordingly as well as clean data up so that they can be encoded for the predictor model. Output stored in data/preprocessed/.
**`02_encode_data.py`**
**`03_train_predictor.py`**
**`04_assess_accuracy_repeat.py`**
**`05_assess_accuracy_nonrepeat.py`**
**`06_run_trajectory_experiment.py`**

## config/

### connection/
Purpose: store database information, credentials, and SSH key (gitignore)

### pull_config_*.yaml
Purpose: config files for pulling from the Constant Therapy database
Notes: date the data pulled in MMDDYY, yaml file contains sql query used as well as output data file path.

## data/
Notes: CSV files should be saved without index column

## notebooks/

## outputs/

## sql/
- `predictor_data_query_*.sql`: preliminary data query with basic filtering information, created based on Claire Cordella's queries. Date in MMDDYY.

## src/
- `test_*.py`: tests functions in respective files

### data/
- `db_utils.py`: contains functions that connects to SQL database and runs SQL queries as well as saving metadata related to database access
- `data_io.py`: contains functions that loads and writes session data in CSV format
- `preprocessing.py`: contains functions that filters and reformat the dataset so that it is suitable for training
- `encoding.py`:


### experiments/
#### predictor
- `training.py`
- `model.py`
- `hyperparameter.py`
- `pipeline.py`
- `evaluation.py`

## test/