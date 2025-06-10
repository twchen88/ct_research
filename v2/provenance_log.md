# Provenance Log for Version 2

This file documents the origin, purpose, and status of source code, notebooks, and datasets that has been refactored from version 1. This version contains all the code from completed section of proposal.

If dates are included in file names, it must be the date when the file is USED (when data is stored, processed, modified, etc), not when the config file is created.

---
## top-level scripts and files
### scripts
**`00_pull_data.py`**: Takes in an SQL query (specified in config file in config/pull/), pulls the data, and stores data in data/raw_data/ in .csv format.
- `python 00_pull_data.py --config config/pull/[YYYYMMDD].yaml`

**`01_preprocess_data.py`**: Takes in filtering parameters and raw data (specified in config files in config/preprocess/), preprocess data accordingly as well as clean data up so that they can be encoded for the predictor model. Output stored in data/preprocessed/ in .csv format.
- `python 01_preprocess_data.py --config config/preprocess/[YYYYMMDD].yaml`

**`02_encode_data.py`**: Takes in the preprocessed data specified in congig files in config/encode/, encode the score data with missing indicators and target data to be used for training. Output is stored in data/encoded in .npy format.
- `python 02_encode_data.py --config config/encode/[YYYYMMDD].yaml`

**`03_train_predictor.py`**
**`04_assess_accuracy_repeat.py`**
**`05_assess_accuracy_nonrepeat.py`**
**`06_run_trajectory_experiment.py`**
### files
- workflow.md describes the workflow from scripts to file

## config/

### connection/
Purpose: store database information, credentials, and SSH key (gitignore due to sensitive information)

### pull/*.yaml
Purpose: config files for pulling from the Constant Therapy database

Notes: date the data pulled in YYYYMMDD, contains sql query used as well as output data file path.

### preprocess/*yaml
Purpose: config files for preprocessing data (filter for usage time and frequency), source from data/raw/ and destination to data/preprocessed/

Notes: date the data is processed in YYYYMMDD, contains filter parameters, source, and output destination

### encode/*yaml
Purpose: config files for encoding data (missing indicators for scores and 0-ing out for targets), source from data/preprocessed/ and destination to data/encoded/

Notes: date the data is processed in YYYYMMDD, contains source and output destination

## data/
Notes: CSV files should be saved without index column

### raw/
Anything that is directly pulled from the Constant Therapy SQL database using 00_pull_data.py

### preprocessed/
Data that is preprocessed using 01_preprocess_data.py, source from raw/
- 20250530: no datetime outlier filter implemented (commit 20250530 936aa9d29c65b1effce55303ab122f6e4c0130a3)

### encoded/
Data that has been encoded using 02_encode_data.py, source from preprocessed/
- 20250609: no target encoding implemented (commit 20250609 138a93f5d8603b018d46fbac6e1a60b935494a19)

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
- `encoding.py`: contains functions that create missing indicators and encode target vectors


### experiments/

### training/
- `training.py`
- `model.py`
- `hyperparameter.py`
- `pipeline.py`
- `evaluation.py`

## utils/

## viz/