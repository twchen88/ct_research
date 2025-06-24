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

**`02_encode_data.py`**: Takes in the preprocessed data specified in config files in config/encode/, encode the score data with missing indicators and target data to be used for training. Output is stored in data/encoded in .npy format.
- `python 02_encode_data.py --config config/encode/[YYYYMMDD].yaml`

**`03_train_predictor.py`**: Takes in the encoded data specified in the config files in config/train/, train the model according to the hyper parameters specified. Outputs (model, metadata, plots, metrics, etc) are stored in a specified directory in outputs/training_runs/.
- `python 03_train_predictor.py --config config/train/[YYYYMMDD].yaml --tag [few word description of the purpose of the run]`

**`04_aggregate_average.py`**

**`05_simulate_trajectory.py`**


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

### encode/*.yaml
Purpose: config files for encoding data (missing indicators for scores and 0-ing out for targets), source from data/preprocessed/ and destination to data/encoded/

Notes: date the data is processed in YYYYMMDD, contains source and output destination

### train/*.yaml
Purpose: config files for model training, source data from data/encoded, outputs are stored in outputs/training_runs/. A copy of this config file is copied during the run to the corresponding directory for the run for documentation purposes. Also describes the run briefly.

Notes: includes various setings and hyperparameters

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
Numbered by create date.

## outputs/

### training_runs/
Subdirectories are named in the format of `YYYYMMDD_[tag]/`. Tag should describe the purpose or specs of the run in a few words at most. Centralize information about each run in one place.

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
- `file_io.py`: contains functions that read and write files in appropriate formats for model training

Files with `_torch` appended are implemented with Pytorch.
- `training.py`: contains functions that enable model training, including data preparation, loss function, custom dataset and loader, and training loops
- `model.py`: defines preliminary predictor neural network
- `evaluation.py`: contains functions that help evaluate a model, including prediction and error function

## utils/
- `config_loading.py`: loads json or yaml files
- `logger.py`: implements logger functionality
- `reproducibility.py`: sets global seed for reproducbility
- `metadata.py`: helper functions for metadata information

## viz/
- `training.py`: plots for model training