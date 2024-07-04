# Getting Started

To get started, you will need to install the repository, download the data and run the `extract_data` and `prepare_data` scripts to allow for model training and evaluation.

## Installation
Refer to [README](https://github.com/nhsengland/mm-healthfair/tree/main) for installation instructions. Recommended to use `poetry` to avoid compatibility issues.

## Data Curation
### 0. Downloading the data
The MIMIC-IV dataset (v2.2) can be downloaded from [PhysioNet](https://physionet.org). This project made use of three modules:

- Hosp: hospital level data for patients: labs, micro, and electronic medication administration
- ED: data from the emergency department
- Notes: deidentified free-text clinical notes.

Steps to download:

1. Create an account on [PhysioNet](https://physionet.org).
2. Complete the required training and credentialisation.
    - Whilst MIMIC-IV is an open dataset, training and credentialisation is required to access and download the zip files. Please refer to the [PhysioNet](https://physionet.org) webpages for instructions on how to gain access.
3. Download the data.


### 1. Extracting the data
`extract_data.py` reads from the downloaded MIMIC-IV files and generates a filtered list of hospital stays (`stays.csv`), time-series events (`events.csv`) and optionally discharge summary notes (`notes.csv`). These entries are filtered and matched according to the hospital admission identifier `hadm_id`, corresponding to a single transfer from the emergency department to the hospital.

```
extract_data.py [-h] --output_path OUTPUT_PATH [--event_tables EVENT_TABLES [EVENT_TABLES ...]]
                    [--include_notes] [--labitems LABITEMS] [--verbose] [--sample SAMPLE]
                    [--lazy] mimic4_path
```

- `mimic4_path`: [Required] Path to root directory containing downloaded `mimiciv/` and `mimic-iv-ed/` subfolders.
- `--output_path` or `-o`: Path to a folder to store filtered csv files.
- `--event_tables` or `-e`: Which MIMIC-IV event files to read from. Options are one or more from ["labevents", "vitalsign"]
- `--include_notes` or `-n`: Flag to determine whether to extract relevant notes.
- `--labitems` or `-i`: Path to a text file containing a list of labitem IDs to speed up and reduce number of events. Highly recommended due to amount of missingness. Refer to MIMIC  [documentation](https://mimic.mit.edu/docs/iv/modules/hosp/labevents/) for more information. Note: A list of the 20 most commonly reported labitems was used and is available in the [project repository](https://github.com/nhsengland/mm-healthfair/blob/main/src/utils/labitems.txt).
- `--verbose` or `-v`: Controls verbosity. Can cause significant overhead when reading from entire directory since it forces all steps to run in eager mode, creating a bottlneck.
- `--sample` or `-s`: Integer number of samples (stays) to extract. Can be used to testing.
- `--lazy`: Whether to run in lazy mode where possible to minimise RAM. Refer to the [Polars](https://docs.pola.rs/py-polars/html/reference/lazyframe/index.html) documentation for more information on lazy mode.

Example:

```
extract_data.py /path/to/data/  -o /data/extracted_data --event_tables labevents vitalsign --sample 1000 --lazy --labitems labitems.txt
```

This command will read from the `/data/mimic` directory and extract relevant events from the MIMIC-IV hosp/labevents.csv and ed/vitalsign.csv for 1000 stays. Output files will be saved under the `/data/extracted_data` folder.

### 2. Preparing the data
Once these core files have been downloaded, run `prepare_data.py` to create a dictionary (key = hadm_id, values = dataframes of preprocessed data) for model training and evaluation. The resulting file will be called **`processed_data.pkl`** and is required for downstream analysis scripts.

Usage:
```
prepare_data.py [-h] [--output_dir OUTPUT_DIR] [--min_events MIN_EVENTS] [--max_events MAX_EVENTS]
                     [--impute IMPUTE] [--no_scale] [--no_resample][--include_dyn_mean]
                     [--max_elapsed MAX_ELAPSED] [--include_notes] [--verbose] data_dir
```

- `data_dir`: [Required] Path to folder containing extracted data.
- `--output_dir`: Path to directory for saving processed data file. If left, will use the same folder as `data_dir`.
- `--min_events`: Filter stays by minimum number of events in any one timeseries. Ensures enough readings available.
- `--max_events`: Filter stays by a maximum number of events in any one timeseries.
- `--impute`: Method to use to impute missing value. Options are `None`, `value` (uses constant value of -1), `mask` (create new feature columns which are booleans indicating missingness - inspired by [Lipton et al. (2016)](http://proceedings.mlr.press/v56/Lipton16.pdf)), `forward` (fill), `backward` (fill)
- `--no_scale`: Flag to disable min/max scaling of continuous variables.
- `--no_resample`: Flag to disable resampling of time-series to fixed frequency (underlying assumption of LSTM).
- `--include_dyn_mean`: Flag to include mean value of time-series features as additional static features.
- `--max_elapsed`: Number of hours (int) to filter time-series (after hospital admission time).
- `--include_notes`: Whether to include notes in processed data file. Requires `notes.csv` to be present in extracted `data_dir`. See `extract_data.py`.
- `--verbose` or `-v`: Controls verbosity.

Example:

```
prepare_data.py /data/extracted_data --min_events 5 --impute value --max_elapsed 24 --include_notes
```

This will process the filtered stays, events and notes csv files in `/data/extracted_data/` and output a single .pkl file at `/data/extracted_data/processed_data.pkl`. This will be a dictionary with `hadm_id` keys and three dataframes as values for each: `static`, `dynamic`and `notes`. This is the file that will be parsed to model training and evaluation scripts.

## Model Development

### 1. Create data splits
To avoid data leakage, it is good practice to split up the data into training, validation and test at the start of experimentation. This ensures that the test set can be held-out until all model development and exploration has been finalised so that fully trained models can be fairly compared. `create_train_test.py` takes in a `processed_data.pkl` file and generates .txt files containing `hadm_id`'s for trainining (70%), validation (10%) and testing (20%). Optionally, stratification can be applied to balance the splits based on the length-of-stay label.

Usage:
```
create_train_test.py [-h] [--output_dir OUTPUT_DIR] [--suffix SUFFIX] [--stratify] [--thresh THRESH] [--seed SEED] data_dir
```
- `data_dir`: [Required] Path to folder containing processed .pkl file.
- `--output_dir`: Path to save generated .txt files.
- `--suffix`: Suffix to append to generated .txt files. Useful for avoiding overwriting or generating multiple sets for repeatibility.
- `--stratify`: Flag to turn on stratification.
- `--thresh`: Threshold to use for stratification based on length-of-stay label (days). Defaults to 2.
- `--seed`: Seed to use for reproducibility during application of random sample.


### 2. Training a model



### 3. Evaluating a model

## General Tips
