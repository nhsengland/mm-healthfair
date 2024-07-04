# Getting Started

To get started, you will need to install the repository, download the data and run the `extract_data` and `prepare_data` scripts for model training and evaluation.

## Installation
Refer to [README](https://github.com/nhsengland/mm-healthfair/tree/main) for installation instructions. Recommended to use `poetry` to avoid compatibility issues.

## Running the scripts
All scripts are best executed via the command-line in order to specify the required and optional arguments.

- If you have created your own virtual environment with dependencies specified by `requirements.txt` then you should run scripts with:

```sh
source activate venv
python3 src/any_script.py [required args] --[optional args]
```

- If using `poetry` (recommended):
```sh
cd mm-healthfair
poetry run python3 src/any_script.py [required args] --[optional args]
```

## Data Curation
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

```sh
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

```sh
extract_data.py /path/to/data/  -o /data/extracted_data --event_tables labevents vitalsign --sample 1000 --lazy --labitems labitems.txt
```

This command will read from the `/data/mimic` directory and extract relevant events from the MIMIC-IV hosp/labevents.csv and ed/vitalsign.csv for 1000 stays. Output files will be saved under the `/data/extracted_data` folder.

### 2. Preparing the data
Once these core files have been downloaded, run `prepare_data.py` to create a dictionary (key = hadm_id, values = dataframes of preprocessed data) for model training and evaluation. The resulting file will be called **`processed_data.pkl`** and is required for downstream analysis scripts.

Usage:
```sh
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

```sh
prepare_data.py /data/extracted_data --min_events 5 --impute value --max_elapsed 24 --include_notes
```

This will process the filtered stays, events and notes csv files in `/data/extracted_data/` and output a single .pkl file at `/data/extracted_data/processed_data.pkl`. This will be a dictionary with `hadm_id` keys and three dataframes as values for each: `static`, `dynamic`and `notes`. This is the file that will be parsed to model training and evaluation scripts.

## Model Development

### 1. Create data splits
To avoid data leakage, it is good practice to split up the data into training, validation and test at the start of experimentation. This ensures that the test set can be held-out until all model development and exploration has been finalised so that fully trained models can be fairly compared. `create_train_test.py` takes in a `processed_data.pkl` file and generates .txt files containing `hadm_id`'s for trainining (70%), validation (10%) and testing (20%). Optionally, stratification can be applied to balance the splits based on the length-of-stay label.

Usage:
```sh
create_train_test.py [-h] [--output_dir OUTPUT_DIR] [--suffix SUFFIX] [--stratify] [--thresh THRESH] [--seed SEED] data_dir
```
- `data_dir`: [Required] Path to folder containing processed .pkl file.
- `--output_dir`: Path to save generated .txt files.
- `--suffix`: Suffix to append to generated .txt files. Useful for avoiding overwriting or generating multiple sets for repeatibility.
- `--stratify`: Flag to turn on stratification.
- `--thresh`: Threshold to use for stratification based on length-of-stay label (days). Defaults to 2.
- `--seed`: Seed to use for reproducibility during application of random sample.

Example:

```sh
create_train_test.py /data/extracted_data/ --stratify --suffix _stratified --thresh 2
```

### 2. Training a model
To train a model use either `train.py` or `train_rf.py` for fusion/deep models and Random Forest training respectively.

Usage:

```sh
train.py [-h] [--config CONFIG] [--cpu] [--train [TRAIN]] [--val [VAL]] [--project PROJECT] [--wandb] data_path
```

- `data_path`: [Required] Path to the processed data .pkl file.
- `--config`: Path to a .toml file containing settings. Refer to `example_config.toml` for available settings.
- `--cpu`: Whether to use cpu for training. Defaults to gpu.
- `--train`: Path to a text file containing `hadm_ids` to use for training.
- `--val`: Path to a text file containing `hadm_ids` to use for validation.
- `--project`: Name of project to save runs on wandb. Defaults to 'nhs-mm-healthfair'.
- `--wandb`: Flag to enable wandb logging to project.

**Note**: If `--wandb` is set, then you will need to login on the first run. Refer to the [wandb + lightning docs](https://docs.wandb.ai/guides/integrations/lightning#sign-up-and-log-in-to-wandb) for more details.

Example:

```sh
train.py /data/extracted_data/processed_data.pkl --train /data/extracted_data/training_ids.txt --val /data/extracted/val_ids.txt --project my_project --wandb
```

Saved model checkpoints (or for Random Forest .pkl files) will be logged to `logs/` folder and online via Weights & Biases if `--wandb` is set (deep models only).

For running experiments on different models and datasets, you can create different processed data files by running `prepare_data.py` with a variety of command-line arguments. For example, you may wish to explore different imputation methods with `--impute` or remove the resampling of time-series data to only use the exact frequency of events `--no_resample`.

Currently two fusion strategies are implemented: `mag` and `concat` which can be specified in the `config.tml` file. Model architectures are defined in `models.py` and built with Pytorch and [Pytorch Lightning](https://lightning.ai) as `LightingModule`'s for simplified training and evaluation loops, callbacks and logging. Refer to their docs for more information on the `LightningModule ` object. You can easily create new architectures or fusion strategies by editing parameters or adding new model classes to the `models.py` file.

Examples:

- To test a model with time-series as the primary modality using the `mag` fusion method, you should set `st_first=False` in the `config.toml` file (otherwise uses default value of True).
- To implement a new fusion method, create another `LightningModule` subclass in `models.py` and edit `train.py` to use your new model.


### 3. Evaluating a model
Once you have trained and saved a model you can run inference on a set of test cases, evaluate the model's fairness and/or produce explanations using the `evaluate.py` script.

Usage:
```sh
 evaluate.py [-h] [--config CONFIG] [--metadata METADATA] [--test TEST] [--explain] [--fairness] data_dir model_path
```

- `data_dir`: [Required]
- `model_path`: [Required] Path to a .pkl (Random Forest) or .ckpt file.
- `--metadata`: Path to the `stays.csv` file outputted by `extract_data.py`. Only required if not under the `data_dir` directory.
- `--test`: Path to text file containing `hadm_id`'s for testing.
- `--explain`: Flag to enable use of **SHAP** for model explanations.
- `--fairness`: Flag to enable use of **FairLearn** to compute fairness metrics and visualise plots.

**Model explanations**: The [SHAP](https://shap.readthedocs.io/en/latest/) package has been used.

**Model fairness**: The [FairLearn](https://fairlearn.org) package has been used to compute fairness metrics across sensitivie/protected attributes. For example, in the MIMIC-IV dataset these include: sex, ethnicity, marital status and insurance type.

## General Usage Tips
