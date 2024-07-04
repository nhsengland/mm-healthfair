# *MM-HealthFair*: Understanding Fairness and Explainability in Multimodal Approaches within Healthcare

A crucial component to facilitating the adoption of AI in healthcare lies in the ability to justify, interpret, and effectively communicate the decisions of the system.
This project aims to explore fairness and explainability in the context of multimodal AI for healthcare. We investigate a the combination of modalities and evaluate approaches to multimodal modelling in terms of predictive accuracy, explainability and fairness. Specifically we address the following research questions:

1. How does the incorporation of time-dependant information impact the model?
2. What is the impact of fusion strategies on explainability and fairness?
3. What mitigation strategies can we apply to reduce bias against protected characteristics?

Specifically, we focus on a case study for **length-of-stay prediction** in hospital following admission to the emergency department. This is important for hospital management, ensuring sufficient resources are available for patients who may require long-term monitoring, as well as helping to support effective and efficient triage for those who may require urgent assistance without needing long-term stay.

In this project, we considered the fusion of information across **static demographic features, time-series events and medical history extracted from discharge summaries** in a multimodal framework that aims to capture modality-specific features and inter-modality relationships. We evalute our models based on performance as well as **fairness** across a set of protected characteristics such as gender, race, marital status and insurance types. Importantly, we address how **bias within the data** can lead to differences in the fairness of a preditive model for different subgroups and how this can be **mitigated** to ensure demographic parity. We also explore how the choice of modelling approach can amplify or reduce these effects.


## Data curation
There are several pipelines available for reading, extracting data from the MIMIC dataset. However, due to the changes in structure with dataset revisions, many of these go out-of-date and it was not straight-forward to adapt them to MIMIC-IV v2.2. Additionally, the introduction of emergency department records and vitalsigns was newly introduced in MIMIC-IV. Existing work often made use of events tables from the hospital and ICU departments only. Moreover, whilst there have been studies exploring the use of MIMIC for multimodal modelling, many have focused on the use of clinical notes or chest-x-rays alongside electronic health data. Few studies have considered the use of **time-series data as a seperate modality** whilst making the dataset and models available for use in further analysis. Therefore, this project required the development of a data extraction and preprocessing pipeline specifically for extracting relevant data for emergency department and hospital admissions:

1. `extract_data.py`: Reads and filters relevant hospital stay data from downloaded MIMIC-IV files.
2. `prepare_data.py`: Cleans, preprocesses and filters stays into a single .pkl file for downstream analysis.

## Model training
In this project, we additionally include scripts to train and evaluate different models on the dataset.

1. `create_train_test.py`: Generates a (balanced, stratified) list of training, validation and test ids for training, development and testing.
2. `train.py`: Script to train a neural network for LOS prediction. Option to log and save models using [Weights & Biases](https://wandb.ai)
3. `train_rf.py`: Script to train a Random Forest classifier for LOS prediction.

Training configurations are specified in a config file. See `example_config.toml` for available settings.

## Model evaluation
Once models have been trained and saved, we also include the scripts used to compare their performance, generate explanations and quantify fairness across protected attributes. Moreover, the [Fairlearn](https://fairlearn.org/) package used for fairness evaluation is also used to explore mitigation strategies in `postprocess.py`

1. `evaluate.py`: Evaluate a trained model's performance. Generates explainations with `--explain` and/or fairness metrics and plots with `--fairness`.
2. `postprocess.py`: Run Fairlearn's [Threshold Optimizer](https://fairlearn.org/v0.10/user_guide/mitigation/postprocessing.html) to mitigate bias for a sensitive attribute.
