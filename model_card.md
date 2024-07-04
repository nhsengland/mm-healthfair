# Model Card: Multimodal Deep Fusion Models

## Model Details

The implementation of the multi-modal fusion models within this repository is based on work done by Sophie Martin during an NHSE PhD internship.
Key features include the multi-modal attention gate (MAG+) which was adapted from [this repository](https://github.com/emnlp-mimic/mimic/blob/main/base.py#L136) and is inspired by the work of [Zhao et al.](https://ieeexplore.ieee.org/document/9746536) to integrate multiple data modalities in and end-to-end training regime.

## Model Use

### Intended Use

This model is intended for use in training a classification model for length of stay prediction on the MIMIC-IV dataset.

## Training Data

Data was downloaded from [MIMIC-IV](https://physionet.org/content/mimiciv/2.2/). MIMIC is a publically accessible repository (subject to data usage agreements and mandatory training) containing mulimodal data across healthcare centers in the US.

## Performance and Limitations

## Acknowledgements
