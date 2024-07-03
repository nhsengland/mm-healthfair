# Understanding Fairness and Explainability in Multimodal Approaches within Healthcare
## NHSE PhD Internship Project

### About the Project

[![status: experimental](https://github.com/GIScience/badges/raw/master/status/experimental.svg)](https://github.com/GIScience/badges#experimental)
[![PyPI - License](https://img.shields.io/pypi/l/nhssynth)](https://github.com/nhsengland/nhssynth/blob/main/LICENSE)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/nhsengland/mm-healthfair/main.svg)](https://results.pre-commit.ci/latest/github/nhsengland/mm-healthfair/main)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)

This repository holds code for the Understanding Fairness and Explainability in Multimodal Approaches within Healthcare project.
See the [original project propsoal](https://nhsx.github.io/nhsx-internship-projects/advances-modalities-explainability/) for more information.

_**Note:** Only public or fake data are shared in this repository._

### Project Stucture

- The main code is found in the root of the repository (see Usage below for more information)
- The accompanying [report](./reports/report.pdf) is also available in the `reports` folder
- More information about the code usage can be found in the [model card](./model_card.md)

### Built With

[![Python v3.10](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/downloads/release/python-3100/)

### Getting Started

#### Installation

To get a local copy up and running follow these simple steps.

To clone the repo:

`git clone https://github.com/nhsengland/mm-healthfair`

To create a suitable environment:

1. Use pip + requirements.txt
- ```python -m venv _env```
- `source _env/bin/activate`
- `pip install -r requirements.txt`

2. Use poetry (*recommended*)
- Install poetry (see [website](https://python-poetry.org) for documentation)
- Navigate to project root directory `cd mm-healthfair`
- Create environment from poetry lock file: `poetry install`
- Run scripts using `poetry run python3 xxx.py`

### Usage
This repository contains code used to extract and preprocess demographic, time-series and clinical notes from MIMIC-IV v2.2. Additionally, it includes the model architectures and training scripts used to train multimodal models on different modalities and generate the results described in the report.

#### Outputs
- Preprocessed features from MIMIC-IV 2.2
- Trained models
- Notebook exploring the dataset and visualising results

Seeds have been set to reproduce the results in the report.

#### Datasets
The MIMIC-IV dataset (v2.2) can be downloaded from [PhysioNet.org](https://physionet.org). This project made use of three modules:
- Hosp: hospital level data for patients: labs, micro, and electronic medication administration
- ED: data from the emergency department
- Notes: deidentified free-text clinical notes

Further information can be found in PhysioNet's [documentation](https://mimic.mit.edu/).

### Roadmap

See the repo [issues](https://github.com/nhsengland/mm-healthfair/issues) for a list of proposed features (and known issues).

### Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

_See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidance._

### License

Unless stated otherwise, the codebase is released under [the MIT Licence][mit].
This covers both the codebase and any sample code in the documentation.

_See [LICENSE](./LICENSE) for more information._

The documentation is [© Crown copyright][copyright] and available under the terms
of the [Open Government 3.0][ogl] licence.

[mit]: LICENCE
[copyright]: http://www.nationalarchives.gov.uk/information-management/re-using-public-sector-information/uk-government-licensing-framework/crown-copyright/
[ogl]: http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/

### Contact

To find out more about the [Analytics Unit](https://www.nhsx.nhs.uk/key-tools-and-info/nhsx-analytics-unit/) visit our [project website](https://nhsx.github.io/AnalyticsUnit/projects.html) or get in touch at [england.tdau@nhs.net](mailto:england.tdau@nhs.net).

<!-- ### Acknowledgements -->
