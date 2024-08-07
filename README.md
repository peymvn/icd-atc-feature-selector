# Unsupervised & Supervised Feature Selection to Identify Important ICD-10 and ATC Codes for Machine Learning

This repository contains the code for the best feature selection method introduced in the paper titled "[_Unsupervised Feature Selection to Identify Important ICD-10 and ATC Codes for Machine Learning on a Cohort of Patients With Coronary Heart Disease: Retrospective Study_](https://medinform.jmir.org/2024/1/e52896/)".
Please cite the following paper if you use this code in your research (and also the [original Concrete Autoencoder paper](https://arxiv.org/abs/1901.09346)):

    @article{ghasemi2024unsupervised,
    title={Unsupervised Feature Selection to Identify Important ICD-10 and ATC Codes for Machine Learning on a Cohort of Patients With Coronary Heart Disease: Retrospective Study},
    author={Ghasemi, P. and Lee, J.},
    journal={JMIR Med Inform},
    year={2024},
    volume={12},
    pages={e52896},
    url={https://medinform.jmir.org/2024/1/e52896/},
    doi={10.2196/52896}
    }

This code contains only a user-friendly implementation of ICD/ATC feature selection using Concrete Autoencoders with hierarchical weight adjustment. If you wish to see the full codes of the experiments of the paper, please refer to the following repository:

https://github.com/data-intelligence-for-health-lab/ICD10-Unsupervised-Feature-Selection

## Requirements
    - Python 3.9
    - Packages included in requirements.txt

## Notes
1. The Concrete Autoencoder code is based on the [original implementation](https://github.com/mfbalin/Concrete-Autoencoders)

2. If you are using the Canadian version of ICD-10 codes (ICD-10-CA), you need to place them in the repository (in `ICD_10_CA_DX.csv`). [Canadian Institute for Health Information](https://secure.cihi.ca/estore/productSeries.htm?pc=PCC84) sells them to researchers and I was not allowed to share them (although I believe that it is inappropriate for a publicly funded organization to sell something like that to researchers). I also shared `ICD_10_CA_DX_EXAMPLE.csv` which contains 3 examples of the ICD-10-CA codes so that you do not buy a wrong format.

3. If you are using just the WHO version and not the Canadian version, just leave `ICD_10_CA_DX.csv` empty (or add your desired local ICD codes).

4. Two sample CSV files are shared in `test_data` folder to show how you need to prepare your data for feature selection.

## How to Run
- Install the required packages using `pip install -r requirements.txt`
- Download the ICD-10-CA codes and replace them with the sample file (if you wish)
- Adjust you required settings in `constants.py`
- Run using:
#
    # process ICD-10 codes and save the required files
    python process_codes.py [PATH_TO_DATA_CSV_FILE]

    # process ATC codes and save the required files
    python process_codes.py [PATH_TO_DATA_CSV_FILE] --atc

    # Selecting the codes
    python code_selection_concrete_ae_with_weights.py

Adjust the settings according to your requirements. These codes will make two folders `processed_data` and `selected_features` to save the one-hot-encoded data and the selected features lists.