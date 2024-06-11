# Constants for the the data file

# The data file should be a CSV file with the following columns:
# - Patient ID column
# - Diagnosis code columns (e.g., DXCODE1, DXCODE2, ...) containing the ICD-10 codes
# - Other columns

# the format for the diagnosis code columns should be as follows: f"{CODE_COL_PREFIX}{i}" where i is the column number starting from 1
MAX_NUM_CODE_COLS = 25  # Maximum number of diagnosis/drug code columns in the dataset
CODE_COL_PREFIX = (
    "CODE"  # Prefix for the diagnosis/drug code columns (e.g., CODE1, CODE2, ...)
)

# Directories to save the processed data and selected features
PROCESSED_DIR = "processed_data"  # Directory to save the processed data
SELECTED_FEATURES_DIR = "selected_features"  # Directory to save the selected features

# Path to the Canadian ICD-10 code list
# if you are not using the Canadian ICD-10 codes, you can replace this file with your own local ICD-10 code list
# or leave it empty if you only want to use the WHO ICD-10 codes
# for WHO ICD-10 codes, we use simple_icd_10 package to get codes, parents, and descriptions
CANADIAN_ICD10_LIST_PATH = "ICD_10_CA_DX.csv"

# Path to ATC code list
ATC_LIST_PATH = "ATC_list.csv"
