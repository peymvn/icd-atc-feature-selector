"""
By: Peyman Ghasemi
Please cite as:
Ghasemi P, Lee J
Unsupervised Feature Selection to Identify Important ICD-10 and ATC Codes for Machine Learning: A Case Study on a Coronary
Artery Disease Patient Cohort
JMIR Med Inform 2024;0:e0
URL: https://medinform.jmir.org/2024/0/e0/
doi: 10.2196/52896


"""

import simple_icd_10 as icd
import numpy as np
import pandas as pd
import os
import re
from datetime import datetime
import constants as cons
import argparse


def process_and_list_all_icd_codes(data_path: str):
    """
    This function reads the dataset and lists all the unique ICD codes in the dataset. It also finds the ancestors of
    each code and saves them in a csv file. It also checks if the code is a Canadian ICD code or not.
    If you are not using the Canadian ICD-10 codes, you can replace the file path in the constants.py file with your own
    local ICD-10 code list or leave it empty if you only want to use the WHO ICD-10 codes.
    :param data_path: path to the dataset (csv file containing the ICD codes in the columns for each sample)
    :return: None (it saves the list of ICD codes in the processed_data folder)
    """
    # create required folders
    os.makedirs(cons.PROCESSED_DIR, exist_ok=True)
    os.makedirs(cons.SELECTED_FEATURES_DIR, exist_ok=True)

    # read the dataset
    df = pd.read_csv(data_path, dtype=str)
    canadian_icd_10 = pd.read_csv(cons.CANADIAN_ICD10_LIST_PATH)
    CODE_COLS = [
        cons.CODE_COL_PREFIX + str(i) for i in range(1, cons.MAX_NUM_CODE_COLS + 1)
    ]
    disease_df = df[CODE_COLS]
    unique_codes = np.unique(disease_df.to_numpy().astype(str))
    all_codes_list = pd.DataFrame(
        columns=["code", "parent", "rank", "chapter", "description", "is_canadian"]
    )

    def get_canadian_codes_ancestors(icd_code: str):
        """
        This code removes the least valued digit from the ICD code until finds its parent in the main ICD codes
        It is useful to detect Canadian ICD codes. In cases that the code is related to the morphology of neoplasm, it
        handles it differently.
        :param icd_code:
        :return:
        """

        # check if the code consists of just numbers
        pattern = re.compile(r"^\d+$")
        if bool(pattern.match(icd_code)) and (len(icd_code) > 3):
            return ["NEOPLASM"]

        # General Cases
        # drop last digit and check general ICD-10
        new_icd_code = icd_code[:-1]
        if icd.is_valid_item(new_icd_code):
            ancestors = icd.get_ancestors(new_icd_code)
            ancestors.insert(0, new_icd_code)
        else:
            if len(new_icd_code) != 0:
                ancestors = get_canadian_codes_ancestors(new_icd_code)
                # ancestors.insert(0, new_icd_code)
            else:
                ancestors = []

        return ancestors

    # find ancestors of the unique code and add them to the code list
    def add_to_codes_list(icd_code):
        nonlocal all_codes_list
        if icd.is_valid_item(icd_code):
            ancestors = icd.get_ancestors(icd_code)

            if not (icd_code in all_codes_list["code"].values):
                code_data = {
                    "code": icd_code,
                    "parent": (
                        None if len(ancestors) == 0 else ancestors[0]
                    ),  # to make sure the list is not empty
                    "rank": len(ancestors),
                    "chapter": (
                        icd_code if len(ancestors) == 0 else ancestors[-1]
                    ),  # to make sure the list is not empty
                    "description": icd.get_description(icd_code),
                    "is_canadian": False,
                }
                all_codes_list = pd.concat(
                    [all_codes_list, pd.DataFrame([code_data])], ignore_index=True
                )
                # Now do the same thing for each ancestor
                for ancestor in ancestors:
                    add_to_codes_list(ancestor)

        else:
            # check if the code is in canadian list
            if icd_code in canadian_icd_10["DX_CD"].values:
                ancestors = get_canadian_codes_ancestors(icd_code)
                if not (icd_code in all_codes_list["code"].values):
                    code_data = {
                        "code": icd_code,
                        "parent": (
                            None if len(ancestors) == 0 else ancestors[0]
                        ),  # to make sure the list is not empty
                        "rank": len(ancestors),
                        "chapter": (
                            icd_code if len(ancestors) == 0 else ancestors[-1]
                        ),  # to make sure the list is not empty
                        "description": canadian_icd_10.loc[
                            canadian_icd_10["DX_CD"] == icd_code, "DX_DESC"
                        ].values[0],
                        "is_canadian": True,
                    }
                    all_codes_list = pd.concat(
                        [all_codes_list, pd.DataFrame([code_data])], ignore_index=True
                    )
                    # Now do the same thing for each ancestor
                    for ancestor in ancestors:
                        add_to_codes_list(ancestor)

    for code in unique_codes:
        add_to_codes_list(code)

    # print(all_codes_list)
    all_codes_list.to_csv(
        os.path.join(cons.PROCESSED_DIR, "all_codes_list.csv"), index=False
    )
    print("Listing the ICD Codes Done...")


def find_atc_ancestors(atc_code: str, atc_code_list: list[str]) -> list[str]:
    """
    Find all ancestors of a given ATC code. It includes the code itself.
    :param atc_code: the ATC code. Case sensitive.
    :param atc_code_list: the list containing all ATC codes.
    :return: a list of ancestors of the given ATC code (including the code itself).
    """
    ancestors = []
    code_lengths = [1, 3, 4, 5, 7]

    if not (1 <= len(atc_code) <= 7):
        raise Exception(f"Invalid ATC code: {atc_code}")

    for i, length in enumerate(code_lengths):
        if len(atc_code) == length:
            if atc_code in atc_code_list:
                ancestors.append(atc_code)
                if length > 1:
                    ancestors.extend(
                        find_atc_ancestors(
                            atc_code[: code_lengths[i - 1]], atc_code_list
                        )
                    )
                return ancestors
            else:
                raise Exception(f"Invalid ATC code: {atc_code}")

    raise Exception(f"Invalid ATC code: {atc_code}")


def process_and_list_all_atc_codes(data_path: str):
    """
    This function reads the dataset and lists all the unique ATC codes in the dataset. It also finds the ancestors of
    each code and saves them in a csv file.
    :param data_path: path to the dataset (csv file containing the ATC codes in the columns for each sample)
    :return: None (it saves the list of ATC codes in the processed_data folder)
    """
    # create required folders
    os.makedirs(cons.PROCESSED_DIR, exist_ok=True)
    os.makedirs(cons.SELECTED_FEATURES_DIR, exist_ok=True)

    # read the dataset
    df = pd.read_csv(data_path, dtype=str)

    # read list of ATC codes
    atc_df = pd.read_csv(cons.ATC_LIST_PATH, dtype=str)

    CODE_COLS = [
        cons.CODE_COL_PREFIX + str(i) for i in range(1, cons.MAX_NUM_CODE_COLS + 1)
    ]
    drugs_df = df[CODE_COLS]
    unique_codes = np.unique(drugs_df.to_numpy().astype(str))

    # find all unique ATC codes in the dataset with different levels
    all_atc_codes = []
    parent_dic = {}
    for code in unique_codes:
        try:
            ancestors = find_atc_ancestors(code, atc_df["atc_code"].to_list())
            all_atc_codes.extend(ancestors)
            if len(ancestors) > 1:
                for i in range(len(ancestors) - 1):
                    parent_dic[ancestors[i]] = ancestors[i + 1]

        except:  # invalid code
            print(f"Invalid ATC code: {code}")
            pass

    all_atc_codes = np.unique(all_atc_codes)
    all_atc_codes = np.sort(all_atc_codes)
    all_atc_parents = [
        parent_dic[code] if code in parent_dic else "" for code in all_atc_codes
    ]
    all_atc_chapters = [code[0] for code in all_atc_codes]
    all_atc_descriptions = [
        atc_df[atc_df["atc_code"] == code]["atc_name"].values[0]
        for code in all_atc_codes
    ]
    all_atc_rank = [
        float(atc_df[atc_df["atc_code"] == code]["rank"].values[0])
        for code in all_atc_codes
    ]

    all_codes_list = pd.DataFrame(
        {
            "code": all_atc_codes,
            "parent": all_atc_parents,
            "rank": all_atc_rank,
            "chapter": all_atc_chapters,
            "description": all_atc_descriptions,
        }
    )

    all_codes_list.to_csv(
        os.path.join(cons.PROCESSED_DIR, "all_codes_list.csv"), index=False
    )
    print("Listing the ATC Codes Done...")


def get_hierarchical_structure(code: str, all_codes_df: pd.DataFrame):
    """
    find the ancestors of a given code by looking to its parents
    :param all_codes_df:
    :param code:
    :return: a list containing the ancestors (until it reaches the chapter name)
    """
    hierarchy_list = []
    while code != "":
        hierarchy_list.append(code)
        try:
            code = all_codes_df.loc[all_codes_df["code"] == code, "parent"].values[0]
        except:
            break

    return hierarchy_list


def one_hot_encode_icd_codes(
    codes: list[str], one_hot_col_list: list[str], ancestor_dic: dict
):
    """
    This function encodes the ICD/ATC codes in a one-hot format.
    :param codes: a list of ICD/ATC codes
    :param one_hot_col_list: a list of all ICD/ATC codes
    :param ancestor_dic: a dictionary containing the ancestors of each ICD/ATC code
    :return: a one-hot encoded row
    """

    one_hot_row = pd.Series(0, index=one_hot_col_list)
    codes_to_encode = []
    for code in codes:
        try:
            hierarchy_list = ancestor_dic[code]
            codes_to_encode = codes_to_encode + hierarchy_list
        except KeyError as e:
            print("Wrong Code - KeyError:", e)
            continue

    one_hot_row.loc[codes_to_encode] = 1

    return one_hot_row


def apply_one_hot_encoding_on_dataset(data_path: str):
    all_codes_df = pd.read_csv(
        os.path.join(cons.PROCESSED_DIR, "all_codes_list.csv"), na_filter=False
    )
    one_hot_col_list = all_codes_df["code"].values

    # generate a lookup table of the ancestors of each code (for optimization)
    ancestor_dic = {}
    for code in all_codes_df["code"].values:
        ancestor_dic[code] = get_hierarchical_structure(code, all_codes_df)
    print("Ancestors Lookup Table Created...")

    # read the length of the dataset
    with open(data_path) as f:
        num_of_samples = sum(1 for line in f) - 1

    # Start the process - Read the sorted dataset with chunks (as the one-hot-encoded df may be too large to fit in memory)
    one_hot_encoded_data = []
    DISEASE_CODE_COLS = [
        cons.CODE_COL_PREFIX + str(i) for i in range(1, cons.MAX_NUM_CODE_COLS + 1)
    ]
    print("Start Time =", datetime.now().strftime("%H:%M:%S"))
    chunksize = 5000
    df_iterator = pd.read_csv(data_path, chunksize=chunksize, dtype=str)

    for i, df in enumerate(df_iterator):
        df_icd = df[DISEASE_CODE_COLS]
        one_hot_df = df_icd.apply(
            lambda row: one_hot_encode_icd_codes(
                row.dropna().tolist(), one_hot_col_list, ancestor_dic
            ),
            axis=1,
        )

        # Append the chunk of one-hot encoded data to the list
        one_hot_encoded_data.append(one_hot_df.values)
        print(
            f"End Time for chuck {i} - {(i + 1) * chunksize}/{num_of_samples} ({round(100*(i + 1) * chunksize / num_of_samples, 2)}%) :",
            datetime.now().strftime("%H:%M:%S"),
        )

    # Concatenate the list of one-hot encoded dataframes
    one_hot_encoded_data = np.concatenate(one_hot_encoded_data, axis=0)

    # Save the one-hot encoded data
    np.save(
        os.path.join(cons.PROCESSED_DIR, "one_hot_encoded_data.npy"),
        one_hot_encoded_data,
    )
    print("One-Hot-Encoding Done...")


def main(data_path: str, process_atc: bool):
    if process_atc:
        process_and_list_all_atc_codes(data_path)
    else:
        process_and_list_all_icd_codes(data_path)
    apply_one_hot_encoding_on_dataset(data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ICD codes in the dataset")
    parser.add_argument(
        "data_path",
        type=str,
        help="path to the dataset (csv file containing the ICD codes in the columns for each sample)",
    )
    parser.add_argument(
        "--atc", action="store_true", help="process ATC codes instead of ICD codes"
    )
    args = parser.parse_args()
    main(args.data_path, args.atc)
