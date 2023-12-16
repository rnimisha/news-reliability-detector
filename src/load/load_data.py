import os

import pandas as pd

current_directory = os.path.abspath("")
os.chdir(current_directory)


def load_fake_data() -> pd.DataFrame:
    """Load fake data csv as add target column as 1

    Returns:
        pd.DataFrame: fake data csv with target
    """
    fake_data = pd.read_csv("data/raw/Fake.csv")
    fake_data["target"] = 1  # true
    return fake_data


def load_true_data() -> pd.DataFrame:
    """Load true data csv as add target column as 0

    Returns:
        pd.DataFrame: true data csv with target
    """
    true_data = pd.read_csv("data/raw/True.csv")
    true_data["target"] = 0  # false
    return true_data


def load() -> pd.DataFrame:
    """Concatenates fake and true data into one dataframe

    Returns:
        pd.DataFrame: concatenated dataframe
    """
    fake_data = load_fake_data()
    true_data = load_true_data()
    dataset = pd.concat([fake_data, true_data]).reset_index(drop=True)
    return dataset
