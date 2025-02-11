import os
import pandas as pd

DATA_DIR = "data"
DATA_NAME = "SAS_Data_Dictionary_FUP226_raw-01Jan2023_v1.xlsx"
DATA_PATH = os.path.join(DATA_DIR, DATA_NAME)


def main():
    """ Explore data
    """
    data_dict = load_data(DATA_PATH)
    for i, (sheet_name, df) in enumerate(data_dict.items()):
        print_df_info(df, sheet_name)
        

def load_data(data_path: str) -> dict[pd.DataFrame]:
    """ Loads all sheets from excel file into a dictionary of DataFrames
    """
    return pd.read_excel(data_path, sheet_name=None)  # Load all sheets


def print_df_info(df: pd.DataFrame, name: str):
    """ Print basic information about a DataFrame
    """
    print(f"\n\n\nSheet name: {name}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns}")
    print(f"Head: {df.head()}")


if __name__ == "__main__":
    main()