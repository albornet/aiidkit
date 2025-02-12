import pandas as pd
from process_data_utils import ProcessDataConfig as cfg

DEBUG = True


def main() -> None:
    """ ...
    """
    raw_data_dict: dict[str, pd.DataFrame]
    raw_data_dict = pd.read_excel(cfg.RAW_DATA_PATH, sheet_name=None)
    process_data(data_dict=raw_data_dict)


def process_data(data_dict: dict[str, pd.DataFrame]) -> ...:
    """ ...
    """
    pat_ids_with_consent = get_pat_ids_with_consent(data_dict)
    for sheet_name, df in data_dict.items():
        if is_part_of_data(sheet_name):
            print(f"Processing {sheet_name} sheet")
            df = base_filtering(df, pat_ids_with_consent)


def get_pat_ids_with_consent(data_dict: dict[str, pd.DataFrame]) -> pd.Series:
    """ Identify patients who gave their consent for retrospective analysis
    """
    consent_df = data_dict["#1_CONSENT"]
    consent_given = consent_df["consstatus"] == "Consent given"
    pat_ids_with_consent = consent_df[consent_given]["patid"]
    
    return pat_ids_with_consent


def is_part_of_data(sheet_name: str):
    """ Check whether a sheet is proceseed by the data pipeline, by sheet name
    """
    if sheet_name[0] != "#" or "_freq" in sheet_name or "#1_" in sheet_name:
        return False
    
    # DEBUG (TO REMOVE)
    if DEBUG and sheet_name not in ["4_PAT_BL", "#5_PAT_PSQ"]:
        return False
    # DEBUG (TO REMOVE)
    
    # Might add more conditions
    return True


def base_filtering(
    df: pd.DataFrame,
    pat_ids_with_consent: list[int],
    stat_rows: list[str]=cfg.STAT_ROW_NAMES,
) -> pd.DataFrame:
    """ Base filtering that is common to all dataframes
    """
    df = df[~df["_STAT_"].isin(stat_rows)]  # unnecessary stat rows below patient info
    df = df.dropna(subset=["_STAT_"])  # remove remaining blank rows around stat rows
    df = df.drop(columns=["_STAT_"])
    df = df[df["patid"].isin(pat_ids_with_consent)]  # keep only patients with consent
    
    import ipdb; ipdb.set_trace()
    return df


def print_df_info(df: pd.DataFrame) -> None:
    """ Print some basic info from any dataframe
    """
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns}")
    print(f"Head: {df.head(5)}")
    

if __name__ == "__main__":
    main()