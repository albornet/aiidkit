import os

class DataConfig:
    """ Configuration for the data processing pipeline
    """
    # Paths
    DATA_DIR = "data"
    RAW_DATA_NAME = "SAS_Data_Dictionary_FUP226_raw-01Jan2023_v1.xlsx"
    RAW_DATA_PATH = os.path.join(DATA_DIR, RAW_DATA_NAME)
    SAVE_DIR = os.path.join(DATA_DIR, "processed")
    
    # Features taken/ignored from raw data
    STAT_ROW_NAMES = ["MIN", "P1", "P25", "MEDIAN", "P75", "P99", "MAX", "MEAN", "MODE"]
    FEAT_INFO_DICTS = {
        # Patient baseline information sheet
        "#4_PAT_BL": {
            "centreid": str,
            "bmi": float,
            # "birthday": pd.Timestamp,
        },
        
        # Patient socio-demographic questionnaire (?)
        "#5_PAT_PSQ": {
            "ethnicity": str,
        }
    }