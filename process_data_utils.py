import os
import pandas as pd

class DataConfig:
    """ Configuration for the data processing pipeline
    """
    # Paths
    DATA_DIR = "data"
    RAW_DATA_NAME = "SAS_Data_Dictionary_FUP226_raw-01Jan2023_v1.xlsx"
    RAW_DATA_PATH = os.path.join(DATA_DIR, RAW_DATA_NAME)
    SAVE_DIR = os.path.join(DATA_DIR, "processed")
    
    # Features and rows taken/ignored from raw data
    STAT_ROW_NAMES = ["MIN", "P1", "P25", "MEDIAN", "P75", "P99", "MAX", "MEAN", "MODE"]
    TAKEN_DF_NAMES = ["#2_KIDNEY_BL", "#3_KIDNEY_FUP", "#4_PAT_BL", "#5_PAT_PSQ"]
    
    # Information to preprocess static patient and transplant features
    STATIC_FEAT_INFO_DICTS = {        
        # Kidney-transplant baseline infromation sheet
        "#2_KIDNEY_BL": {
            "organtype": str,  # should always be kidney?
            "tpxdate": pd.Timestamp,  # transplantation date
        },
        
        # Kidney-transplant follow-up information sheet
        "#3_KIDNEY_FUP": {
            "tpxdate": pd.Timestamp,  # transplantation date
        },
        
        # Patient baseline information sheet
        "#4_PAT_BL": {
            "centreid": str,
            "bmi": float,
            "birthday": pd.Timestamp,
        },
        
        # Patient socio-demographic questionnaire sheet
        "#5_PAT_PSQ": {
            "ethnicity": str,
        },
    }
    
    # Information to preprocess longitudinal patient and transplant features
    # TODO: ALWAYS CHECK FOR KEY + "oth" and KEY + "date"
    # TODO MAYBE: KEY + "clinical" for associated clinical note
    LONGITUDINAL_FEAT_INFO_DICTS = {
        # Kidney-transplant baseline infromation sheet
        "#2_KIDNEY_BL": {},
        
        # Kidney-transplant follow-up information sheet
        "#3_KIDNEY_FUP": {  
            "glo": str,  # reason of graft loss
            "rj": str,  # graft rejection, type, and grading
            "alldis": str,  # allograft disease
            "txcomp": str,  # transplant-related complication
            "immu": str,  # type of immuno test
            # "bkv": int,  # bkv viremia (copies/ml)  # DATED BUT NOT LONGITUDINAL
            # "proteinuria": float,  # urinary protein to creatinin ratio (mg/mmol)  # DATED BUT NOT LONGITUDINAL
        },
        
        # Patient baseline information sheet
        "#4_PAT_BL": {
            "mek": str,
        },
        
        # Patient socio-demographic questionnaire sheet
        "#5_PAT_PSQ": {},
    }
    
    # Data types for the final patient records
    EAVT_DTYPES = {
        "entity": "string",
        "attribute": "string",
        "value": "object",
        "time": "datetime64[ns]",
    }