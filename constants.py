import os
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass(frozen=True)
class ConstantsNamespace():

    # Paths
    EXCEL_DATA_PATH = os.path.join("data", "data_files", "raw", "Datacut_FUP226_raw-01Jan2023_v1.xlsx")
    PICKLE_DATA_PATH = os.path.join("data", "data_files", "raw", "Datacut_FUP226_raw-01Jan2023_v1.pkl")
    OUTPUT_DIR_PATH = os.path.join("data", "data_files", "raw", "processed")
    
    # Sheet names
    CONSENT_SHEET = "Consent"  # "#1_CONSENT" <- names change for the full data file
    KIDNEY_BL_SHEET = "Kidney_BL"  # "#2_KIDNEY_BL"
    KIDNEY_FUP_SHEET = "Kidney_FUP"  # "#3_KIDNEY_FUP"
    PATIENT_BL_SHEET = "PAT_BL"  # "#4_PAT_BL"
    PATIENT_PSQ_SHEET = "PAT_PSQ"  # "#5_PAT_PSQ"
    PATIENT_INFECTION_SHEET = "PAT_ID"  # "#6_PAT_ID" <- renamed to avoid confusion with patient identifier (ID)
    PATIENT_DRUG_SHEET = "PAT_Drug"  # "#7_PAT_DRUG"
    PATIENT_STOP_SHEET = "PAT_Stop"  # "#8_PAT_STOP"
    ORGAN_BASE_SHEET = "Organ_Base"  # "#9_ORGAN_BASE"

    # To use for removing missing or invalid values
    VALID_DATE_RANGE = (pd.Timestamp("1900-01-01"), pd.Timestamp("2030-01-01"))
    NAN_LIKE_DATES = (pd.NaT, pd.Timestamp("9999-01-01"), pd.Timestamp("2000-01-01"))
    NAN_LIKE_NUMBERS = (np.nan, pd.NA, -555.0, -666.0, -777.0, -888.0, -999.0)
    NAN_LIKE_CATEGORIES = ("NaN", "nan", "Nan", pd.NA, np.nan, "Global consent refused", "Refused", "Not done", "Not applicable", "Unknown")
