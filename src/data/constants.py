import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass(frozen=True)
class ConstantsNamespace():

    # Paths
    EXCEL_DATA_PATH = os.path.join("data", "raw", "Datacut_FUP226_raw-01Jan2023_v1.xlsx")
    PICKLE_DATA_PATH = os.path.join("data", "raw", "Datacut_FUP226_raw-01Jan2023_v1.pkl")
    PREPROCESSED_DIR_PATH = os.path.join("data", "preprocessed")
    HUGGINGFACE_DIR_PATH = os.path.join("data", "huggingface")
    RESULT_DIR_PATH = os.path.join("results")
    
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
    NAN_LIKE_CATEGORIES = (
        "NaN", "nan", "Nan", pd.NA, np.nan, "NA in FUP", -999.0,  # "Unknown"
        "Global consent refused", "Refused", "Not done", "Not applicable",
    )

    # Global mapping for patient ethnicity normalization
    ETHNICITY_NORMALIZATION_MAP = {
        "African": "African",
        "african": "African",
        "American Indian": "American Indian",
        "amerindien": "American Indian",
        "AmÃ©rique du Sud": "South American",
        "Amérique du Sud": "South American",
        "Bolivian": "South American (Bolivian)",
        "Brasil": "South American (Brazilian)",
        "Brazil": "South American (Brazilian)",
        "Brazilian": "South American (Brazilian)",
        "Caucasian": "Caucasian",
        "Caucasian and Africain": "Mixed Race (Caucasian and African)",
        "african/caucasian": "Mixed Race (Caucasian and African)",
        "Chile": "South American (Chilean)",
        "chile": "South American (Chilean)",
        "Dominican Republic": "Latin American (Dominican Republic)",
        "India": "Asian (Indian)",
        "Jemen": "Middle Eastern (Yemeni)",
        "Kurde": "Middle Eastern (Kurdish)",
        "Latin american": "Latin American",
        "latino american": "Latin American",
        "Mauritian": "African (Mauritian)",
        "Mongolian": "Asian (Mongolian)",
        "Moroccan": "North African (Moroccan)",
        "North Africa": "North African",
        "South American": "South American",
        "South American (American Indian)": "South American (American Indian)",
        "Sud American (Chili)": "South American (Chilean)",
        "Syria": "Middle Eastern (Syrian)",
        "syria": "Middle Eastern (Syrian)",
        "Syrian, Saoudian, Caucasian": "Mixed Race (Syrian, Saudi, Caucasian)",
        "Tamil": "Asian (Tamil)",
        "tamil": "Asian (Tamil)",
        "thamil": "Asian (Tamil)",
        "Tunesier": "North African (Tunisian)",
        "Tunisia": "North African (Tunisian)",
        "Tunisian": "North African (Tunisian)",
        "Turkey": "Middle Eastern (Turkish)",
        "Turkish": "Middle Eastern (Turkish)",
        "turkish": "Middle Eastern (Turkish)",
        "Unknown": "Unknown",
        "arab": "Middle Eastern (Arab)",
        "arabic": "Middle Eastern (Arab)",
        "caucasian and arabic": "Mixed Race (Caucasian and Arab)",
        "caucasian and latino": "Mixed Race (Caucasian and Latin American)",
        "caucasian and latino american": "Mixed Race (Caucasian and Latin American)",
        "hispanic": "Latin American (Hispanic)",
        "lebanese": "Middle Eastern (Lebanese)",
        "marokko and caucasian": "Mixed Race (Moroccan and Caucasian)",
        "mixed race": "Mixed Race",
        "persian": "Middle Eastern (Persian)",
    }
