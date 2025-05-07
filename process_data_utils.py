import os
import pandas as pd

class DataConfig:
    """ Configuration for the data processing pipeline
    """
    # General
    DEBUG = False
    DATA_DIR = "data"
    RAW_DATA_NAME = "Datacut_FUP226_raw-01Jan2023_v1.xlsx"  # "SAS_Data_Dictionary_FUP226_raw-01Jan2023_v1.xlsx"
    RAW_DATA_PATH = os.path.join(DATA_DIR, RAW_DATA_NAME)
    RAW_DATA_PATH_PICKLE = RAW_DATA_PATH.replace(".xlsx", ".pkl")
    SAVE_DIR = os.path.join(DATA_DIR, "processed")
    
    # Features and rows taken/ignored from raw data
    STAT_ROW_NAMES = ["MIN", "P1", "P25", "MEDIAN", "P75", "P99", "MAX", "MEAN", "MODE"]
    TAKEN_DF_NAMES = ["kidney_bl", "kidney_fup", "pat_bl", "pat_psq"]
    
    # Useful to remove invalid date (NOT USED YET)
    NAN_VALUES = [
        "NaN", "Global consent refused", "Not applicable", "Unknown", ...,
        -777.0, -888.0, -999.0, ...,
        "9999-01-01", "2000-01-01", ...,
    ]

    # Information to preprocess static patient and transplant features
    STATIC_FEAT_INFO_DICTS = {

        # Kidney-transplant baseline infromation sheet /!\ specific to organid /!\
        "kidney_bl": {
            "tpxdate": pd.Timestamp,  # transplantation date
            "hospstart": pd.Timestamp,  # hospital admission date
            "hospend": pd.Timestamp,  # hospital discharge date
            "dialysisdate": pd.Timestamp,  # initial dialysis type <- maybe combine to tpxdate to derive time since last dialysis?
            "dialysistype": str,  # HD 2334, NaN 576, PD 480, Global consent refused 3 - initial dialysis type
            "warmisch": float,  # warm ischemia time (minutes)
            "coldischmin": float,  # cold ischemia time (minutes) - time between aorta closed at donor till re-opened at reciptient
            "coldischmin2": float,  # cold ischemia time for second kidney (minutes), not sure how this works...
            "tpxtype": str,  # Right 1545, NaN 1073, Left, 702, Double 53, Unknown 15, Global consent refused 5
            "ec": str,  # No 1715, Yes 1004, Unknown 498, NaN 173, Global consent refused 3 - blood transfusion event
            "ss": str,  # Not applicable 1448, Yes 794, NaN 718, No 401, Unknown 29 Global consent refused 3 - pregnancy event
            "hbsag": str,  # Negative 3331, Positive 53, Unknown 3, Global consent refused 3, NaN 2, Not applicable 1 - hepatitis-B surface antigen
            "antihbs": str,  # Positive 1970, Negative 1254, NaN 126, Unknown 40, Global consent refused 3  - antibody for hepatitis-B surface antigen
            "antihbc": str,  # Negative 2957, Positive 347, NaN 57, Unknown 29, Global consent refused 3 - antibody for hepatitis-B core antigen
            "antihcv": str,  # Negative 3261, Positive 123, NaN 3, Global consent refused 3, Unknown 2, Not applicable 1 - antibody for hepatitis-C virus
            "anticmv": str,
            "antiebv": str,
            "antihiv": str,
            "antitoxo": str,
            "antivzv": str,
            "antihsv": str,
            "tpha": str,  # Negative 2729, NaN 362, Not applicable 222, Positive 46, Unknown 31, Global consent refused 3
            "donage": float,  # donor age (years) -> or donbirthdate?
            "donsex": str,  # Female 1698, Male 1690, Global consent refused 3, NaN 2 - donor sex
            "doncod": str,  # CHE 951, Not applicable 818, ANX 555, CTR 424, NaN 420, CDI 138, Other 47, SUI 32, CTU 7, Unknown 1 - donor cause of death
            "donorbg": str,  # A 1491, 0 1488, B 303, AB 111 - donor blood group
            "donorpool": str,  # NaN 2512, No 575, Yes 306 - donor belongs to extended pool (true = probably less compatible)
            "donhbsag": str,  # same tests and values as for receiver
            "donantihbs": str,  # same tests and values as for receiver
            "donantihbc": str,  # same tests and values as for receiver
            "donantihcv": str,  # same tests and values as for receiver
            "donanticmv": str,  # same tests and values as for receiver
            "donantiebv": str,  # same tests and values as for receiver
            "donantihiv": str,  # same tests and values as for receiver
            "donantitoxo": str,  # same tests and values as for receiver
            "donantivzv": str,  # same tests and values as for receiver
            "donantihsv": str,  # same tests and values as for receiver
            "hlaamismatch": float,  # number of HLA-A mismatches
            "hlabmismatch": float,  # number of HLA-B mismatches
            "hladrmismatch": float,  # number of HLA-DR mismatches
            "sumhlamismatch": float,  # sum of HLA mismatches (provides good bias)
            "witprim": float, # asystolic ischemia time (min), cardiac arrest till preservation solution perfusion
            "organtype_counter": int,  # "how-many-th" transplanted organ is this one for the patient
            "resecstcs_organtype": str,  # First 3264, Re 105, Sec 24 - re would be same organ "place" but new transplantation event

        },
        
        # Kidney-transplant follow-up information sheet
        "kidney_fup": {
            "tpxdate": pd.Timestamp,  # transplantation date
        },
        
        # Patient baseline information sheet
        "pat_bl": {
            "centreid": str,
            "bmi": float,
            "birthday": pd.Timestamp,
        },
        
        # Patient socio-demographic questionnaire sheet
        "pat_psq": {
            "ethnicity": str,
        },
    }
    
    # Information to preprocess longitudinal patient and transplant features
    LONGITUDINAL_FEAT_INFO_DICTS = {
        # Kidney-transplant baseline infromation sheet
        "kidney_bl": {
            "etio": str,  # etiology of kidney disease
            "immu": str,  # immuno-test -> /!\ also (immunum, immures, immumeth)
        },
        
        # Kidney-transplant follow-up information sheet
        "kidney_fup": {  
            "glo": str,  # reason of graft loss
            "rj": str,  # graft rejection, type, and grading
            "alldis": str,  # allograft disease
            "txcomp": str,  # transplant-related complication
            "immu": str,  # type of immuno test
            # "bkv": int,  # bkv viremia (copies/ml)  # DATED BUT NOT LONGITUDINAL
            # "proteinuria": float,  # urinary protein to creatinin ratio (mg/mmol)  # DATED BUT NOT LONGITUDINAL
        },
        
        # Patient baseline information sheet
        "pat_bl": {
            "mek": str,
        },
        
        # Patient socio-demographic questionnaire sheet
        "pat_psq": {},
    }
    
    # Data types for the final patient records
    EAVT_DTYPES = {
        "entity": "string",
        "attribute": "string",
        "value": "object",
        "time": "datetime64[ns]",
    }