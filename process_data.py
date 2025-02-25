import os
import pandas as pd
from process_data_utils import DataConfig as cfg


############
# COMMENTS #
############

# BL:  BASELINE
# FUP: FOLLOW-UP
# ID:  MAY SOMETIMES MEAN INFECTION DISEASE (BUT USUALLY IDENTIFIER...)
# PAT: PATIENT

# - #1_CONSENT:    skipped (only consent info from this one, abandonned for #9_ORGAN_BASE)
# - #2_KIDNEY_BL:  not done (contains transplantation + donor info)
# - #3_KIDNEY_FUP: not done
# - #4_PAT_BL:     not done
# - #5_PAT_PSQ:    done (only ethnicity info)
# - #6_PAT_ID:     not done (may be from where target labels can be extracted)
# - #7_PAT_DRUG:   not done (pay attention: some drugs could leak label information)
# - #8_PAT_STOP:   not done (may be from where censoring info can be extracted)
# - #9_ORGAN_BASE: not done (contains a lot of static info and also about censoring)

############
# COMMENTS #
############


def main() -> None:
    """ ...
    """
    raw_data_dict: dict[str, pd.DataFrame]
    raw_data_dict = pd.read_excel(cfg.RAW_DATA_PATH, sheet_name=None)
    process_data(data_dict=raw_data_dict)


def process_data(data_dict: dict[str, pd.DataFrame]) -> None:
    """ ...
    """
    # Initialize patients (and only keep the ones with consent)
    patient_ids_with_consent = get_pat_ids_with_consent(data_dict)
    patient_records = {
        pat_id: pd.DataFrame(columns=["entity", "attribute", "value", "time"]).astype(cfg.EAVT_DTYPES)
        for pat_id in patient_ids_with_consent
    }
    
    # Iterate over all sheets of the excel file data to fill patient records
    for sheet_name, sheet_df in data_dict.items():
        if sheet_name in cfg.TAKEN_DF_NAMES:
            print(f"Processing {sheet_name} sheet")
            data_df = base_filtering(sheet_df, patient_ids_with_consent)
            add_static_features_to_patient_records(patient_records, data_df, sheet_name)
            add_longitudinal_features_to_patient_records(patient_records, data_df, sheet_name)
    
    # Save each patient record to a csv file
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    for pat_id, pat_df in patient_records.items():
        save_path = os.path.join(cfg.SAVE_DIR, f"patient_{pat_id}.csv")
        pat_df.to_csv(save_path, index=False)


def add_static_features_to_patient_records(
    patient_records: dict[int, pd.DataFrame],
    data_df: pd.DataFrame,
    data_key: str,
) -> None:
    """ Add static features to patient records (...)
    """
    # Extract static features data
    feat_dict: dict = cfg.STATIC_FEAT_INFO_DICTS[data_key]
    static_df = data_df[["patid"] + list(feat_dict.keys())]
    static_df = static_df.melt(  # tall format: columns -> attribute/value rows
        id_vars=["patid"],
        value_vars=feat_dict.keys(),
        var_name="attribute",
        value_name="value",
    )
    
    # Populate patient records with static features
    static_df["entity"] = data_key  # could depend on a mapping between keys and data_key, e.g., one type of test in PAT_BL?
    static_df["time"] = pd.NaT  # static features are assigned time "not a time"
    for pat_id, group in static_df.groupby("patid", as_index=False):
        patient_records[pat_id] = pd.concat([patient_records[pat_id], group], ignore_index=True)


def add_longitudinal_features_to_patient_records(
    patient_records: dict[int, pd.DataFrame],
    data_df: pd.DataFrame,
    data_key: str,
) -> None:
    """ Add longitudinal features to patient records (...)
    """
    # Extract longitudinal features data
    longitudinal_dict: dict = cfg.LONGITUDINAL_FEAT_INFO_DICTS[data_key]
    for attr_field in longitudinal_dict.keys():
        relevant_columns = f"^(patid|currentformid|{attr_field}_|{attr_field}date_)"  # |{attr_field}oth_)"
        longitudinal_df = data_df.filter(regex=relevant_columns)
        
        # Melt all events to a long format  # TODO: CHECK IF THERE IS A BETTER WAY THAN USING "currentformid" to identify rows
        longitudinal_df = pd.wide_to_long(
            df=longitudinal_df,
            stubnames=[attr_field, f"{attr_field}date"],
            i="currentformid",
            j="seq_id",
            sep="_",
            suffix=r"\d+",
        ).reset_index().drop(columns=["seq_id", "currentformid"])
        
        # Populate patient records with longitudinal features
        longitudinal_df = longitudinal_df.rename(columns={attr_field: "value", f"{attr_field}date": "time"})
        longitudinal_df = longitudinal_df.dropna(subset=["value"])
        longitudinal_df["entity"] = data_key
        longitudinal_df["attribute"] = attr_field
        for pat_id, group in longitudinal_df.groupby("patid"):
            patient_records[pat_id] = pd.concat([patient_records[pat_id], group], ignore_index=True)

        
def get_pat_ids_with_consent(data_dict: dict[str, pd.DataFrame]) -> pd.Series:
    """ Identify patients who gave their consent for retrospective analysis
    """
    consent_df = data_dict["#1_CONSENT"]
    consent_given = consent_df["consstatus"] == "Consent given"
    pat_ids_with_consent = consent_df.loc[consent_given, "patid"]
    
    return pat_ids_with_consent.astype(int)


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
    
    return df


def print_df_info(df: pd.DataFrame) -> None:
    """ Print some basic info from any dataframe
    """
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns}")
    print(f"Head: {df.head(5)}")
    

if __name__ == "__main__":
    main()