import os
import re
import pandas as pd
from tqdm import tqdm
from process_data_utils import DataConfig as cfg


def main() -> None:
    raw_data_dict: dict[str, pd.DataFrame]
    raw_data_dict = pd.read_pickle(cfg.RAW_DATA_PATH_PICKLE)
    # raw_data_dict = pd.read_excel(cfg.RAW_DATA_PATH, sheet_name=None)
    raw_data_dict = {re.sub(r'#\d_', '', k).lower(): v for k, v in raw_data_dict.items()}
    if not cfg.DEBUG:
        create_eavt_patient_records(data_dict=raw_data_dict)
    else:
        explore_data(raw_data_dict)


def explore_data(data_dict: dict[str, pd.DataFrame]) -> None:
    """ Explore the data for debugging, without creating patient records
    """
    consent = data_dict["consent"]
    kidney_bl = data_dict["kidney_bl"]
    kidney_fup = data_dict["kidney_fup"]
    pat_bl= data_dict["pat_bl"]
    pat_psq = data_dict["pat_psq"]
    pat_id = data_dict["pat_id"]
    pat_drug = data_dict["pat_drug"]
    pat_stop = data_dict["pat_stop"]
    organ_base = data_dict["organ_base"]
    import ipdb; ipdb.set_trace()


def create_eavt_patient_records(data_dict: dict[str, pd.DataFrame]) -> None:
    """ Create patient records in EAVT format (entity, attribute, value, time)
    """
    # Initialize patients (and only keep the ones with consent)
    patient_ids_with_consent = get_pat_ids_with_consent(data_dict)
    patient_records = {
        pat_id: pd.DataFrame(columns=["entity", "attribute", "value", "time"]).astype(cfg.EAVT_DTYPES)
        for pat_id in patient_ids_with_consent
    }
    
    # Iterate over all sheets of the excel file data to fill patient records
    all_feats = pd.DataFrame()
    for sheet_name, sheet_df in data_dict.items():
        if sheet_name in cfg.TAKEN_DF_NAMES:
            print(f"Processing sheet {sheet_name}")
            data_df = base_filtering(sheet_df, patient_ids_with_consent)
            
            # Extract static and longitudinal features from sheet
            stat_feats = extract_static_features(data_df, sheet_name)
            long_feats = extract_longitudinal_features(data_df, sheet_name)
            all_feats = pd.concat([all_feats, stat_feats, long_feats], ignore_index=True)
            
    # Populate the patient records with identified features, sorted by timestamp
    all_feats["time"] = pd.to_datetime(all_feats["time"], errors="coerce")
    all_feats = all_feats.sort_values(by="time")
    add_features_to_patient_records(patient_records, all_feats.drop_duplicates())
    
    # Save each patient record to a csv file
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    for pat_id, pat_df in patient_records.items():
        save_path = os.path.join(cfg.SAVE_DIR, f"patient_{pat_id}.csv")
        pat_df.to_csv(save_path, index=False)


def extract_static_features(
    data_df: pd.DataFrame,
    data_key: str,
) -> None:
    """ Add static features to patient records
    """
    # Extract static features data
    print(f" - Extracting static features")
    feat_dict: dict = cfg.STATIC_FEAT_INFO_DICTS[data_key]
    static_df = data_df[["patid"] + list(feat_dict.keys())]
    static_df = static_df.melt(  # tall format: columns -> attribute / value rows
        id_vars=["patid"],
        value_vars=feat_dict.keys(),
        var_name="attribute",
        value_name="value",
    )
    static_df["entity"] = data_key
    static_df["time"] = pd.NaT  # no time for static features
    
    # Invert value and time columns for "static" features that are dates
    time_attrs = [attr for attr, dtype in feat_dict.items() if dtype == pd.Timestamp]
    is_a_time = static_df["attribute"].isin(time_attrs)
    static_df.loc[is_a_time, "value"] = pd.to_datetime(static_df.loc[is_a_time, "value"], errors="coerce")
    static_df.loc[is_a_time, ["value", "time"]] = static_df.loc[is_a_time, ["time", "value"]].values
    
    return static_df.drop_duplicates()
    

def extract_longitudinal_features(
    data_df: pd.DataFrame,
    data_key: str,
) -> None:
    """ Add longitudinal features to patient records
    """
    # Initial setup
    longitudinal_dict: dict = cfg.LONGITUDINAL_FEAT_INFO_DICTS[data_key]
    longitudinal_df = pd.DataFrame()
    
    # Extract longitudinal features data
    for attr_field in tqdm(longitudinal_dict.keys(), desc=f" - Extracting longitudinal features"):
        relevant_columns = f"^(patid|currentformid|{attr_field}_|{attr_field}date_)"  # |{attr_field}oth_)"
        attr_df = data_df.filter(regex=relevant_columns)
        
        # Melt all events to a long format  # TODO: CHECK IF THERE IS A BETTER WAY THAN USING "currentformid" to identify rows
        attr_df = pd.wide_to_long(
            df=attr_df,
            stubnames=[attr_field, f"{attr_field}date"],
            i="currentformid",
            j="seq_id",
            sep="_",
            suffix=r"\d+",
        ).reset_index().drop(columns=["seq_id", "currentformid"])
        
        # Rename columns to the EAVT format
        attr_df = attr_df.rename(columns={attr_field: "value", f"{attr_field}date": "time"})
        attr_df = attr_df.dropna(subset=["value"])
        attr_df["entity"] = data_key
        attr_df["attribute"] = attr_field
        
        # Fill missing dates with a default date (to distinguish from static features)
        attr_df["time"] = attr_df["time"].fillna(pd.Timestamp("2000-01-01 00:00:00"))
        
        # Add identified features to the longitudinal dataframe
        longitudinal_df = pd.concat([longitudinal_df, attr_df], ignore_index=True)
        
    return longitudinal_df.drop_duplicates()
        
        
def add_features_to_patient_records(
    patient_records: dict[int, pd.DataFrame],
    features_df: pd.DataFrame,
) -> None:
    """ Add features to patient records
    """
    assert "patid" in features_df.columns, "Ensure patid column is part of features_df"
    for pat_id, group in tqdm(features_df.groupby("patid"), desc="Populating patient records"):
        group = group.drop(columns=["patid"])  # csv files are already named after patid
        patient_records[pat_id] = pd.concat([patient_records[pat_id], group], ignore_index=True)
        
        
def get_pat_ids_with_consent(data_dict: dict[str, pd.DataFrame]) -> pd.Series:
    """ Identify patients who gave their consent for retrospective analysis
    """
    consent_df = data_dict["consent"]
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
    if "__STAT__" in df.columns:
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