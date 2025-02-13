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

# - #1_CONSENT:    sheet done (only consent info from this one)
# - #2_KIDNEY_BL:  not done
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
        pat_id: pd.DataFrame(columns=["identity", "attribute", "value"])
        for pat_id in patient_ids_with_consent
    }
    
    # Iterate over all dataframes (different sheets in the excel file)
    for sheet_name, df in data_dict.items():
        if sheet_name in cfg.FEAT_INFO_DICTS:
            print(f"Processing {sheet_name} sheet")
            
            # Extract necessary data
            df = base_filtering(df, patient_ids_with_consent)
            feat_dict = cfg.FEAT_INFO_DICTS[sheet_name]
            df = df[["patid"] + list(feat_dict.keys())]
            
            # Melt dataframe to get a tall format: columns -> attribute/value rows
            melted_df = df.melt(
                id_vars=["patid"],
                value_vars=feat_dict.keys(),
                var_name="attribute",
                value_name="value",
            )
            
            # Populate patient records
            melted_df["identity"] = sheet_name  # could depend on a mapping between keys and sheet_name, e.g., one type of test in PAT_BL (TODO)
            melted_df["time"] = pd.NA  # should be updated if for timed events (TODO)
            for pat_id, group in melted_df.groupby("patid", as_index=False):
                patient_records[pat_id] = pd.concat(
                    [patient_records[pat_id], group], ignore_index=True
                )
    
    # Save patient dataframes
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    for pat_id, pat_df in patient_records.items():
        save_path = os.path.join(cfg.SAVE_DIR, f"patient_{pat_id}.csv")
        pat_df.to_csv(save_path, index=False)
        
        
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