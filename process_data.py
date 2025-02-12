import pandas as pd
from process_data_utils import DataConfig, PatientRecord


def main() -> None:
    """ ...
    """
    raw_data_dict: dict[str, pd.DataFrame]
    raw_data_dict = pd.read_excel(DataConfig.RAW_DATA_PATH, sheet_name=None)
    process_data(data_dict=raw_data_dict)


def process_data(data_dict: dict[str, pd.DataFrame]) -> None:
    """ ...
    """
    # Initialize patients (and only keep the ones with consent)
    patient_ids_with_consent = get_pat_ids_with_consent(data_dict)
    patient_records = {
        pat_id: PatientRecord(pat_id) for pat_id in patient_ids_with_consent
    }
    
    # Iterate over all dataframes (different sheets in the excel file)
    for sheet_name, df in data_dict.items():
        if sheet_name in DataConfig.FEAT_INFO_DICTS:
            print(f"Processing {sheet_name} sheet")
            
            # Extract necessary data
            df = base_filtering(df, patient_ids_with_consent)
            feat_dict = DataConfig.FEAT_INFO_DICTS[sheet_name]
            df = df[["patid"] + list(feat_dict.keys())]
            
            # Melt dataframe to get a tall format: columns -> attribute/value rows
            melted_df = df.melt(
                id_vars="patid",
                value_vars=feat_dict.keys(),
                var_name="attribute",
                value_name="value",
            )
            
            # Group by patient ID once, and create EAVRecords
            for patid, group in melted_df.groupby("patid"):
                pat_record = patient_records[patid]
                for _, row in group.iterrows():
                    pat_record.add_element(
                        entity=sheet_name, attribute=row.attribute, value=row.value
                    )
                
            

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
    stat_rows: list[str]=DataConfig.STAT_ROW_NAMES,
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