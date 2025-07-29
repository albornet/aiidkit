import pandas as pd
from src.data.data_utils import *


################################
# PATIENT PRESCRIPTION HISTORY #
################################

def get_drug_events(
    patient_ID: int,
    data: pd.DataFrame,
    event_type: str,
    time_imputation_strategy: str|None="normal",  # None, "normal", "aggressive", "remove"
    other_drug_type_mapping_strategy: str|None="normal",  # None, "normal", "coarse"
) -> pd.DataFrame:
    """ Build start or stop drug events for a patient, with the type of drug as
        the entitiy, the drug name as attribute, the nature of the event as value,
        and the event date as time
    """
    # Get relevant data as time-value pairs
    impute_str = "_impute" if time_imputation_strategy is not None else ""
    time_key = f"{event_type}_date{impute_str}"
    drug_class = get_time_value_pairs(patient_ID, data, time_key, "class")
    drug_type = get_time_value_pairs(patient_ID, data, time_key, "type")
    # drug_counter = get_numerical_feature_by_key(patient_ID, data, "drug_counter", valid_range=(1, 100))

    # Identify drug type category for the ones that are named "other"
    if other_drug_type_mapping_strategy is not None:
        drug_mapping = {
            "fine": csts.DRUG_CATEGORY_MAP,
            "coarse": csts.DRUG_COARSE_NORMALIZATION_MAP,
        }[other_drug_type_mapping_strategy]

        drug_type_other = get_time_value_pairs(patient_ID, data, time_key, "other")
        other_drugs_mapped = drug_type_other["value"].map(drug_mapping)
        other_mask = drug_type["value"].isin(["Other", "other"])  # should just be "Other"
        drug_type.loc[other_mask, "value"] = other_drugs_mapped
    
    # Use regimen data to augment Tacrolimus ("FK") prescription information
    valid_regimens = ["Regular-release", "Extended-release", "Unknown"]
    drug_regimen = get_categorical_feature_by_key(patient_ID, data, "drugreg", valid_regimens)
    if not drug_regimen.empty:
        drug_type.loc[drug_regimen.index, "value"] += "-" + drug_regimen["drugreg"]

    # Combine extracted values using the indices to align rows correctly
    # "value": drug_counter["drug_counter"].map(lambda c: f"{event_type} ({c})"),
    entity_map_fn = lambda v: f"Drug event ({re.sub(r'(?<!^)(?=[A-Z])', ' ', v).lower()})"
    drug_df = pd.DataFrame({
        "entity": drug_class["value"].map(entity_map_fn),
        "attribute": drug_type["value"].map(lambda v: f"Drug - {v}"),
        "value": event_type.capitalize(),
        "time": drug_type["time"],
    })

    # Impute or remove missing / invalid times that are still missing
    if time_imputation_strategy == "aggressive":
        drug_df["time"] = drug_df["time"].ffill()
        drug_df["time"] = drug_df["time"].bfill()
    elif time_imputation_strategy == "remove":
        drug_df = drug_df.dropna(subset="time")
    
    return drug_df


#########################
# DATA POOLING FUNCTION #
#########################

def pool_patient_drug_data(
    patient_ID: int,
    patient_drug_df: pd.DataFrame,
    start_time_imputation_strategy: str|None="normal",  # None, "normal", "aggressive", "remove"
    stop_time_imputation_strategy: str|None="normal",  # None, "normal", "aggressive", "remove"
    other_drug_type_mapping_strategy: str|None="normal",  # None, "normal", "coarse"
) -> pd.DataFrame:
    """ Get date, static, and longitudinal kidney baseline data for one patient
    """
    # Build timed features
    drug_starts = get_drug_events(
        patient_ID=patient_ID,
        data=patient_drug_df,
        event_type="start",
        time_imputation_strategy=start_time_imputation_strategy,
        other_drug_type_mapping_strategy=other_drug_type_mapping_strategy,
    )
    drug_stops = get_drug_events(
        patient_ID=patient_ID,
        data=patient_drug_df,
        event_type="stop",
        time_imputation_strategy=stop_time_imputation_strategy,
        other_drug_type_mapping_strategy=other_drug_type_mapping_strategy,
    )
    pbl_timed_results = concatenate_clinical_information([drug_starts, drug_stops])

    # Finalize patient dataframe
    patient_df = concatenate_clinical_information([pbl_timed_results])
    patient_df = patient_df.drop_duplicates()
    patient_df = patient_df.sort_values(by=["time"])
    patient_df = patient_df[["entity", "attribute", "value", "time"]]

    return patient_df
