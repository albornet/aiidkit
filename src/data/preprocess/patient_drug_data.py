import pandas as pd
from src.data.data_utils import *


################################
# PATIENT PRESCRIPTION HISTORY #
################################

def get_drug_events(patient_ID: int, data: pd.DataFrame, event_type: str) -> pd.DataFrame:
    """ Build start or stop drug events for a patient, with the type of drug as
        the entitiy, the drug name as attribute, the nature of the event as value,
        and the event date as time
    """
    # Get relevant data as time-value pairs
    drug_class = get_time_value_pairs(patient_ID, data, f"{event_type}_date_impute", "class")
    drug_type = get_time_value_pairs(patient_ID, data, f"{event_type}_date_impute", "type")
    # drug_counter = get_numerical_feature_by_key(patient_ID, data, "drug_counter", valid_range=(1, 100))

    # Use regimen data to augment Tacrolimus ("FK") prescription information
    valid_regimens = ["Regular-release", "Extended-release", "Unknown"]
    drug_regimen = get_categorical_feature_by_key(patient_ID, data, "drugreg", valid_regimens)
    if not drug_regimen.empty:
        drug_type.loc[drug_regimen.index, "value"] += "-" + drug_regimen["drugreg"]
        
    # Combine extracted values using the indices to align rows correctly
    return pd.DataFrame({
        "entity": drug_class["value"].map(lambda v: f"Drug event: {v}"),
        "attribute": drug_type["value"].map(lambda v: f"Drug - {v}"),
        # "value": drug_counter["drug_counter"].map(lambda c: f"{event_type} ({c})"),
        "value": event_type.capitalize(),
        "time": drug_type["time"],
    })


#########################
# DATA POOLING FUNCTION #
#########################

def pool_patient_drug_data(
    patient_ID: int,
    patient_drug_df: pd.DataFrame,
) -> pd.DataFrame:
    """ Get date, static, and longitudinal kidney baseline data for one patient
    """
    # Build timed features
    pbl_timed_results = concatenate_clinical_information([
        get_drug_events(patient_ID, patient_drug_df, "start"),
        get_drug_events(patient_ID, patient_drug_df, "stop"),
    ])

    # Finalize patient dataframe
    patient_df = concatenate_clinical_information([pbl_timed_results])
    patient_df = patient_df.drop_duplicates()
    patient_df = patient_df.sort_values(by=["time"])
    patient_df = patient_df[["entity", "attribute", "value", "time"]]

    return patient_df
