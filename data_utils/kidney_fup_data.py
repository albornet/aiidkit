import numpy as np
import pandas as pd
from .data_utils import *  # what is wrong with this import ? nothing, it looks great


##############
# DATES INFO #
##############

def get_assessment_date(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    # INFO: this may be used for dating biopsy tests (to check)
    return get_date_by_key(patient_ID, data, "assdate")

def get_transplantation_date(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_date_by_key(patient_ID, data, "tpxdate")


########################
# PAIRED PATIENT INFOS #
########################

def get_insufficient_urine_level(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    # INFO: this computes if the patient has insufficient urine level in the 24 hours after transplantation
    urine24 = get_time_value_pairs(patient_ID, data, "tpxdate", "urine24", valid_categories=["No", "Yes"])
    urine24["time"] = urine24["time"] + pd.Timedelta(days=1)
    return urine24

def get_bkv_uremia_level(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    # INFO: in copies/ml, from 0 to approx. 10_000_000 (+ nan-like values)
    bkv = get_time_value_pairs(
        patient_ID, data, "bkvdate", "bkv",
        value_type="numerical", valid_number_range=(0, 10_000_000),
    )
    bkv["value"] = np.log10(bkv["value"].add(1.0))
    return bkv

def get_protein_uria_level(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    # INFO: in copies/ml, from 0 to approx. 4_000 (+ nan-like values)
    protein_uria = get_time_value_pairs(
        patient_ID, data, "proteinuriadate", "proteinuria",
        value_type="numerical", valid_number_range=(0, 10_000),
    )
    protein_uria["value"] = np.log10(protein_uria["value"].add(1))
    return protein_uria

def get_early_allograft_dysfunction(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    # INFO: This function combines two features into attribute / value pair rows
    all_pairs = data.loc[data["patid"] == patient_ID, ["dgf", "dgfduration"]]
    valid_categories = ["DGF", "PNF"]
    relevant_pairs = all_pairs[all_pairs["dgf"].isin(valid_categories)]
    attr_val_pairs = relevant_pairs.rename(columns={"dgf": "attribute", "dgfduration": "value"})
    return attr_val_pairs.assign(time=pd.NaT)


#############################
# PATIENT LONGITUDINAL DATA #
#############################

def get_reason_for_graft_loss(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_longitudinal_data(patient_ID, data, "glodate", "glo")

def get_rejection_event(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_longitudinal_data(patient_ID, data, "rjdate", "rj")

def get_graft_loss_event(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_longitudinal_data(patient_ID, data, "rjdate", "rjclinical")

def get_allograft_disease_event(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_longitudinal_data(patient_ID, data, "alldisdate", "alldis")

def get_transplant_related_complication_event(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_longitudinal_data(patient_ID, data, "txcompdate", "txcomp")

def get_immuno_test_results(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_longitudinal_data(patient_ID, data, "immudate", "immures", attribute_key="immu")


#########################
# DATA POOLING FUNCTION #
#########################

def pool_kidney_fup_data(
    patient_ID:int,
    kidney_fup_df:pd.DataFrame,
) -> pd.DataFrame:
    """ Get date, static, and longitudinal kidney follow-up data for one patient
    """
    # Build paired dataframe (either time-paired or two static features)
    kfup_paired = pd.concat([
        get_insufficient_urine_level(patient_ID, kidney_fup_df),  # time-paired
        get_bkv_uremia_level(patient_ID, kidney_fup_df),  # time-paired
        get_protein_uria_level(patient_ID, kidney_fup_df),  # time-paired
        get_early_allograft_dysfunction(patient_ID, kidney_fup_df),  # static pair
    ], ignore_index=True)

    # Build longitudinal features dataframe
    kfup_longitudinals = pd.concat([
        get_reason_for_graft_loss(patient_ID, kidney_fup_df),
        get_rejection_event(patient_ID, kidney_fup_df),
        get_graft_loss_event(patient_ID, kidney_fup_df),
        get_allograft_disease_event(patient_ID, kidney_fup_df),
        get_transplant_related_complication_event(patient_ID, kidney_fup_df),
        get_immuno_test_results(patient_ID, kidney_fup_df),
    ], ignore_index=True)

    # Finalize patient dataframe
    patient_kfup_df = pd.concat([kfup_paired, kfup_longitudinals], ignore_index=True)
    patient_kfup_df = patient_kfup_df.drop_duplicates()
    patient_kfup_df = patient_kfup_df.assign(entity="kidney_fup")  # TODO: CHECK FOR MORE FINE-GRAINED ENTITY ASSIGNATION STRATEGY
    patient_kfup_df = patient_kfup_df.sort_values(by=["time"])
    patient_kfup_df = patient_kfup_df[["entity", "attribute", "value", "time"]]

    return patient_kfup_df
