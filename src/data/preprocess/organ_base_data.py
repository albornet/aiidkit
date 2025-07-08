import pandas as pd
from src.data.data_utils import *


#####################
# LONGITUDINAL DATA #
#####################

def get_transplantation_event(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    """ Get transplantation event, for any organ, including non-kidney
        TODO: COMPARE WITH GET_TRANSPLANTATION_DATE FROM KIDNEY_BL_DATA
    """
    tpx_event = get_time_value_pairs(
        patient_ID, data, "tpxdate", "organtype",
        valid_categories=["Kidney", "Pancreas", "Liver", "Islets", "Heart", "Lung"]  # the ones present in the database
    )
    if tpx_event.empty: return tpx_event
    return tpx_event.assign(attribute="Transplant event")


def get_organ_status_uptate(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    """ Get graft loss or PNF events for any transplanted organ, including non-kidney
    """
    status_update = get_time_value_pairs(
        patient_ID, data, "organtype_statusdate", "organtype_status",
        valid_categories=["Graft loss", "PNF/Graft loss", "PNF"],
    )
    if status_update.empty: return status_update

    # For PNF, date is not defined, so take the transplantation date
    pnf_rows = status_update["value"].isin(["PNF/Graft loss", "PNF"])
    if pnf_rows.any():
        # Matched with index row and column field
        status_update.loc[pnf_rows, "time"] = get_transplantation_event(patient_ID, data)
    
    return status_update.assign(attribute="Organ status update")


def get_donor_type(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    """ Get status of the donor, adding organ name to the attribute, and date to
        avoid any mix-up for patients with several transplanted organs
        TODO: COMPARE WITH GET_DONOR_TYPE FROM KIDNEY_BL_DATA
    """
    valid_categories = ["Brain dead", "Living unrelated", "Living related", "NHBD", "KPD"]
    donor_type = get_categorical_feature_by_key(patient_ID, data, "dontype", valid_categories)
    
    tpx_event = get_transplantation_event(patient_ID, data)
    if tpx_event.empty: return donor_type

    donor_type = donor_type.rename(columns={"dontype": "value"})
    donor_type["attribute"] = "Donor type - " + tpx_event["value"]
    return donor_type.assign(time=tpx_event["time"])


def get_resection_status(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    """ Get resection status of the transplanted organ, adding organ name to the
        attribute, and date to avoid any mix-up with other transplanted organs
        TODO: COMPARE WITH GET_RESECTION_STATUS FROM KIDNEY_BL_DATA
    """
    valid_categories = ["First", "Re", "Sec"]
    resec_status = get_categorical_feature_by_key(patient_ID, data, "resec_organtype", valid_categories)
    
    tpx_event = get_transplantation_event(patient_ID, data)
    if tpx_event.empty: return resec_status

    resec_status = resec_status.rename(columns={"resec_organtype": "value"})
    resec_status["attribute"] = "Resec status - " + tpx_event["value"]
    return resec_status.assign(time=tpx_event["time"])


#########################
# DATA POOLING FUNCTION #
#########################

def pool_organ_base_data(
    patient_ID:int,
    organ_base_df:pd.DataFrame,
) -> pd.DataFrame:
    """ Get information about all organs transplanted to a patient (including non-kidney)
    """
    # Build longitudinal features dataframe
    kbl_longitudinals = concatenate_clinical_information([
        get_transplantation_event(patient_ID, organ_base_df),
        get_organ_status_uptate(patient_ID, organ_base_df),
        get_donor_type(patient_ID, organ_base_df),
        get_resection_status(patient_ID, organ_base_df),
    ])

    # Finalize patient dataframe
    patient_df = concatenate_clinical_information([kbl_longitudinals])
    patient_df = patient_df.drop_duplicates()
    patient_df = patient_df.assign(entity="organ_base")  # TODO: CHECK FOR MORE FINE-GRAINED ENTITY ASSIGNATION STRATEGY
    patient_df = patient_df.sort_values(by=["time"])
    patient_df = patient_df[["entity", "attribute", "value", "time"]]
    
    return patient_df
