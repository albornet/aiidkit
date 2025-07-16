import pandas as pd
from src.data.data_utils import *


#####################
# LONGITUDINAL DATA #
#####################

def get_any_organ_transplantation_event(
    patient_ID: int,
    data: pd.DataFrame,
    include_kidneys: bool=True,
) -> pd.DataFrame:
    """ Get transplantation event, for Organ, including non-kidney
        TODO: COMPARE WITH GET_TRANSPLANTATION_DATE FROM KIDNEY_BL_DATA
    """
    # Select transplanted organ categories present in the database
    valid_categories = ["Pancreas", "Liver", "Islets", "Heart", "Lung"]
    if include_kidneys: valid_categories += ["Kidney"]

    tpx_event = get_time_value_pairs(
        patient_ID, data, "tpxdate", "organtype",
        context_key="organid",
        valid_categories=valid_categories,
    )
    if tpx_event.empty: return tpx_event

    return tpx_event.assign(
        entity="Organ info (" + tpx_event["value"] + ")",
        attribute="Transplantation event",
    )


def get_any_organ_status_uptate(
    patient_ID: int,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """ Get graft loss or PNF events for any transplanted organ, including non-kidney
    """
    status_update = get_time_value_pairs(
        patient_ID, data, "organtype_statusdate", "organtype_status",
        context_key="organid",
        valid_categories=["Graft loss", "PNF/Graft loss", "PNF"],
    )
    if status_update.empty: return status_update

    tpx_event = get_any_organ_transplantation_event(patient_ID, data)
    if tpx_event.empty: return status_update

    # For PNF, date is not defined, so take the transplantation date
    pnf_rows = status_update["value"].isin(["PNF/Graft loss", "PNF"])
    if pnf_rows.any():
        organid_to_time_map = tpx_event.set_index("organid")["time"]
        imputed_times = status_update.loc[pnf_rows, "organid"].map(organid_to_time_map)
        status_update.loc[pnf_rows, "time"] = imputed_times
    
    updated_organ = status_update["organid"].map(tpx_event.set_index("organid")["value"])
    status_update = status_update.drop(columns="organid")

    return status_update.assign(
        entity="Organ info (" + updated_organ + ")",
        attribute="Status update",
    )


def get_any_organ_donor_type(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Get status of the donor, adding organ name to the attribute, and date to
        avoid any mix-up for patients with several transplanted organs
        TODO: COMPARE WITH GET_DONOR_TYPE FROM KIDNEY_BL_DATA
    """
    valid_categories = ["Brain dead", "Living unrelated", "Living related", "NHBD", "KPD"]
    donor_type = get_categorical_feature_by_key(
        patient_ID=patient_ID,
        data=data,
        value_key="dontype",
        context_key="organid",
        valid_categories=valid_categories,
    ).rename(columns={"dontype": "value"})
    
    # Retrieve the organ(s) and date of donor_type based on "organid"
    tpx_event = get_any_organ_transplantation_event(patient_ID, data)
    if tpx_event.empty: return donor_type
    organid_to_tpx_info = tpx_event.set_index("organid")
    retrieved_organ = donor_type["organid"].map(organid_to_tpx_info["value"])
    retrieved_tpx_time = donor_type["organid"].map(organid_to_tpx_info["time"])

    return donor_type.assign(
        entity="Organ info (" + retrieved_organ + ")",
        attribute="Donor type",
        time=retrieved_tpx_time,
    )


def get_any_organ_resection_status(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Get resection status of the transplanted organ, adding organ name to the
        attribute, and date to avoid any mix-up with other transplanted organs
        TODO: COMPARE WITH GET_RESECTION_STATUS FROM KIDNEY_BL_DATA
    """
    valid_categories = ["First", "Re", "Sec"]
    resec_status = get_categorical_feature_by_key(
        patient_ID=patient_ID,
        data=data,
        value_key="resec_organtype",
        context_key="organid",
        valid_categories=valid_categories,
    ).rename(columns={"resec_organtype": "value"})
    
    tpx_event = get_any_organ_transplantation_event(patient_ID, data)
    if tpx_event.empty: return resec_status

    # Retrieve the organ(s) and date of donor_type based on "organid"
    tpx_event = get_any_organ_transplantation_event(patient_ID, data)
    if tpx_event.empty: return resec_status
    organid_to_tpx_info = tpx_event.set_index("organid")
    retrieved_organ = resec_status["organid"].map(organid_to_tpx_info["value"])
    retrieved_tpx_time = resec_status["organid"].map(organid_to_tpx_info["time"])
    
    return resec_status.assign(
        entity="Organ info (" + retrieved_organ + ")",
        attribute="Resec status",
        time=retrieved_tpx_time,
    )


#########################
# DATA POOLING FUNCTION #
#########################

def pool_organ_base_data(
    patient_ID: int,
    organ_base_df: pd.DataFrame,
) -> pd.DataFrame:
    """ Get information about all organs transplanted to a patient (including non-kidney)
    """
    # Build longitudinal features dataframe
    kbl_longitudinals = concatenate_clinical_information([
        get_any_organ_transplantation_event(patient_ID, organ_base_df),
        get_any_organ_status_uptate(patient_ID, organ_base_df),
        get_any_organ_donor_type(patient_ID, organ_base_df),
        get_any_organ_resection_status(patient_ID, organ_base_df),
    ])

    # Finalize patient dataframe
    patient_df = concatenate_clinical_information([kbl_longitudinals])
    patient_df = patient_df.drop_duplicates()
    # patient_df = patient_df.assign(entity="Organ info")  # TODO: CHECK FOR MORE FINE-GRAINED ENTITY ASSIGNATION STRATEGY
    patient_df = patient_df.sort_values(by=["time"])
    patient_df = patient_df[["entity", "attribute", "value", "time"]]
    
    return patient_df
