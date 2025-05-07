import numpy as np
import pandas as pd
from enum import Enum


class Consent(Enum):
    CONSENT_GIVEN = 0
    CONSENT_REFUSED = 1


def get_patients_IDs(data_dict: pd.DataFrame) -> pd.DataFrame:
    """ Get all patients IDs, even non consenting ones, and remove dupplicates
    """
    patients_IDs = data_dict["patid"]
    patients_IDs.replace([np.inf, -np.inf], np.nan, inplace=True)
    patients_IDs.dropna(inplace=True)
    patients_IDs.drop_duplicates(inplace=True)
    patients_IDs = patients_IDs.astype(np.int64, copy=None, errors="raise")
    return patients_IDs


def get_patient_consent(patient_ID: pd.DataFrame, data_dict: pd.DataFrame) -> Consent:
    """ Get effectve patient consent status, i.e., only if the consent is given
        at the last consent date
    """
    # Get all consent statuses and dates (and assumes no consent if missing)
    all_consents = data_dict.loc[data_dict["patid"] == patient_ID][["consdate", "consstatus"]]
    if len(all_consents) < 1:
        return Consent.CONSENT_REFUSED

    # Extract latest consent status as the effective patient consent status
    all_consents["consdate"] = pd.to_datetime(all_consents["consdate"], errors="coerce")
    latest_consent_status = all_consents.sort_values("consdate").iloc[-1]["consstatus"]
    
    if latest_consent_status != "Consent given":
        return Consent.CONSENT_REFUSED
    
    return Consent.CONSENT_GIVEN
    