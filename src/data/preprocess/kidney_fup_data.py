import numpy as np
import pandas as pd
from src.data.data_utils import *
from src.data.preprocess.kidney_bl_data import (
    get_transplantation_date as get_transplantation_date_from_kid_bl_module,
    get_immuno_test_hla_antibodies as get_immuno_test_hla_antibodies_from_kid_bl_module,
)


##############
# DATES INFO #
##############

def get_assessment_date(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ NOTE: this could be used for dating biopsy tests (but not used for now)
    """
    assdate = get_date_by_key(patient_ID, data, "assdate")
    return assdate.rename(columns={"assdate": "Kidney FUP assessment date"})

def get_transplantation_date(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ NOTE: not used for now (same info can be obtained in kidney_bl_data and organ_base_data)
        TODO: check that all info sources are consistent
    """
    return get_transplantation_date_from_kid_bl_module(patient_ID, data)


#############################
# LONGITUDINAL PATIENT INFO #
#############################

def get_insufficient_urine_level(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Whether patient has insufficient urine level in the 24 hours after transplantation
    """
    urine24 = get_time_value_pairs(
        patient_ID, data, time_key="tpxdate", value_key="urine24",
        value_type="categorical", valid_categories=["No", "Yes"],
    )
    urine24["time"] = urine24["time"] + pd.Timedelta(days=1)
    return urine24.assign(attribute="Insufficient urine level")

def get_bkv_uremia_level(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ BKV level in copies/ml, from 0 to approx. 10_000_000 (+ nan-like values),
        hence, we transform the value with log_10(value + 1)
        QUESTION: WHY data[data["bkvdesc"] == "Above"]["bkv"] GIVES LOW VALUES?
    """
    bkv = get_time_value_pairs(
        patient_ID, data, time_key="bkvdate", value_key="bkv",
        value_type="numerical", valid_number_range=(0, 10_000_000),
    )
    bkv["value"] = np.log10(bkv["value"].add(1.0))
    return bkv.assign(attribute="BKV uremia level [log-(copies/ml)]")

def get_protein_uria_level(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Protein uria level in copies/ml, from 0 to approx. 4_000 (+ nan-like values)
        hence, we transform the value with log_10(value + 1)
        QUESTION: WHY data[data["proteinuriadesc"] == "Above"]["proteinuria"] GIVES LOW VALUES?
    """
    protein_uria = get_time_value_pairs(
        patient_ID, data, time_key="proteinuriadate", value_key="proteinuria",
        value_type="numerical", valid_number_range=(0, 10_000),
    )
    protein_uria["value"] = np.log10(protein_uria["value"].add(1))
    return protein_uria.assign(attribute="Protein uria level [log-(copies/ml)]")

def get_early_allograft_dysfunction(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Whether patient had a PNF or a DGF event, with DGF duration as DGF value
        Note: there are 25 patients with "PNF" events, for which value will simply be "occured"
        QUESTION: WHAT IS THE UNIT OF dgfduration?
    """
    dgf_data = data.loc[data["patid"] == patient_ID, ["dgf", "dgfduration", "tpxdate"]]
    dgf_data.loc[dgf_data["dgfduration"].isin(csts.NAN_LIKE_NUMBERS), "dgfduration"] = np.nan

    valid_dgf_categories = ["DGF", "PNF"]
    relevant_data = dgf_data[dgf_data["dgf"].isin(valid_dgf_categories)]

    if relevant_data.empty:
        return pd.DataFrame()

    relevant_data = relevant_data.astype(object)
    relevant_data = relevant_data.rename(
        columns={"dgf": "attribute", "dgfduration": "value", "tpxdate": "time"},
    )
    relevant_data.loc[relevant_data["attribute"] == "PNF", "value"] = "Occured"
    relevant_data.loc[relevant_data["attribute"] == "DGF", "attribute"] = "DGF [days?]"
    
    return relevant_data

def get_reason_for_graft_loss(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Graft loss is any event that leads to the removal of the transplanted organ,
        which can include a PNF event, but not only
    """
    reason_for_glo = get_longitudinal_data(patient_ID, data, "glodate", "glo")
    return reason_for_glo.assign(attribute="Reason for graft loss")

def _define_rj_status(value: Union[str, float]) -> str:
    """ If present, add clinical status of rejection event
    """
    if (isinstance(value, float) and np.isnan(value))\
    or value.lower() in ["not applicable", "unknown"]:
        return "Rejection type"
    else:
        return f"Rejection type ({str(value).lower()} status)"

def get_rejection_event(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Get rejection event and associated date, using the rejection event
        clinical status (clinical, subclinical) as an attribute, if existing
    """
    rj_events = get_longitudinal_data(
        patient_ID=patient_ID,
        data=data,
        time_key="rjdate",
        value_key="rj",
        attribute_key="rjclinical",
    )

    rj_events["attribute"] = rj_events["attribute"].apply(_define_rj_status)

    return rj_events

def get_allograft_disease_event(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    alldis = get_longitudinal_data(patient_ID, data, "alldisdate", "alldis")
    return alldis.assign(attribute="Allograft disease")

def get_transplant_related_complication_event(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    tpxcomp = get_longitudinal_data(patient_ID, data, "txcompdate", "txcomp")
    return tpxcomp.assign(attribute="Transplant related complication")

def get_immuno_test_hla_antibodies(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Retrieve results from autoimmune tests (human leukocyte antigen - HLA - antibodies)
    """
    return get_immuno_test_hla_antibodies_from_kid_bl_module(patient_ID, data)


################################
# HISTOLOGY AND IMMUNOLOGY TESTS
################################

def _parse_banff_scores(val: str, prefix: str) -> int:
    """ Parse a string-based Banff score (e.g., "t0", "v1") into the corresponding integer
    """
    if pd.isna(val): return np.nan
    result = val.split(prefix)[-1]

    if result.isdigit():
        return int(result)
    else:
        return np.nan

def _get_raw_banff_scores(patient_ID, data, prefix, biopsy_date_key="immudate"):
    """ Get one particular score from Banff assessment data, assuming its date is
        the same as the immunology test date
        QUESTION: IS THE DATE THE GOOD ONE? OR SHOULD WE TAKE THE FOLLOW-UP ASSDATE?
    """
    banff_scores = get_longitudinal_data(
        patient_ID, data,
        time_key=biopsy_date_key,
        value_key=prefix,
        attribute_key="adequacy",
    )

    # Select only valid test results
    banff_scores = banff_scores[banff_scores["attribute"].apply(str.lower) == "satisfactory"]
    if banff_scores.empty:
        return pd.DataFrame()
    
    # Turn raw scores into int values (from 0 to 3)
    banff_scores["value"] = banff_scores["value"].apply(lambda x: _parse_banff_scores(x, prefix))
    
    # Drop the adequacy information (all satisfactory) and retrieve the score name
    banff_scores["attribute"] = f"Banff score - {prefix}"

    return banff_scores

def get_banff_results(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Retrieve all possible scores from Banff assessment data for a given patient
        Each score is defined as such: 0: no lesion, 1: mild, 2: moderate, 3: severe
    """
    all_scores_list = [

        # Tubulointerstitial lesions (core for T-cell mediated rejection and chronicity)
        _get_raw_banff_scores(patient_ID, data, "i"),   # Interstitial inflammation (acute)
        _get_raw_banff_scores(patient_ID, data, "t"),   # Tubulitis (acute)
        _get_raw_banff_scores(patient_ID, data, "ti"),  # Total inflammation score (derived from i, t)
        _get_raw_banff_scores(patient_ID, data, "ci"),  # Interstitial fibrosis (chronic)
        _get_raw_banff_scores(patient_ID, data, "ct"),  # Tubular atrophy (chronic)

        # Vascular lesions (key for severe acute rejection and chronic vascular changes)
        _get_raw_banff_scores(patient_ID, data, "v"),   # Intimal arteritis score (acute)
        _get_raw_banff_scores(patient_ID, data, "ptc"), # Peritubular capillaritis (acute, ABMR marker)
        _get_raw_banff_scores(patient_ID, data, "cv"),  # Fibrous intimal thickening (chronic)
        _get_raw_banff_scores(patient_ID, data, "ah"),  # Arteriolar hyalinosis (chronic)
        _get_raw_banff_scores(patient_ID, data, "aah"), # Alternative arteriolar hyalinosis (chronic)

        # Glomerular lesions (key for antibody-mediated rejection)
        _get_raw_banff_scores(patient_ID, data, "g"),   # Glomerulitis (acute)
        _get_raw_banff_scores(patient_ID, data, "cg"),  # Glomerulopathy (chronic, often ABMR-related)
        _get_raw_banff_scores(patient_ID, data, "mm"),  # Mesangial matrix increase (chronic/non-specific)

        # Immunohistochemical / complement deposition (direct marker of ABMR)
        _get_raw_banff_scores(patient_ID, data, "c4d"), # C4d staining (acute/active ABMR biomarker)
    
    ]

    return pd.concat(all_scores_list, ignore_index=True)


#########################
# DATA POOLING FUNCTION #
#########################

def pool_kidney_fup_data(
    patient_ID: int,
    kidney_fup_df: pd.DataFrame,
) -> pd.DataFrame:
    """ Get date, static, and longitudinal kidney follow-up data for one patient
    """
    # Build paired dataframe (either time-paired or two static features)
    kfup_paired = concatenate_clinical_information([
        get_insufficient_urine_level(patient_ID, kidney_fup_df),
        get_bkv_uremia_level(patient_ID, kidney_fup_df),
        get_protein_uria_level(patient_ID, kidney_fup_df),
        get_early_allograft_dysfunction(patient_ID, kidney_fup_df),
    ])

    # Build longitudinal features dataframe
    kfup_longitudinals = concatenate_clinical_information([
        get_reason_for_graft_loss(patient_ID, kidney_fup_df),
        get_rejection_event(patient_ID, kidney_fup_df),
        get_allograft_disease_event(patient_ID, kidney_fup_df),
        get_transplant_related_complication_event(patient_ID, kidney_fup_df),
        get_immuno_test_hla_antibodies(patient_ID, kidney_fup_df),
        get_banff_results(patient_ID, kidney_fup_df),
    ])

    # Finalize patient dataframe
    patient_kfup_df = concatenate_clinical_information([kfup_paired, kfup_longitudinals])
    patient_kfup_df = patient_kfup_df.drop_duplicates()
    patient_kfup_df = patient_kfup_df.assign(entity="Kidney follow-up info")  # TODO: CHECK FOR MORE FINE-GRAINED ENTITY ASSIGNATION STRATEGY
    patient_kfup_df = patient_kfup_df.sort_values(by=["time"])
    patient_kfup_df = patient_kfup_df[["entity", "attribute", "value", "time"]]

    return patient_kfup_df
