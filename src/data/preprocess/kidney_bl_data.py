import pandas as pd
from src.data.data_utils import *


##############
# DATES DATA #
##############

def get_donor_birth_date(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ NOTE: NOT USED SINCE WE USE DONAGE INSTEAD
        NOTE: SHOULD WE HAVE A TOKEN FOR EVERY PATIENT BIRTHDAYS?
    """
    donbirthdate = get_date_by_key(patient_ID, data, "donbirthdate")
    return donbirthdate.rename(columns={"donbirthdate": "Donor birthdate"})

def get_hospitalization_start_date(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    hospstart = get_date_by_key(patient_ID, data, "hospstart")
    return hospstart.rename(columns={"hospstart": "Hospitalization start"})

def get_transplantation_date(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    tpxdate = get_date_by_key(patient_ID, data, "tpxdate")
    return tpxdate.rename(columns={"tpxdate": "Transplantation event"})

def get_hospitalization_end_date(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    hospend = get_date_by_key(patient_ID, data, "hospend")
    return hospend.rename(columns={"hospend": "Hospitalization end"})


###################################
# ORGAN TRANSPLANT AND MATCH DATA #
###################################

def get_kidney_resection_status(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    resec = get_categorical_feature_by_key(patient_ID, data, "resecstcs_organtype", ["First", "Re", "Sec"])
    return resec.rename(columns={"resecstcs_organtype": "Organ resection status"})

def get_kidney_counter(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    org_count = get_numerical_feature_by_key(patient_ID, data, "organtype_counter", (1, 10))
    return org_count.rename(columns={"organtype_counter": "Organ counter"})

def get_kidney_warm_ischemia_time_primary(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ NOTE: WHAT IS THE RELATIONSHIP BETWEEN THIS FIELD, AND THE FIELD WARM ISCHEMIA ("WARMISCH")?
    """
    witprim = get_numerical_feature_by_key(patient_ID, data, "witprim", (0, 200))
    return witprim.rename(columns={"witprim": "Warm ischemia time - primary [min]"})

def get_kidney_warm_ischemia_time(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    warmisch = get_numerical_feature_by_key(patient_ID, data, "warmisch", (0, 200))
    return warmisch.rename(columns={"warmisch": "Warm ischemia time [min]"})

def get_kidney_cold_ischemia_time(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    coldisch = get_numerical_feature_by_key(patient_ID, data, "coldischmin", (0, 4000))
    return coldisch.rename(columns={"coldischmin": "Cold ischemia time [min]"})

def get_kidney_cold_ischemia_time_other_kidney(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ TODO: CHECK IF WE WANT TO NAME IT THE SAME AS FOR COLD_ISCHEMIA_TIME
    """
    coldisch_2 = get_numerical_feature_by_key(patient_ID, data, "coldischmin2", (0, 4000))
    return coldisch_2.rename(columns={"coldischmin2": "Cold ischemia time (other kidney) [min]"})

def get_hla_a_mismatch(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    hlaa_mismatch = get_numerical_feature_by_key(patient_ID, data, "hlaamismatch", (0, 5))
    return hlaa_mismatch.rename(columns={"hlaamismatch": "Number HLA-A mismatch"})

def get_hla_b_mismatch(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    hlab_mismatch = get_numerical_feature_by_key(patient_ID, data, "hlabmismatch", (0, 5))
    return hlab_mismatch.rename(columns={"hlabmismatch": "Number HLA-B mismatch"})

def get_hla_dr_mismatch(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    hladr_mismatch = get_numerical_feature_by_key(patient_ID, data, "hladrmismatch", (0, 5))
    return hladr_mismatch.rename(columns={"hladrmismatch": "Number HLA-DR mismatch"})

def get_sum_hla_mismatch(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    sum_hla_mismatch = get_numerical_feature_by_key(patient_ID, data, "sumhlamismatch", (0, 20))
    return sum_hla_mismatch.rename(columns={"sumhlamismatch": "Sum HLA mismatch"})


######################
# GENERAL DONOR DATA #
######################

def get_donor_sex(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    donsex = get_categorical_feature_by_key(patient_ID, data, "donsex", ["Female", "Male"])
    return donsex.rename(columns={"donsex": "Donor sex"})

def get_donor_age(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    donage = get_numerical_feature_by_key(patient_ID, data, "donage", (0, 150))
    return donage.rename(columns={"donage": "Donor age at transplant [years]"})

def get_donor_type(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:    
    valid_categories = ["Brain dead", "Living unrelated", "Living related", "NHBD", "KPD"]
    dontype = get_categorical_feature_by_key(patient_ID, data, "dontype", valid_categories)
    return dontype.rename(columns={"dontype": "Donor type"})

def get_donor_cause_of_death(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    # INFO: "Not applicable" is clinically relevant, because it means the donor is alive
    valid_categories = [
        "CHE", "ANX", "CTR", "CDI", "SUI", "CTU",
        "Other", "Unknown", "Not applicable",
    ]
    doncod = get_categorical_feature_by_key(patient_ID, data, "doncod", valid_categories)
    return doncod.rename(columns={"doncod": "Donor cause of death"})

def get_donor_blood_group(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    donorbg = get_categorical_feature_by_key(patient_ID, data, "donorbg", ["A", "0", "B", "AB"])
    return donorbg.rename(columns={"donorbg": "Donor blood group"})

def get_donor_is_extended_pool(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    donorpool = get_categorical_feature_by_key(patient_ID, data, "donorpool", ["No", "Yes"])
    return donorpool.rename(columns={"donorpool": "Donor from extended pool"})


#########################
# GENERAL RECEIVER DATA #
#########################

def get_centre_id(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Center entering the patient data
    """
    valid_categories = ["USZ", "USB", "CHUV", "BE", "HUG", "SG"]
    centreid = get_categorical_feature_by_key(patient_ID, data, "centreid", valid_categories)
    return centreid.rename(columns={"centreid": "Organ data entry centre"})

def get_transplantation_type(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Type of kidney transplant (right, left, double, unknown)
        - NOTE: Should consider not using this function and rather use it as a "value" of transplantation event?
        - NOTE: For all double-kidney transplants, the donor is either "Brain dead" or "NHBD" (non-heart beating donor)
        - NOTE: So, that means that even for double-kidney transplants, there is only one donor corresponding
    """
    valid_categories = ["Right", "Left", "Double", "Unknown"]
    tpxtype = get_categorical_feature_by_key(patient_ID, data, "tpxtype", valid_categories)
    return tpxtype.rename(columns={"tpxtype": "Transplantation type"})

def get_had_blood_transfusion_event(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    ec = get_categorical_feature_by_key(patient_ID, data, "ec", ["Yes", "No", "Unknown"])
    return ec.rename(columns={"ec": "Receiver blood transfusion event"})

def get_had_pregnancy_event(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    ss = get_categorical_feature_by_key(patient_ID, data, "ss", ["Yes", "No", "Unknown"])
    return ss.rename(columns={"ss": "Receiver pregnancy event"})


##########################
# RECEIVER ANTIGEN TESTS #
##########################

def get_surface_antigen_for_hbv_receiver(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    hbsag = get_categorical_feature_by_key(patient_ID, data, "hbsag", ["Positive", "Negative", "Unknown"])
    return hbsag.rename(columns={"hbsag": "Receiver HBV Ag test"})

def get_surface_antibody_for_hbv_receiver(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    antihbs = get_categorical_feature_by_key(patient_ID, data, "antihbs", ["Positive", "Negative", "Unknown"])
    return antihbs.rename(columns={"antihbs": "Receiver HBV surf. Ab test"})

def get_core_antibody_for_hbv_receiver(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    antihbc = get_categorical_feature_by_key(patient_ID, data, "antihbc", ["Positive", "Negative", "Unknown"])
    return antihbc.rename(columns={"antihbc": "Receiver HBV core Ab test"})

def get_antibody_for_hcv_receiver(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    antihcv = get_categorical_feature_by_key(patient_ID, data, "antihcv", ["Positive", "Negative", "Unknown"])
    return antihcv.rename(columns={"antihcv": "Receiver HCV Ab test"})

def get_antibody_for_cmv_receiver(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    anticmv = get_categorical_feature_by_key(patient_ID, data, "anticmv", ["Positive", "Negative", "Unknown"])
    return anticmv.rename(columns={"anticmv": "Receiver CMV Ab test"})

def get_antibody_for_ebv_receiver(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    antiebv = get_categorical_feature_by_key(patient_ID, data, "antiebv", ["Positive", "Negative", "Unknown"])
    return antiebv.rename(columns={"antiebv": "Receiver EBV Ab test"})

def get_antibody_for_hiv_receiver(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    antihiv = get_categorical_feature_by_key(patient_ID, data, "antihiv", ["Positive", "Negative", "Unknown"])
    return antihiv.rename(columns={"antihiv": "Receiver HIV Ab test"})

def get_antibody_for_toxo_receiver(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    antitoxo = get_categorical_feature_by_key(patient_ID, data, "antitoxo", ["Positive", "Negative", "Unknown"])
    return antitoxo.rename(columns={"antitoxo": "Receiver toxo. Ab test"})

def get_antibody_for_vzv_receiver(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    antivzv = get_categorical_feature_by_key(patient_ID, data, "antivzv", ["Positive", "Negative", "Unknown"])
    return antivzv.rename(columns={"antivzv": "Receiver VZV Ab test"})

def get_antibody_for_hsv_receiver(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    antihsv = get_categorical_feature_by_key(patient_ID, data, "antihsv", ["Positive", "Negative", "Unknown"])
    return antihsv.rename(columns={"antihsv": "Receiver HSV Ab test"})

def get_antibody_for_trep_receiver(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    tpha = get_categorical_feature_by_key(patient_ID, data, "tpha", ["Positive", "Negative", "Unknown"])
    return tpha.rename(columns={"tpha": "Receiver trep. Pal. test"})


#######################
# DONOR ANTIGEN TESTS #
#######################

def get_surface_antigen_for_hbv_donor(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    donhbsag = get_categorical_feature_by_key(patient_ID, data, "donhbsag", ["Positive", "Negative", "Unknown"])
    return donhbsag.rename(columns={"donhbsag": "Donor HBV Ag test"})

def get_surface_antibody_for_hbv_donor(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    donantihbs = get_categorical_feature_by_key(patient_ID, data, "donantihbs", ["Positive", "Negative", "Unknown"])
    return donantihbs.rename(columns={"donantihbs": "Donor HBV surf. Ab test"})

def get_core_antibody_for_hbv_donor(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    donantihbc = get_categorical_feature_by_key(patient_ID, data, "donantihbc", ["Positive", "Negative", "Unknown"])
    return donantihbc.rename(columns={"donantihbc": "Donor HBV core Ab test"})

def get_antibody_for_hcv_donor(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    donantihcv = get_categorical_feature_by_key(patient_ID, data, "donantihcv", ["Positive", "Negative", "Unknown"])
    return donantihcv.rename(columns={"donantihcv": "Donor HCV Ab test"})

def get_antibody_for_cmv_donor(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    donanticmv = get_categorical_feature_by_key(patient_ID, data, "donanticmv", ["Positive", "Negative", "Unknown"])
    return donanticmv.rename(columns={"donanticmv": "Donor CMV Ab test"})

def get_antibody_for_ebv_donor(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    donantiebv = get_categorical_feature_by_key(patient_ID, data, "donantiebv", ["Positive", "Negative", "Unknown"])
    return donantiebv.rename(columns={"donantiebv": "Donor EBV Ab test"})

def get_antibody_for_hiv_donor(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    donantihiv = get_categorical_feature_by_key(patient_ID, data, "donantihiv", ["Positive", "Negative", "Unknown"])
    return donantihiv.rename(columns={"donantihiv": "Donor HIV Ab test"})

def get_antibody_for_toxo_donor(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    donantitoxo = get_categorical_feature_by_key(patient_ID, data, "donantitoxo", ["Positive", "Negative", "Unknown"])
    return donantitoxo.rename(columns={"donantitoxo": "Donor toxo. Ab test"})

def get_antibody_for_vzv_donor(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    donantivzv = get_categorical_feature_by_key(patient_ID, data, "donantivzv", ["Positive", "Negative", "Unknown"])
    return donantivzv.rename(columns={"donantivzv": "Donor VZV Ab test"})

def get_antibody_for_hsv_donor(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    donantihsv = get_categorical_feature_by_key(patient_ID, data, "donantihsv", ["Positive", "Negative", "Unknown"])
    return donantihsv.rename(columns={"donantihsv": "Donor HSV Ab test"})


#####################
# LONGITUDINAL DATA #
#####################

def get_immuno_test_hla_antibodies(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Retrieve results from immunological tests (human leukocyte antigen - HLA - antibodies)
        NOTE: WHICH OF IMMUNUM VS IMMURES IS THE MOST RELEVANT?
        NOTE: IS IMMUMETH_# A FIELD CLINICALLY RELEVANT FOR INFECTION PREDICTION?
    """
    # The test names are given by the fields "immu_#"
    immu_tests = get_longitudinal_data(
        patient_ID=patient_ID,
        data=data,
        time_key="immudate",
        value_key="immures",
        attribute_key="immu",
    )
    
    # Some "immu_#" are "Unknown", all with "Unknown" as value, and "NaT" as time
    immu_tests = immu_tests[immu_tests["attribute"] != "Unknown"]
    
    # Keep where the attribute comes from in the attribute key
    immu_tests["attribute"] = immu_tests["attribute"].apply(lambda s: f"Immuno test - {s}")
    
    return immu_tests

def get_etiology(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Get the cause / reason for kidney transplantation
    """
    return get_longitudinal_data(patient_ID, data, "etiodate", "etio").assign(attribute="Etiology")

def get_etiology_histology_confirmation(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ NOTE: IS THIS FIELD USEFUL OR IS THE ETIODATE/ETIO COMBO SUFFICIENT?
    """
    etiohisto = get_longitudinal_data(patient_ID, data, "etiohistodate", "etiohisto")
    return etiohisto.assign(attribute="Etiology histology")

def get_initial_dialysis(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Get date and type of initial dialysis
    """
    dialysis_type = get_time_value_pairs(
        patient_ID=patient_ID,
        data=data,
        time_key="dialysisdate",
        value_key="dialysistype",
        valid_categories=["HD", "PD"],
    )

    return dialysis_type.assign(attribute="Initial dialysis type")



#########################
# DATA POOLING FUNCTION #
#########################

def pool_kidney_bl_data(
    patient_ID: int,
    kidney_bl_df: pd.DataFrame,
) -> pd.DataFrame:
    """ Get date, static, and longitudinal kidney baseline data for one patient
    """
    # Build dates dataframe
    kbl_dates = concatenate_clinical_information([
        df.melt(var_name="attribute", value_name="time") for df in [
            # get_donor_birth_date(patient_ID, kidney_bl_df),  # <- we already have donor age
            get_hospitalization_start_date(patient_ID, kidney_bl_df),
            get_transplantation_date(patient_ID, kidney_bl_df),
            get_hospitalization_end_date(patient_ID, kidney_bl_df),
        ]
    ]).assign(value="Occured")
    
    # Build static features dataframe
    kbl_statics = concatenate_clinical_information([
        df.melt(var_name="attribute", value_name="value") for df in [
            get_kidney_resection_status(patient_ID, kidney_bl_df),
            get_kidney_counter(patient_ID, kidney_bl_df),
            get_kidney_warm_ischemia_time(patient_ID, kidney_bl_df),
            get_kidney_cold_ischemia_time(patient_ID, kidney_bl_df),
            get_kidney_cold_ischemia_time_other_kidney(patient_ID, kidney_bl_df),
            get_kidney_warm_ischemia_time_primary(patient_ID, kidney_bl_df),
            get_hla_a_mismatch(patient_ID, kidney_bl_df),
            get_hla_b_mismatch(patient_ID, kidney_bl_df),
            get_hla_dr_mismatch(patient_ID, kidney_bl_df),
            get_sum_hla_mismatch(patient_ID, kidney_bl_df),

            get_donor_sex(patient_ID, kidney_bl_df),
            get_donor_age(patient_ID, kidney_bl_df),
            get_donor_type(patient_ID, kidney_bl_df),
            get_donor_cause_of_death(patient_ID, kidney_bl_df),
            get_donor_blood_group(patient_ID, kidney_bl_df),
            get_donor_is_extended_pool(patient_ID, kidney_bl_df),
            get_centre_id(patient_ID, kidney_bl_df),
            get_transplantation_type(patient_ID, kidney_bl_df),
            get_had_blood_transfusion_event(patient_ID, kidney_bl_df),
            get_had_pregnancy_event(patient_ID, kidney_bl_df),

            get_surface_antigen_for_hbv_receiver(patient_ID, kidney_bl_df),
            get_surface_antibody_for_hbv_receiver(patient_ID, kidney_bl_df),
            get_core_antibody_for_hbv_receiver(patient_ID, kidney_bl_df),
            get_antibody_for_hcv_receiver(patient_ID, kidney_bl_df),
            get_antibody_for_cmv_receiver(patient_ID, kidney_bl_df),
            get_antibody_for_ebv_receiver(patient_ID, kidney_bl_df),
            get_antibody_for_hiv_receiver(patient_ID, kidney_bl_df),
            get_antibody_for_toxo_receiver(patient_ID, kidney_bl_df),
            get_antibody_for_vzv_receiver(patient_ID, kidney_bl_df),
            get_antibody_for_hsv_receiver(patient_ID, kidney_bl_df),
            get_antibody_for_trep_receiver(patient_ID, kidney_bl_df),

            get_surface_antigen_for_hbv_donor(patient_ID, kidney_bl_df),
            get_surface_antibody_for_hbv_donor(patient_ID, kidney_bl_df),
            get_core_antibody_for_hbv_donor(patient_ID, kidney_bl_df),
            get_antibody_for_hcv_donor(patient_ID, kidney_bl_df),
            get_antibody_for_cmv_donor(patient_ID, kidney_bl_df),
            get_antibody_for_ebv_donor(patient_ID, kidney_bl_df),
            get_antibody_for_hiv_donor(patient_ID, kidney_bl_df),
            get_antibody_for_toxo_donor(patient_ID, kidney_bl_df),
            get_antibody_for_vzv_donor(patient_ID, kidney_bl_df),
            get_antibody_for_hsv_donor(patient_ID, kidney_bl_df),
        ]
    ]).assign(time=pd.NaT)

    # Build longitudinal features dataframe
    kbl_longitudinals = concatenate_clinical_information([
        get_initial_dialysis(patient_ID, kidney_bl_df),
        get_immuno_test_hla_antibodies(patient_ID, kidney_bl_df),
        get_etiology(patient_ID, kidney_bl_df),
        get_etiology_histology_confirmation(patient_ID, kidney_bl_df),  # not sure if this one is important but why not
    ])

    # Finalize patient dataframe
    patient_kbl_df = concatenate_clinical_information([kbl_dates, kbl_statics, kbl_longitudinals])
    patient_kbl_df = patient_kbl_df.drop_duplicates()
    patient_kbl_df = patient_kbl_df.assign(entity="Kidney baseline info")  # TODO: CHECK FOR MORE FINE-GRAINED ENTITY ASSIGNATION STRATEGY
    patient_kbl_df = patient_kbl_df.sort_values(by=["time"])
    patient_kbl_df = patient_kbl_df[["entity", "attribute", "value", "time"]]

    return patient_kbl_df
