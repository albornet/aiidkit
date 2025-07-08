import pandas as pd
from src.data.data_utils import *


##############
# DATES DATA #
##############

def get_donor_birth_date(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    # TODO: check if this is useful? we have "donage"
    return get_date_by_key(patient_ID, data, "donbirthdate")

def get_initial_dialysis_start_date(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    # NOTE: NOT USED SINCE IT IS USED BELOW DIRECTLY IN THE LONGITUDINAL GET FUNCTION
    return get_date_by_key(patient_ID, data, "dialysisdate")

def get_hospitalization_start_date(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_date_by_key(patient_ID, data, "hospstart")

def get_transplantation_date(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_date_by_key(patient_ID, data, "tpxdate")

def get_hospitalization_end_date(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_date_by_key(patient_ID, data, "hospend")


###################################
# ORGAN TRANSPLANT AND MATCH DATA #
###################################

def get_organ_resection_status(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_categorical_feature_by_key(patient_ID, data, "resecstcs_organtype", ["First", "Re", "Sec"])

def get_organ_counter(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_numerical_feature_by_key(patient_ID, data, "organtype_counter", (1, 10))

def get_witprim(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    """ QUESTION: WHAT IS THE RELATIONSHIP BETWEEN THIS FIELD, AND THE FIELD WARM ISCHEMIA ("WARMISCH")?
    """
    # TODO: check relation with warmisch
    return get_numerical_feature_by_key(patient_ID, data, "witprim", (0, 200))

def get_organ_warm_ischemia_time(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_numerical_feature_by_key(patient_ID, data, "warmisch", (0, 200))

def get_organ_cold_ischemia_time(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_numerical_feature_by_key(patient_ID, data, "coldischmin", (0, 2000))

def get_organ_cold_ischemia_time_2(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    # TODO: CHECK IF WE WANT TO NAME IT THE SAME AS FOR COLD_ISCHEMIA_TIME
    return get_numerical_feature_by_key(patient_ID, data, "coldischmin2", (0, 2000))

def get_hla_a_mismatch(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_numerical_feature_by_key(patient_ID, data, "hlaamismatch", (0, 5))

def get_hla_b_mismatch(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_numerical_feature_by_key(patient_ID, data, "hlabmismatch", (0, 5))

def get_hla_dr_mismatch(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_numerical_feature_by_key(patient_ID, data, "hladrmismatch", (0, 5))

def get_sum_hla_mismatch(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_numerical_feature_by_key(patient_ID, data, "sumhlamismatch", (0, 20))


######################
# GENERAL DONOR DATA #
######################

def get_donor_sex(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_categorical_feature_by_key(patient_ID, data, "donsex", ["Female", "Male"])

def get_donor_age(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_numerical_feature_by_key(patient_ID, data, "donage", (0, 150))

def get_donor_type(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:    
    valid_categories = ["Brain dead", "Living unrelated", "Living related", "NHBD", "KPD"]
    return get_categorical_feature_by_key(patient_ID, data, "dontype", valid_categories)

def get_donor_cause_of_death(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    # INFO: "Not applicable" is clinically relevant, because it means the donor is alive
    valid_categories = [
        "CHE", "ANX", "CTR", "CDI", "SUI", "CTU",
        "Other", "Unknown", "Not applicable",
    ]
    return get_categorical_feature_by_key(patient_ID, data, "doncod", valid_categories)

def get_donor_blood_group(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_categorical_feature_by_key(patient_ID, data, "donorbg", ["A", "0", "B", "AB"])

def get_donor_is_extended_pool(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_categorical_feature_by_key(patient_ID, data, "donorpool", ["No", "Yes"])


#########################
# GENERAL RECEIVER DATA #
#########################

def get_centre_id(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    """ Center entering the patient data
    """
    return get_categorical_feature_by_key(patient_ID, data, "centreid", ["USZ", "USB", "CHUV", "BE", "HUG", "SG"])

def get_initial_dialysis_type(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    # INFO: NOT USED SINCE IT IS USED BELOW DIRECTLY IN THE LONGITUDINAL GET FUNCTION
    return get_categorical_feature_by_key(patient_ID, data, "dialysistype", ["HD", "PD"])

def get_transplantation_type(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    """ Type of kidney transplant (right, left, double, unknown)
        - NOTE: Should consider not using this function and rather use it as a "value" of transplantation event?
        - NOTE: For all double-kidney transplants, the donor is either "Brain dead" or "NHBD" (non-heart beating donor)
        - NOTE: So, that means that even for double-kidney transplants, there is only one donor corresponding
    """
    return get_categorical_feature_by_key(patient_ID, data, "tpxtype", ["Right", "Left", "Double", "Unknown"])

def get_had_blood_transfusion_event(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_categorical_feature_by_key(patient_ID, data, "ec", ["Yes", "No", "Unknown"])

def get_had_pregnancy_event(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_categorical_feature_by_key(patient_ID, data, "ss", ["Yes", "No", "Unknown"])


##########################
# RECEIVER ANTIGEN TESTS #
##########################

def get_hbs_antigen(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_categorical_feature_by_key(patient_ID, data, "hbsag", ["Positive", "Negative", "Unknown"])

def get_antibody_for_hbs_antigen(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_categorical_feature_by_key(patient_ID, data, "antihbs", ["Positive", "Negative", "Unknown"])

def get_antibody_for_hbc_antigen(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_categorical_feature_by_key(patient_ID, data, "antihbc", ["Positive", "Negative", "Unknown"])

def get_antibody_for_hcv_antigen(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_categorical_feature_by_key(patient_ID, data, "antihcv", ["Positive", "Negative", "Unknown"])

def get_antibody_for_cmv_antigen(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_categorical_feature_by_key(patient_ID, data, "anticmv", ["Positive", "Negative", "Unknown"])

def get_antibody_for_ebv_antigen(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_categorical_feature_by_key(patient_ID, data, "antiebv", ["Positive", "Negative", "Unknown"])

def get_antibody_for_hiv_antigen(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_categorical_feature_by_key(patient_ID, data, "antihiv", ["Positive", "Negative", "Unknown"])

def get_antibody_for_toxo_antigen(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_categorical_feature_by_key(patient_ID, data, "antitoxo", ["Positive", "Negative", "Unknown"])

def get_antibody_for_vzv_antigen(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_categorical_feature_by_key(patient_ID, data, "antivzv", ["Positive", "Negative", "Unknown"])

def get_antibody_for_hsv_antigen(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_categorical_feature_by_key(patient_ID, data, "antihsv", ["Positive", "Negative", "Unknown"])

def get_antibody_for_trep_antigen(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_categorical_feature_by_key(patient_ID, data, "tpha", ["Positive", "Negative", "Unknown"])


#######################
# DONOR ANTIGEN TESTS #
#######################

def get_hbs_antigen_donor(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_categorical_feature_by_key(patient_ID, data, "donhbsag", ["Positive", "Negative", "Unknown"])

def get_antibody_for_hbs_antigen_donor(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_categorical_feature_by_key(patient_ID, data, "donantihbs", ["Positive", "Negative", "Unknown"])

def get_antibody_for_hbc_antigen_donor(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_categorical_feature_by_key(patient_ID, data, "donantihbc", ["Positive", "Negative", "Unknown"])

def get_antibody_for_hcv_antigen_donor(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_categorical_feature_by_key(patient_ID, data, "donantihcv", ["Positive", "Negative", "Unknown"])

def get_antibody_for_cmv_antigen_donor(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_categorical_feature_by_key(patient_ID, data, "donanticmv", ["Positive", "Negative", "Unknown"])

def get_antibody_for_ebv_antigen_donor(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_categorical_feature_by_key(patient_ID, data, "donantiebv", ["Positive", "Negative", "Unknown"])

def get_antibody_for_hiv_antigen_donor(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_categorical_feature_by_key(patient_ID, data, "donantihiv", ["Positive", "Negative", "Unknown"])

def get_antibody_for_toxo_antigen_donor(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_categorical_feature_by_key(patient_ID, data, "donantitoxo", ["Positive", "Negative", "Unknown"])

def get_antibody_for_vzv_antigen_donor(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_categorical_feature_by_key(patient_ID, data, "donantivzv", ["Positive", "Negative", "Unknown"])

def get_antibody_for_hsv_antigen_donor(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    return get_categorical_feature_by_key(patient_ID, data, "donantihsv", ["Positive", "Negative", "Unknown"])


#####################
# LONGITUDINAL DATA #
#####################

# def get_immuno_test_highest_pra(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
#     """ Retrieve results from immunological tests (peak of panel reactive antibody - PRA)
#         QUESTION: WHICH OF IMMUNUM VS IMMURES IS THE MOST RELEVANT?
#         ANSWER -> USED IMMURES AND NOT IMMUNUM (so: not using this function)
#                   BECAUSE THIS ONE HAS TOO FEW ACTUAL + NUMERICAL VALUES
#     """
#     # The test names are given by the fields "immu_#"
#     return get_longitudinal_data(patient_ID, data, "immudate", "immunum", attribute_key="immu")

def get_immuno_test_hla_antibodies(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    """ Retrieve results from immunological tests (human leukocyte antigen - HLA - antibodies)
        QUESTION: WHICH OF IMMUNUM VS IMMURES IS THE MOST RELEVANT?
        QUESTION: IS IMMUMETH_# A FIELD CLINICALLY RELEVANT FOR INFECTION PREDICTION?
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
    immu_tests["attribute"] = immu_tests["attribute"].apply(lambda s: f"Immu Test - {s}")
    
    return immu_tests

def get_etiology(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    """ ...
    """
    return get_longitudinal_data(patient_ID, data, "etiodate", "etio")

def get_etiology_histology_confirmation(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    """ QUESTION: IS THIS FIELD USEFUL OR IS THE ETIODATE/ETIO COMBO SUFFICIENT?
    """
    return get_longitudinal_data(patient_ID, data, "etiohistodate", "etiohisto")

def get_initial_dialysis(patient_ID:int, data:pd.DataFrame) -> pd.DataFrame:
    # INFO: this feature set doesn't have the same longitudinal structure
    #       so it is queried in a different way (i.e., pairing values with dates)
    return get_time_value_pairs(patient_ID, data, "dialysisdate", "dialysistype", valid_categories=["HD", "PD"])


#########################
# DATA POOLING FUNCTION #
#########################

def pool_kidney_bl_data(
    patient_ID:int,
    kidney_bl_df:pd.DataFrame,
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
    ]).assign(value=pd.NA)
    
    # Build static features dataframe
    kbl_statics = concatenate_clinical_information([
        df.melt(var_name="attribute", value_name="value") for df in [
            get_organ_resection_status(patient_ID, kidney_bl_df),
            get_organ_counter(patient_ID, kidney_bl_df),
            get_organ_warm_ischemia_time(patient_ID, kidney_bl_df),
            get_organ_cold_ischemia_time_2(patient_ID, kidney_bl_df),
            get_witprim(patient_ID, kidney_bl_df),
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
            get_hbs_antigen(patient_ID, kidney_bl_df),
            get_antibody_for_hbs_antigen(patient_ID, kidney_bl_df),
            get_antibody_for_hbc_antigen(patient_ID, kidney_bl_df),
            get_antibody_for_hcv_antigen(patient_ID, kidney_bl_df),
            get_antibody_for_cmv_antigen(patient_ID, kidney_bl_df),
            get_antibody_for_ebv_antigen(patient_ID, kidney_bl_df),
            get_antibody_for_hiv_antigen(patient_ID, kidney_bl_df),
            get_antibody_for_toxo_antigen(patient_ID, kidney_bl_df),
            get_antibody_for_vzv_antigen(patient_ID, kidney_bl_df),
            get_antibody_for_trep_antigen(patient_ID, kidney_bl_df),
            get_hbs_antigen_donor(patient_ID, kidney_bl_df),
            get_antibody_for_hbs_antigen_donor(patient_ID, kidney_bl_df),
            get_antibody_for_hbc_antigen_donor(patient_ID, kidney_bl_df),
            get_antibody_for_hcv_antigen_donor(patient_ID, kidney_bl_df),
            get_antibody_for_cmv_antigen_donor(patient_ID, kidney_bl_df),
            get_antibody_for_ebv_antigen_donor(patient_ID, kidney_bl_df),
            get_antibody_for_hiv_antigen_donor(patient_ID, kidney_bl_df),
            get_antibody_for_toxo_antigen_donor(patient_ID, kidney_bl_df),
            get_antibody_for_vzv_antigen_donor(patient_ID, kidney_bl_df),
            get_antibody_for_hsv_antigen_donor(patient_ID, kidney_bl_df),
        ]
    ]).assign(time=pd.NaT)

    # Build longitudinal features dataframe
    kbl_longitudinals = concatenate_clinical_information([
        get_initial_dialysis(patient_ID, kidney_bl_df),
        # get_immuno_test_highest_pra(patient_ID, kidney_bl_df),  # keep? -> for now no, since very few values compared to immures
        get_immuno_test_hla_antibodies(patient_ID, kidney_bl_df),
        get_etiology(patient_ID, kidney_bl_df),
        get_etiology_histology_confirmation(patient_ID, kidney_bl_df),  # not sure if this one is important but why not
    ])

    # Finalize patient dataframe
    patient_kbl_df = concatenate_clinical_information([kbl_dates, kbl_statics, kbl_longitudinals])
    patient_kbl_df = patient_kbl_df.drop_duplicates()
    patient_kbl_df = patient_kbl_df.assign(entity="kidney_bl")  # TODO: CHECK FOR MORE FINE-GRAINED ENTITY ASSIGNATION STRATEGY
    patient_kbl_df = patient_kbl_df.sort_values(by=["time"])
    patient_kbl_df = patient_kbl_df[["entity", "attribute", "value", "time"]]

    return patient_kbl_df
