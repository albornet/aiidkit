import pandas as pd
from src.data.preprocess.kidney_bl_data import get_transplantation_date
from src.data.data_utils import *


############################
# PATIENT BASE INFORMATION #
############################

def get_birth_date(patient_ID: int, data:  pd.DataFrame) -> pd.DataFrame:
    birthdate = get_date_by_key(patient_ID, data, "birthday")
    return birthdate.rename(columns={"birthday": "Receiver birthdate"})

def get_age_at_transplantation(patient_ID: int, patient_data: pd.DataFrame, kidney_data: pd.DataFrame) -> pd.DataFrame:
    """ TODO: CHECK IF WE CAN USE A DYNAMIC FUNCTION TO RECALL PATIENT AGE IN SEQUENCES
              LIKE GET_PATIENT_BIRTHDATE_EVENT WHICH RETURNS THE DATE AS TIME, THE AGE AS VALUE
    """
    birthdate = get_birth_date(patient_ID, patient_data).assign(patient_id=patient_ID)
    tpx_date = get_transplantation_date(patient_ID, kidney_data).assign(patient_id=patient_ID)

    merged_df = pd.merge(tpx_date, birthdate, on="patient_id", how="left")
    merged_birthdate = pd.to_datetime(merged_df["Receiver birthdate"])
    merged_tpx_date = pd.to_datetime(merged_df["Transplantation event"])

    age_at_tpx = (merged_tpx_date - merged_birthdate).dt.days / 365.25
    age_at_tpx_df = pd.DataFrame({"Receiver age at transplant [years]": age_at_tpx})

    return age_at_tpx_df

def get_sex(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    sex = get_categorical_feature_by_key(patient_ID, data, "sex", valid_categories=["Female", "Male"])
    return sex.rename(columns={"sex": "Receiver sex"})

def get_ethnicity(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Patient broad-level ethnicity
    """
    valid_categories = ["Caucasian", "Asian", "African", "Other", "Unknown"]
    ethn = get_categorical_feature_by_key(patient_ID, data, "ethnicity", valid_categories)

    ethn_other = get_categorical_feature_by_key(patient_ID, data, "ethnother", valid_categories=None)
    if not ethn_other.empty:
        ethn = ethn.replace("Other", ethn_other.iloc[0].item())
    
    ethn = ethn.map(lambda s: s.lower())
    # Check that all values will indeed be replaced
    if not ethn.empty and not ethn.isin(csts.ETHNICITY_NORMALIZATION_MAP.keys())["ethnicity"].any():
        print(ethn)
    
    # Normalize race to have a consistent format (some come in French, other in German, etc.)
    ethn = ethn.replace(csts.ETHNICITY_NORMALIZATION_MAP)

    return ethn.rename(columns={"ethnicity": "Receiver ethnicity"})

def get_weight(patient_ID: int, data: pd.DataFrame) ->  pd.DataFrame:
    """ Patient weight in kilograms
    """
    weight = get_numerical_feature_by_key(patient_ID, data, "weight", valid_range=(0, 500))
    return weight.rename(columns={"weight": "Receiver weight [kg]"})

def get_height(patient_ID: int, data: pd.DataFrame) ->  pd.DataFrame:
    """ Patient height in centimeters
    """
    height = get_numerical_feature_by_key(patient_ID, data, "height", valid_range=(0, 300))
    return height.rename(columns={"height": "Receiver height [cm]"})

def get_bmi(patient_ID: int, data: pd.DataFrame) ->  pd.DataFrame:
    """ Patient body mass index
        - NOTE: anytime height or weight values are either outside the valid
          range or are NaNs, the BMI value is also NaN
        - NOTE: so, a good strategy is to try to get the BMI, then, only if not
          found, try to compute it from height and weight 
    """
    bmi = get_numerical_feature_by_key(patient_ID, data, "bmi", valid_range=(1, 100))
    if bmi["bmi"].isna().any() or bmi.empty:
        height = get_height(patient_ID, data)  # returns height in centimeters
        weight = get_weight(patient_ID, data)  # returns weight in kilograms
    
        if bmi["bmi"].isna().any() or bmi.empty or bmi["bmi"].isna().any() or bmi.empty:
            return pd.DataFrame()
        else:
            bmi = pd.DataFrame({"bmi": weight["weight"] / (height["height"] / 100) ** 2})
    
    return bmi.rename(columns={"bmi": "Receiver BMI [kg/m^2]"})

def get_household_income(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Get current available monthly budget (after tax) in patient household
        - NOTE: there are 4 values in field "income_1", presumably if the patients
          has several households
        - QUESTION: SHALL WE TAKE THEM, WITH E.G., THE SUM OR AVERAGE?
    """
    valid_categories = ["< 4500", "4501-6000", "6001-9000", "> 9000", "Refused", "Unknown"]
    income = get_categorical_feature_by_key(patient_ID, data, "income_0", valid_categories)
    return income.rename(columns={"income_0": "Receiver income"})


###############################
# PATIENT MEDICAL INFORMATION #
###############################

def get_systolic_blood_pressure(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Patient systolic blood pressure in mmHg
    """
    sbp = get_numerical_feature_by_key(patient_ID, data, "sbp", valid_range=(0, 300))
    return sbp.rename(columns={"sbp": "Receiver SBP [mmHg]"})

def get_diastolic_blood_pressure(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Patient diastolic blood pressure in mmHg
    """
    dbp = get_numerical_feature_by_key(patient_ID, data, "dbp", valid_range=(0, 300))
    return dbp.rename(columns={"dbp": "Receiver DBP [mmHg]"})

def get_reference_centre(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Center of reference at time of transplant
    """
    refcenter = get_categorical_feature_by_key(patient_ID, data, "refcentre", ["USZ", "USB", "CHUV", "BE", "HUG", "SG"])
    return refcenter.rename(columns={"refcentre": "Receiver transplant centre"})

def get_blood_group(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Patient blood group (QUESTION: is it 0 or O?)
    """
    blood_group = get_categorical_feature_by_key(patient_ID, data, "hembg", ["A", "B", "AB", "0"])
    return blood_group.rename(columns={"hembg": "Receiver blood group"})

def get_past_immuno_treatment(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Past immunosuppressive treatment (including systemic corticosteroids)
    """
    immu_treat = get_categorical_feature_by_key(patient_ID, data, "istreat", ["No", "Yes", "Unknown"])
    return immu_treat.rename(columns={"istreat": "Past immunosuppr. treatment"})

def get_drug_addiction(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Any pre-transplant drug addiction, like heroin, cocain, and so on
        NOTE: we only take "ivdruguse_0" since "ivdruguse_1" contains nothing (and no ivdruguse_#, # > 1)
    """
    valid_categories = ["Never", "Stopped > one year ago", "Stopped < one year ago", "Yes", "Refused", "Unknown"]
    drug_use = get_categorical_feature_by_key(patient_ID, data, "ivdruguse_0", valid_categories)
    return drug_use.rename(columns={"ivdruguse_0": "Drug addiction"})


##############################
# PATIENT DATED TEST RESULTS #
##############################

def _categorize_hba1c_level(hba1c_level: float) -> str:
    """ Categorize HbA1c level given clinical guidelines
    """
    return "Normal" if hba1c_level <= 5.7 else ("Prediabetes" if hba1c_level <= 6.5 else "Diabetes")

def _categorize_glucose_level(row: pd.Series) -> str:
    """ Categorize glucose level for fasting/non-fasting patient, given clinical guidelines
    """
    glucose = row["value_gluc"]
    fasting = row["value_glucfast"]

    if fasting == "Fasting":
        return "Normal" if glucose < 5.5 else "Prediabetes" if glucose <= 7.0 else "Diabetes"
    elif fasting in "Random":
        return "Normal" if glucose < 7.0 else "Prediabetes" if glucose <= 11.1 else "Diabetes"
    
    return "Unknown"

def get_hba1c_level(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Get patient HbA1c level [%], with date
    """
    hba1c_level = get_time_value_pairs(
        patient_ID, data,
        value_type="numerical", time_key="hba1cdate", value_key="hba1c",
        valid_number_range=(0, 100),
    )

    return hba1c_level.assign(attribute="HBA1C level [%]")

def get_hba1c_level_test(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Get diabetes test based on average blood HbA1c (glycated hemoglobin) level [%], with date
    """
    hba1c_level_test = get_hba1c_level(patient_ID, data)
    if hba1c_level_test.empty: return pd.DataFrame()

    hba1c_level_test["value"] = hba1c_level_test["value"].apply(_categorize_hba1c_level)

    return hba1c_level_test.assign(attribute="HBA1C test result")

def get_glucose_level(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Get patient glucose level [mmol/L], with date
    """
    return get_time_value_pairs(
        patient_ID, data, value_type="numerical",
        time_key="glucdate", value_key="gluc",
        valid_number_range=(0, 100),
    ).assign(attribute="Glucose level [mmol/L]")

def get_fasting_state(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Get patient fasting state (fasting or random), with date
    """
    return get_time_value_pairs(
        patient_ID, data,
        time_key="glucdate", value_key="glucfast",
        valid_categories=["Random", "Fasting", "Unknown"],
    ).assign(attribute="Fasting state")

def get_glucose_level_test(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Get diabetes test based on both patient blood glucose level [mmol/L]
        and patient fasting status, with date
        - QUESTION: "gluc" field is described as "fasting glucose level", but the
          value of "glucfast" is very often "random" or "unknown" (only 100 fasting)
          so: should we consider these values in the test categorization?
    """
    glucose_level = get_glucose_level(patient_ID, data)
    if glucose_level.empty: return pd.DataFrame()
    
    fasting_state = get_fasting_state(patient_ID, data)
    if fasting_state.empty: return pd.DataFrame()

    merged_df = glucose_level.merge(fasting_state, on="time", how="inner", suffixes=("_gluc", "_glucfast"))
    merged_df["value"] = merged_df.apply(_categorize_glucose_level, axis=1)
    test_result = merged_df.assign(attribute="Glucose test result")
    
    return test_result[["attribute", "value", "time"]]

def get_plasma_creatinine(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Get patient plasma creatinine level in µmol/L, with date
    """
    return get_time_value_pairs(
        patient_ID, data, "creatinindate", "crea",
        value_type="numerical", valid_number_range=(0, 3000),
    ).assign(attribute="Plasma creatinine level [µmol/L]")

def get_ldl_cholesterol_level(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Get patient LDL cholesterol level in mmol/L, with date
        QUESTION: QUELLE DATE PRENDRE? HEMDATE = DATE DU BLOOD SAMPLING, LIPIDDATE = DATE DU LIPID TEST
    """
    return get_time_value_pairs(
        patient_ID, data, "hemdate", "ldlchol",
        value_type="numerical", valid_number_range=(0, 20),
    ).assign(attribute="LDL chol. level [mmol/L]")

def get_hdl_cholesterol_level(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Get patient HDL cholesterol level in mmol/L, with date
        QUESTION: QUELLE DATE PRENDRE? HEMDATE = DATE DU BLOOD SAMPLING, LIPIDDATE = DATE DU LIPID TEST
    """
    return get_time_value_pairs(
        patient_ID, data, "hemdate", "hdlchol",
        value_type="numerical", valid_number_range=(0, 20),
    ).assign(attribute="HDL chol. level [mmol/L]")

def get_total_cholesterol_level(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Get patient total cholesterol level in mmol/L, with date
        QUESTION: QUELLE DATE PRENDRE? HEMDATE = DATE DU BLOOD SAMPLING, LIPIDDATE = DATE DU LIPID TEST
    """
    return get_time_value_pairs(
        patient_ID, data, "hemdate", "chol",
        value_type="numerical", valid_number_range=(0, 20),
    ).assign(attribute="Total chol. level [mmol/L]")


###########################
# PATIENT MEDICAL HISTORY #
###########################

def get_previous_metabolic_endocrine_diseases(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Type of metabolic, endocrine or kidney diagnosis, with associated date
    """
    return get_longitudinal_data(patient_ID, data, "mekdate", "mek").assign(attribute="MEK diagnosis")

def get_previous_skin_cancer(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Skin cancer, with associated date (separated from other cancers)
    """
    return get_longitudinal_data(patient_ID, data, "caskindate", "caskin").assign(attribute="Skin cancer diagnosis")

def get_previous_non_skin_cancer(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Type of non-skin cancer or neoplasia, with associated date
    """
    return get_longitudinal_data(patient_ID, data, "cadate", "ca").assign(attribute="Non-skin cancer diagnosis")

def get_previous_infectious_diseases(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Infectious disease, with associated date
        TODO: MAKE SURE THIS FIELD DOES NOT CONTAIN LEAKING INFORMATION AT "CLOSE" DATES
        TODO: MAKE SURE THIS IS ABOUT INFECTION DISEASE OCCURRENCE(S) PRIOR TO TRANSPLANTATION
    """
    return get_longitudinal_data(patient_ID, data, "iddate", "id").assign(attribute="Inf. disease diagnosis")
    
def get_previously_transplanted_organ(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Type of organ in previous transplantation event, with associated date
    """
    return get_longitudinal_data(patient_ID, data, "pretxdate", "pretx").assign(attribute="Previous tpx event")

def get_previous_cardio_pulmonary_disease(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Cardiovascular & pulmonary diagnosis, with associated date
    """
    return get_longitudinal_data(patient_ID, data, "cpdate", "cp").assign(attribute="Cardio-pulmonary diagnosis")

def get_previous_other_diagnose_events(patient_ID: int, data: pd.DataFrame) -> pd.DataFrame:
    """ Get any "other" event or diagnosis that might be clinically relevant
    """
    return get_longitudinal_data(patient_ID, data, "evdisdate", "evdis").assign(attribute="Any other diagnosis")


#########################
# DATA POOLING FUNCTION #
#########################

def pool_patient_bl_data(
    patient_ID: int,
    patient_bl_df: pd.DataFrame,
    patient_psq_df: pd.DataFrame,
    kidney_bl_df: pd.DataFrame,
) -> pd.DataFrame:
    """ Get date, static, and longitudinal kidney baseline data for one patient
    """
    # Build static features dataframe
    pbl_statics = concatenate_clinical_information([
        df.melt(var_name="attribute", value_name="value") for df in [
            get_age_at_transplantation(patient_ID, patient_bl_df, kidney_bl_df),
            get_sex(patient_ID, patient_bl_df),
            get_ethnicity(patient_ID, patient_psq_df),
            get_weight(patient_ID, patient_bl_df),
            get_height(patient_ID, patient_bl_df),
            get_bmi(patient_ID, patient_bl_df),
            get_household_income(patient_ID, patient_bl_df),
            get_reference_centre(patient_ID, patient_bl_df),
            get_systolic_blood_pressure(patient_ID, patient_bl_df),
            get_diastolic_blood_pressure(patient_ID, patient_bl_df),
            get_blood_group(patient_ID, patient_bl_df),
            get_drug_addiction(patient_ID, patient_bl_df),
            get_past_immuno_treatment(patient_ID, patient_bl_df),
        ]
    ]).assign(time=pd.NaT)
    
    # Build timed features with blood (or so) test results
    pbl_blood_test_results = concatenate_clinical_information([
        get_hba1c_level(patient_ID, patient_bl_df),
        get_hba1c_level_test(patient_ID, patient_bl_df),
        get_glucose_level(patient_ID, patient_bl_df),
        get_glucose_level_test(patient_ID, patient_bl_df),
        get_plasma_creatinine(patient_ID, patient_bl_df),
        get_ldl_cholesterol_level(patient_ID, patient_bl_df),
        get_hdl_cholesterol_level(patient_ID, patient_bl_df),
        get_total_cholesterol_level(patient_ID, patient_bl_df),
    ])

    # Build medical history events
    pbl_medical_history = concatenate_clinical_information([
        get_previous_metabolic_endocrine_diseases(patient_ID, patient_bl_df),
        get_previous_skin_cancer(patient_ID, patient_bl_df),
        get_previous_non_skin_cancer(patient_ID, patient_bl_df),
        get_previous_cardio_pulmonary_disease(patient_ID, patient_bl_df),
        get_previously_transplanted_organ(patient_ID, patient_bl_df),
        get_previous_infectious_diseases(patient_ID, patient_bl_df),
        get_previous_other_diagnose_events(patient_ID, patient_bl_df),
    ])

    # Finalize patient dataframe
    patient_df = concatenate_clinical_information([pbl_statics, pbl_blood_test_results, pbl_medical_history])
    patient_df = patient_df.drop_duplicates()
    patient_df = patient_df.assign(entity="Patient baseline info")  # TODO: CHECK FOR MORE FINE-GRAINED ENTITY ASSIGNATION STRATEGY
    patient_df = patient_df.sort_values(by=["time"])
    patient_df = patient_df[["entity", "attribute", "value", "time"]]

    return patient_df
