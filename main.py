import os
import pandas as pd

from data_processing.data_utils import *
from data_processing.consent_data import *
from data_processing.kidney_fup_data import *
from data_processing.kidney_bl_data import *
from data_processing.patient_stop_data import *
from data_processing.patient_infectious_disease_data import *
from data_processing.data_output import *

from functools import partial
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import argparse
import constants
csts = constants.ConstantsNamespace()
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
args = parser.parse_args()
DEBUG_FLAG = args.debug


def main():
    """ Create patient records from the raw STCS data file
        TODO: CHECK HOW THINGS WORK FOR PATIENTS WITH MULTIPLE TRANSPLANTS AND
              CONSIDER USING ORGAN_ID INSTEAD OF / IN COMBINATION WITH PATIENT_ID
    """
    # Create output directory if it does not exist
    output_dir_path = os.path.dirname(csts.OUTPUT_DIR_PATH)
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    
    # Load patient data sheets from the pickle file
    data_dict = pd.read_pickle(csts.PICKLE_DATA_PATH)
    patients_IDs = get_patients_IDs(data_dict[csts.CONSENT_SHEET])

    # Process some patients, using only one process, if DEBUG_FLAG is enabled
    if DEBUG_FLAG:
        import ipdb; ipdb.set_trace()
        for patient_ID in tqdm(patients_IDs[:1000], "Creating patient records"):
            create_patient_record(patient_ID, data_dict)

    # Process all patients using multiprocesing to create csv records
    else:
        num_workers = os.cpu_count() - 1
        chunksize = max(1, len(patients_IDs) // (4 * num_workers))
        pooled_fn = partial(create_patient_record, data_dict=data_dict)
        process_map(
            pooled_fn,
            patients_IDs,
            max_workers=num_workers,
            desc=f"Creating patient records with {num_workers} workers and chunksize {chunksize}",
            chunksize=chunksize,
        )

    # Success!
    print("Dataset created successfully")


def create_patient_record(patient_ID: int, data_dict: pd.DataFrame) -> pd.DataFrame:
    """ Create a patient record from the raw data
    """
    # Only process patients who have given consent
    consent_status = get_patient_consent(patient_ID, data_dict[csts.CONSENT_SHEET])
    if consent_status == Consent.CONSENT_REFUSED: return None

    # Load patient information from all the sheets
    patient_df = pd.concat([
        # pool_patient_bl_data(patient_ID, data_dict[csts.PATIENT_BL_SHEET]),
        # pool_patient_psq_data(patient_ID, data_dict[csts.PATIENT_PSQ_SHEET]),
        # pool_patient_drug_data(patient_ID, data_dict[csts.PATIENT_DRUG_SHEET]),
        # pool_patient_stop_data(patient_ID, data_dict[csts.PATIENT_STOP_SHEET]),
        # pool_patient_infection_data(patient_ID, data_dict[csts.PATIENT_INFECTION_SHEET]),
        pool_kidney_bl_data(patient_ID, data_dict[csts.KIDNEY_BL_SHEET]),
        pool_kidney_fup_data(patient_ID, data_dict[csts.KIDNEY_FUP_SHEET]),
        # pool_organ_base_data(patient_ID, data_dict[csts.ORGAN_BASE_SHEET]),
    ], ignore_index=True)
    patient_df = patient_df.drop_duplicates(subset=["attribute", "value", "time"])
    patient_df = patient_df.sort_values(by=["time"])

    # Save the patient record to a CSV file
    save_path = os.path.join(csts.OUTPUT_DIR_PATH, f"patient_{patient_ID}.csv")
    patient_df.to_csv(save_path, index=False)

    
if __name__ == "__main__":
    main()

























# import os
# import pandas as pd
# from tqdm import tqdm
# from data_processing.data_utils import *
# from data_processing.consent_data import *
# from data_processing.patient_bl_data import *
# from data_processing.kidney_bl_data import *
# from data_processing.patient_stop_data import *
# from data_processing.patient_infectious_disease_data import *
# from data_processing.data_output import *

# import constants
# csts = constants.ConstantsNamespace()


# if __name__ == "__main__":
#     # Create output directory if it does not exist
#     output_dir_path = os.path.dirname(csts.OUTPUT_DIR_PATH)
#     if not os.path.exists(output_dir_path):
#         os.makedirs(output_dir_path)
    
#     # Load main data file
#     data_dict = pd.read_pickle(csts.PICKLE_DATA_PATH)

#     # Get separate dataframes from the excel file
#     # Get the dataframe containing all patients consents
#     consent_df = data_dict[csts.CONSENT_SHEET]
    
#     # Get the dataframe contaning all patients baseline
#     patients_bl_df = data_dict[csts.PATIENTS_BL_SHEET]

#     # Get the dataframe containing data related to kidney baseline
#     kidney_bl_df = data_dict[csts.KIDNEY_BL_SHEET]

#     import ipdb; ipdb.set_trace()

#     # Get the dataframe containing data related to kidney follow-up
#     kidney_fup_df = data_dict[csts.KIDNEY_FUP_SHEET]

#     # Get the dataframe containing data related to patient questionnaire
#     patients_psq_df = data_dict[csts.PATIENTS_PSQ_SHEET]

#     # Get the dataframe containing data related to patient infectious diseases 
#     patients_id_df = data_dict[csts.PATIENTS_ID_SHEET]

#     # Get the dataframe containing data related to patient drug
#     patients_drug_df = data_dict[csts.PATIENTS_DRUG_SHEET]

#     # Get the dataframe containing data related to patient stop
#     patients_stop_df = data_dict[csts.PATIENTS_STOP_SHEET]

#     # Get the dataframe containing data related to organ baseline
#     organ_base_df = data_dict[csts.ORGAN_BASE_SHEET]

#     # Get all patients IDs
#     patients_IDs = get_patients_IDs(consent_df)

#     for patient_ID in tqdm(patients_IDs, "Creating patient records"):

#         # PATIENT CONSENT STATUS ------------------------------------------------
#         consent_status = get_patient_consent_status(patient_ID, consent_df)
#         last_consent_status = get_patient_last_consent_status(consent_status)

#         # PATIENT BASELINE DATA --------------------------------------------------
#         # Get patient birthday
#         birthday = get_patient_birthday(patient_ID, patients_bl_df)
#         gender = get_patient_gender(patient_ID, patients_bl_df)
#         total_cholesterol = get_total_cholesterol(patient_ID, patients_bl_df)
#         weight = get_weight(patient_ID, patients_bl_df)
#         height = get_height(patient_ID, patients_bl_df)
#         hba1c = get_hba1c(patient_ID, patients_bl_df)
#         hba1c_date = get_hba1c_date(patient_ID, patients_bl_df)
#         exit_status = get_exit_status(patient_ID, patients_bl_df)

#         # KIDNEY BASELINE DATA ---------------------------------------------------
#         # TODO check that the end date is after the start date
#         hospitalization_start_date = get_hospitalization_start_date(patient_ID, kidney_bl_df)
#         hospitalization_end_date = get_hospitalization_end_date(patient_ID, kidney_bl_df)
#         donor_birth_date = get_donor_birth_date(patient_ID, kidney_bl_df)
#         donor_gender = get_donor_gender(patient_ID, kidney_bl_df)
#         transplantation_date = get_transplantation_date(patient_ID, kidney_bl_df)
        
#         infectious_diseases = get_infectious_diseases(patient_ID,data=patients_bl_df)
#         for disease in infectious_diseases:
#             print(disease.value)  

#         # PATIENT QUESTIONNAIRE DATA ---------------------------------------------

#         # PATIENT INFECTIOUS DISEASES DATA ---------------------------------------
#         # get_infection_data(0, patients_id_df)

#         # PATIENT DRUG DATA ------------------------------------------------------

#         # PATIENT STOP DATA ------------------------------------------------------
#         death_date = get_death_date(patient_ID, patients_stop_df)
#         dropout_date = get_dropout_date(patient_ID, patients_stop_df)
#         last_alive_date = get_latest_date_known_to_be_alive(patient_ID, patients_stop_df)

#         # ORGAN BASELINE DATA ----------------------------------------------------


