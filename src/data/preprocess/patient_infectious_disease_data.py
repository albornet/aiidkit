##################################################################
# Code from https://github.com/melhk/AIIDKIT-Multi-states-models #
##################################################################

import pandas as pd
from src.data.data_utils import generate_wide_eavt_df_from_object
from src.data.preprocess.clinical_objects import (
    Bacteria, Virus, Fungus, Parasite, Infection, InfectionType, InfectionSite,
    ImmunosuppressionReduced, DonorRelatedInfection, RequiredHospitalization,
    BacterialInfection, ViralInfection, FungalInfection, ParasiticInfection,
)
from src.constants import ConstantsNamespace
csts = ConstantsNamespace()


def create_infections_dict_from_df(
    patient_IDs: list[int],
    patid_df: pd.DataFrame,
) -> dict[int, list[Infection]]:
    """
    Create a dictionary of infections for the given patient IDs, keyed by patient ID.
    There can be multiple infections for the same patient ID on the same date!
    """
    # Create a dictionary to hold the infections for each patient ID
    infections_dict = {patient_ID: [] for patient_ID in patient_IDs}

    # Select only the dataframe rows with the patient IDs in the list of patient IDs
    patid_df = patid_df[patid_df["patid"].isin(patient_IDs)]
    for row in patid_df.iterrows():
        pat_id_data = row[1]
        clinical_infection_type = get_clinical_infection_type(pat_id_data)

        # Create a new infection only if the clinical infection type is not NO_INFECTION
        if clinical_infection_type != InfectionType.NO_INFECTION:
            infection = get_infection(pat_id_data, clinical_infection_type)
            infections_dict[infection.patient_ID].append(infection)
    
    return infections_dict


def get_immunosuppression_reduced(infection_data:pd.DataFrame) -> ImmunosuppressionReduced:
    """
    Check if the immunosuppression was reduced.
    """
    immunosuppression_reduced = infection_data["isred"]
    if immunosuppression_reduced == "Yes":
        return ImmunosuppressionReduced.YES
    elif immunosuppression_reduced == "No":
        return ImmunosuppressionReduced.NO
    return ImmunosuppressionReduced.UNKNOWN
    

def get_donor_related_infections(infection_data:pd.DataFrame) -> DonorRelatedInfection:
    """
    Check if the infection is a potential donor-related infection.
    """
    donor_related_infection = infection_data["donorrelid"]
    if donor_related_infection == "Yes":
        return DonorRelatedInfection.YES
    elif donor_related_infection == "No":
        return DonorRelatedInfection.NO
    return DonorRelatedInfection.UNKNOWN


def get_infection_sites(infection_data:pd.DataFrame) -> list[InfectionSite]:
    infection_sites = get_data_serie_from_serie("infsite", infection_data)
    infenction_sites_enum = []
    for infection_site in infection_sites:
        if infection_site == "Eye":
            infenction_sites_enum.append(InfectionSite.EYE)
        elif infection_site == "Liver":
            infenction_sites_enum.append(InfectionSite.LIVER)
        elif infection_site == "Bone_Joint":
            infenction_sites_enum.append(InfectionSite.BONE_AND_JOINT)
        elif infection_site == "Urinary tract":
            infenction_sites_enum.append(InfectionSite.URINARY_TRACT)
        elif infection_site == "Blood":
            infenction_sites_enum.append(InfectionSite.BLOODSTREAM)
        elif infection_site == "Mucocutaneous":
            infenction_sites_enum.append(InfectionSite.MUSCULOCUTANEOUS)
        elif infection_site == "RT":
            infenction_sites_enum.append(InfectionSite.RESPIRATORY_TRACT)
        elif infection_site == "Prosthetic":
            infenction_sites_enum.append(InfectionSite.PROSTHETIC_DEVICE)
        elif infection_site == "Catheter":
            infenction_sites_enum.append(InfectionSite.CATHETER)
        elif infection_site == "Unidentified":
            infenction_sites_enum.append(InfectionSite.UNIDENTIFIED)
        elif infection_site == "SSI":
            infenction_sites_enum.append(InfectionSite.SURGICAL_SITE_INFECTION)
        elif infection_site == "GI":
            infenction_sites_enum.append(InfectionSite.GASTROINTESTINAL_TRACT)
        elif infection_site == "Heart":
            infenction_sites_enum.append(InfectionSite.HEART)
        elif infection_site == "CNS":
            infenction_sites_enum.append(InfectionSite.CENTRAL_NERVOUS_SYSTEM)
        else:
            infenction_sites_enum.append(InfectionSite.OTHER)
    return infenction_sites_enum


def get_clinical_infection_type(infection_data:pd.DataFrame) -> InfectionType:
    clinical_infection_type = infection_data["inftype"]
    # If this field is NaN or "Not applicable", consider it as a non-infection
    if pd.isna(clinical_infection_type) or clinical_infection_type == "Not applicable":
        return InfectionType.NO_INFECTION
    elif clinical_infection_type == "Asymptomatic":
        return InfectionType.ASYMPTOMATIC
    elif clinical_infection_type == "Proven disease":
        return InfectionType.PROVEN_DISEASE
    elif clinical_infection_type == "Probable disease":
        return InfectionType.PROBABLE_DISEASE
    elif clinical_infection_type == "Possible disease":
        return InfectionType.POSSIBLE_DISEASE
    elif clinical_infection_type == "Colonization":
        return InfectionType.COLONIZATION
    elif clinical_infection_type == "Viral syndrome":
        return InfectionType.VIRAL_SYNDROME
    elif clinical_infection_type == "Primary infection":
        return InfectionType.PRIMARY_INFECTION
    elif clinical_infection_type == "Fever neutropenia":
        return InfectionType.FEVER_NEUTROPENIA
    elif clinical_infection_type == "Unknown":
        return InfectionType.UNKNOWN
    else:
        raise ValueError(f"Unknown clinical infection type: {clinical_infection_type}")


def get_infection_date(infection_data:pd.DataFrame) -> pd.Timestamp:
    """
    Use in preference the date of sampling of a positive clinical specimen
    (for instance date of a positive blood culture), in the absence of a positive
    clinical specimen use the date the clinician made the diagnosis. 
    If no exact date is available, use middle of the month (15th) and/or middle of year.
    """
    if pd.isna(infection_data["infdate"]):
        return None
    else:
        return pd.to_datetime(infection_data["infdate"], errors="coerce")


def get_infection(
    infection_data: pd.DataFrame,
    clinical_infection_type: InfectionType,
) -> Infection:
    patient_ID = infection_data["patid"]
    infection_date = get_infection_date(infection_data)
    required_hospitalization = get_required_hospitalization(infection_data)
    infection_sites = get_infection_sites(infection_data)
    donor_related_infection = get_donor_related_infections(infection_data)
    immunosuppression_reduced = get_immunosuppression_reduced(infection_data) 

    pathogen = infection_data["pathogen"]

    if pathogen == "Not applicable":
        raise ValueError("Pathogen cannot be 'Not applicable'.")
    
    elif pathogen == "Bacteria":
        return create_bacterial_infection(
            infection_data, clinical_infection_type, patient_ID, infection_date,
            required_hospitalization, infection_sites, donor_related_infection,
            immunosuppression_reduced,
        )
    
    elif pathogen == "Virus":
        return create_viral_infection(
            infection_data, clinical_infection_type, patient_ID, infection_date,
            required_hospitalization, infection_sites, donor_related_infection,
            immunosuppression_reduced,
        )
    
    elif pathogen == "Fungi":
        return create_fungal_infection(
            infection_data, clinical_infection_type, patient_ID, infection_date,
            required_hospitalization, infection_sites, donor_related_infection,
            immunosuppression_reduced,
        )
    elif pathogen == "Parasites":
        return create_parasitic_infection(
            infection_data, clinical_infection_type, patient_ID, infection_date,
            required_hospitalization, infection_sites, donor_related_infection,
            immunosuppression_reduced,
        )
    elif pathogen == "Unidentified":
        return Infection(
            patient_ID=patient_ID,
            infection_date=infection_date,
            clinical_infection_type=clinical_infection_type,
            required_hospitalization=required_hospitalization,
            infection_sites=infection_sites,
            donor_related_infection=donor_related_infection,
            immunosuppression_reduced=immunosuppression_reduced,
        )
    else:
        raise ValueError(f"Unknown pathogen type: {pathogen}")

    
def get_required_hospitalization(
    infection_data: pd.DataFrame,
) -> RequiredHospitalization:
    """
    Check if the infection required hospitalization.
    """
    required_hospitalization = infection_data["hospreqid"]
    if required_hospitalization == "Yes":
        return RequiredHospitalization.YES
    elif required_hospitalization == "No":
        return RequiredHospitalization.NO
    return RequiredHospitalization.UNKNOWN


def create_bacterial_infection(
    infection_data: pd.DataFrame,
    clinical_infection_type: InfectionType,
    patient_ID,
    infection_date,
    required_hospitalization,
    infection_sites,
    donor_related_infection,
    immunosuppression_reduced,
) -> BacterialInfection:
    """
    Create a bacterial infection from the infection data.
    """
    bacteria_names = get_serie_from_data_serie("bact", infection_data)
    ESBL_resistance_serie = get_serie_from_data_serie("rp_esbl", infection_data)
    multidrug_resistance_serie = get_data_serie_from_serie("rp_mdr", infection_data)
    staph_aureus_resistance_phenotype_serie = get_data_serie_from_serie("rp_staph", infection_data)
    enterococcis_resistance_phenotype_serie = get_data_serie_from_serie("rp_entero", infection_data)
    cpe_resistance_phenotype_serie = get_data_serie_from_serie("rp_cpe", infection_data)

    indices = [key.split("_")[-1] for key in bacteria_names.keys()]
    ESBL_resistance = [
        ESBL_resistance_serie.get("rp_esbl_" + index, None)
        for index in indices
    ]
    multidrug_resistance = [
        multidrug_resistance_serie.get("rp_mdr_" + index, None)
        for index in indices
    ]
    staph_aureus_resistance_phenotype = [
        staph_aureus_resistance_phenotype_serie.get("rp_staph_" + index, None)
        for index in indices
    ]
    enterococcis_resistance_phenotype = [
        enterococcis_resistance_phenotype_serie.get("rp_entero_" + index, None)
        for index in indices
    ]
    cpe_resistance_phenotype = [
        cpe_resistance_phenotype_serie.get("rp_cpe_" + index, None)
        for index in indices
    ]   

    bacteria_list = []
    for i, name in enumerate(bacteria_names.tolist()):
        bacteria = Bacteria(
            pathogen_type=name, 
            ESBL_resistance=ESBL_resistance[i],
            multidrug_resistance=multidrug_resistance[i],
            staph_aureus_resistance_phenotype=staph_aureus_resistance_phenotype[i],
            enterococcis_resistance_phenotype=enterococcis_resistance_phenotype[i],
            cpe_resistance_phenotype=cpe_resistance_phenotype[i],
        )
        bacteria_list.append(bacteria)

    antibacterial_treatment= infection_data["antibact"] # Yes / No / not applicable
    antibacterial_treatment = True if antibacterial_treatment == "Yes" else False

    return BacterialInfection(
        patient_ID=patient_ID,
        infection_date=infection_date,
        clinical_infection_type=clinical_infection_type,
        required_hospitalization=required_hospitalization,
        infection_sites=infection_sites,
        donor_related_infection=donor_related_infection,
        immunosuppression_reduced=immunosuppression_reduced,
        pathogens=bacteria_list,  # List of Bacteria objects
        antibacterial_treatment=antibacterial_treatment,
    )


def create_viral_infection(
    infection_data: pd.DataFrame, 
    clinical_infection_type: InfectionType,
    patient_ID,
    infection_date,
    required_hospitalization,
    infection_sites,
    donor_related_infection,
    immunosuppression_reduced
) -> ViralInfection:
    virus_names = get_data_serie_from_serie("virus", infection_data)
    viral_primary_infection_serie = get_data_serie_from_serie("virus_prim", infection_data)

    indices = lambda names : [key.split("_")[-1] for key in names.keys()]
    viral_primary_infection = [
        viral_primary_infection_serie.get("virus_prim_" + index, None)
        for index in indices(virus_names)
    ]

    virus_list = [
        Virus(name, viral_primary_infection=viral_primary_infection[i])
        for i, name in enumerate(virus_names)
    ]

    antiviral_treatment = infection_data["antivir"]  # Yes / No / not applicable
    antiviral_treatment = True if antiviral_treatment == "Yes" else False

    return ViralInfection(
        patient_ID=patient_ID,
        infection_date=infection_date,
        clinical_infection_type=clinical_infection_type,
        required_hospitalization=required_hospitalization,
        infection_sites=infection_sites,
        donor_related_infection=donor_related_infection,
        immunosuppression_reduced=immunosuppression_reduced,
        pathogens=virus_list,  # List of Virus objects
        antiviral_treatment=antiviral_treatment,
    )


def create_fungal_infection(
    infection_data: pd.DataFrame,
    clinical_infection_type: InfectionType,
    patient_ID,
    infection_date,
    required_hospitalization,
    infection_sites,
    donor_related_infection,
    immunosuppression_reduced,
) -> FungalInfection:
    fungi_names = get_data_serie_from_serie("fungi", infection_data)
    antifungal_treatment = infection_data["antifungismp"]  # Yes / No / not applicable
    antifungal_treatment_med_serie = get_data_serie_from_serie("antifungi", infection_data)  # List of antifungal treatments
    
    indices = [key.split("_")[-1] for key in fungi_names.keys()]
    antifungal_treatment_med = [
        antifungal_treatment_med_serie.get("antifungi_" + index, None)
        for index in indices
    ]
    
    fungi_list = [
        Fungus(name, antifungal_treatment=antifungal_treatment_med[i])
        for i, name in enumerate(fungi_names)
    ]

    return FungalInfection(
        patient_ID=patient_ID,
        infection_date=infection_date,
        clinical_infection_type=clinical_infection_type,
        required_hospitalization=required_hospitalization,
        infection_sites=infection_sites,
        donor_related_infection=donor_related_infection,
        immunosuppression_reduced=immunosuppression_reduced,
        pathogens=fungi_list,  # List of Fungus objects
        antifungal_treatment=antifungal_treatment,
    )


def create_parasitic_infection(
    infection_data: pd.DataFrame,
    clinical_infection_type: InfectionType,
    patient_ID,
    infection_date,
    required_hospitalization,
    infection_sites,
    donor_related_infection,
    immunosuppression_reduced,
) -> ParasiticInfection:
    
    parasites_names = get_data_serie_from_serie("parasit", infection_data)
    parasitic_primary_infection_serie = get_data_serie_from_serie("parasite_prim", infection_data)

    indices = [key.split("_")[-1] for key in parasites_names.keys()]
    parasitic_primary_infection = [
        parasitic_primary_infection_serie.get("parasite_prim_" + index, None)
        for index in indices
    ]

    parasites_list = [
        Parasite(name, parasitic_primary_infection=parasitic_primary_infection[i])
        for i, name in enumerate(parasites_names)
    ]

    antiparasitic_treatment = infection_data["antiparasit"]
    antiparasitic_treatment = True if antiparasitic_treatment == "Yes" else False

    return ParasiticInfection(
        patient_ID=patient_ID,
        infection_date=infection_date,
        clinical_infection_type=clinical_infection_type,
        required_hospitalization=required_hospitalization,
        infection_sites=infection_sites,
        donor_related_infection=donor_related_infection,
        immunosuppression_reduced=immunosuppression_reduced,
        pathogens=parasites_list,
        antiparasitic_treatment=antiparasitic_treatment,
    )


def get_data_serie(data_name:str, data:pd.DataFrame) -> list:
    
    # Select all columns with the data_name
    event_name = "^" + data_name + "_" + "[0-9]+$"
    event_data = data.filter(regex=event_name)

    # Remove nul or invalid values
    event_data = event_data.where(~event_data.isin(csts.NAN_LIKE_CATEGORIES), other=pd.NaT)
    event_data = event_data.dropna(axis=1, how="all")
    longitudinal_data = []
    for col in event_data.columns:
        longitudinal_data.append(event_data[col].item())

    return longitudinal_data


def get_serie_from_data_serie(data_name:str, data:pd.DataFrame) -> pd.Series:
    
    # Select all columns with the data_name
    event_name = "^" + data_name + "_" + "[0-9]+$"
    event_data = data.filter(regex=event_name)

    # Remove nul or invalid values
    event_data = event_data.where(~event_data.isin(csts.NAN_LIKE_CATEGORIES), other=pd.NaT)
    event_data = event_data.dropna(how="all")

    return event_data


def get_data_serie_from_serie(data_name:str, data: pd.Series) -> pd.Series:
    
    # Select all columns with the data_name
    event_name = "^" + data_name + "_" + "[0-9]+$"
    event_data = data.filter(regex=event_name)
    
    # Remove nul or invalid values
    event_data = event_data.where(~event_data.isin(csts.NAN_LIKE_CATEGORIES), other=pd.NaT)
    event_data = event_data.dropna(how="all")

    return event_data


def pool_patient_infection_data(
    patient_ID: int,
    patient_infection_df: pd.DataFrame,
) -> pd.DataFrame:
    """ Get information about infections occured to a patient
    """
    # Load detailed infection information from the data
    pat_inf_events = create_infections_dict_from_df(
        patient_IDs=[patient_ID],
        patid_df=patient_infection_df,
    )[patient_ID]

    # Flatten each patient infection event to an EAVT table
    pat_inf_dfs = []
    for inf_event in pat_inf_events:
        eavt_table = generate_wide_eavt_df_from_object(
            root_obj=inf_event,
            time_key="infection_date",
            filtered_attributes=["patient_ID"],
            filtered_values=["UNKNOWN", "Unknown", None]
        )
        pat_inf_dfs.append(eavt_table)
    
    # Create patient dataframe by concatenating all EAVT tables
    if not pat_inf_dfs:
        patient_df = pd.DataFrame()
    else:
        patient_df = pd.concat(pat_inf_dfs, ignore_index=True)

    return patient_df
