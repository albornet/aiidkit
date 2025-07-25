import os
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass(frozen=True)
class ConstantsNamespace():

    # Data-related paths
    EXCEL_DATA_PATH = os.path.join("data", "raw", "Datacut_FUP226_raw-01Jan2023_v1.xlsx")
    PICKLE_DATA_PATH = os.path.join("data", "raw", "Datacut_FUP226_raw-01Jan2023_v1.pkl")
    PREPROCESSED_DIR_PATH = os.path.join("data", "preprocessed")
    HUGGINGFACE_DIR_PATH = os.path.join("data", "huggingface")
    EXPLORE_DIR_PATH = os.path.join("data", "explore")

    # Sheet names
    CONSENT_SHEET = "Consent"  # "#1_CONSENT" <- names change for the full data file
    KIDNEY_BL_SHEET = "Kidney_BL"  # "#2_KIDNEY_BL"
    KIDNEY_FUP_SHEET = "Kidney_FUP"  # "#3_KIDNEY_FUP"
    PATIENT_BL_SHEET = "PAT_BL"  # "#4_PAT_BL"
    PATIENT_PSQ_SHEET = "PAT_PSQ"  # "#5_PAT_PSQ"
    PATIENT_INFECTION_SHEET = "PAT_ID"  # "#6_PAT_ID" <- renamed to avoid confusion with patient identifier (ID)
    PATIENT_DRUG_SHEET = "PAT_Drug"  # "#7_PAT_DRUG"
    PATIENT_STOP_SHEET = "PAT_Stop"  # "#8_PAT_STOP"
    ORGAN_BASE_SHEET = "Organ_Base"  # "#9_ORGAN_BASE"

    # To use for removing missing or invalid values
    VALID_DATE_RANGE = (pd.Timestamp("1900-01-01"), pd.Timestamp("2030-01-01"))
    NAN_LIKE_DATES = (pd.NaT, pd.Timestamp("9999-01-01"), pd.Timestamp("2000-01-01"))
    NAN_LIKE_NUMBERS = (np.nan, pd.NA, -555.0, -666.0, -777.0, -888.0, -999.0)
    NAN_LIKE_CATEGORIES = (
        "NaN", "nan", "Nan", pd.NA, np.nan, "NA in FUP", -999.0,  # "Unknown"
        "Global consent refused", "Refused", "Not done", "Not applicable",
    )

    # Global mapping for patient ethnicity normalization
    ETHNICITY_NORMALIZATION_MAP = {
        "african": "African",
        "american indian": "American Indian",
        "amerindien": "American Indian",
        "amã©rique du sud": "South American",
        "amérique du sud": "South American",
        "asian": "Asian",
        "bolivian": "South American (Bolivian)",
        "brasil": "South American (Brazilian)",
        "brazil": "South American (Brazilian)",
        "brazilian": "South American (Brazilian)",
        "caucasian": "Caucasian",
        "caucasian and africain": "Mixed Race (Caucasian and African)",
        "african/caucasian": "Mixed Race (Caucasian and African)",
        "chile": "South American (Chilean)",
        "dominican republic": "Latin American (Dominican Republic)",
        "india": "Asian (Indian)",
        "jemen": "Middle Eastern (Yemeni)",
        "kurde": "Middle Eastern (Kurdish)",
        "latin american": "Latin American",
        "latino american": "Latin American",
        "mauritian": "African (Mauritian)",
        "mongolian": "Asian (Mongolian)",
        "moroccan": "North African (Moroccan)",
        "north africa": "North African",
        "south american": "South American",
        "south american (american indian)": "South American (American Indian)",
        "sud american (chili)": "South American (Chilean)",
        "syria": "Middle Eastern (Syrian)",
        "syrian, saoudian, caucasian": "Mixed Race (Syrian, Saudi, Caucasian)",
        "tamil": "Asian (Tamil)",
        "tamil": "Asian (Tamil)",
        "thamil": "Asian (Tamil)",
        "tunesier": "North African (Tunisian)",
        "tunisia": "North African (Tunisian)",
        "tunisian": "North African (Tunisian)",
        "turkey": "Middle Eastern (Turkish)",
        "turkish": "Middle Eastern (Turkish)",
        "unknown": "Unknown",
        "arab": "Middle Eastern (Arab)",
        "arabic": "Middle Eastern (Arab)",
        "caucasian and arabic": "Mixed Race (Caucasian and Arab)",
        "caucasian and latino": "Mixed Race (Caucasian and Latin American)",
        "caucasian and latino american": "Mixed Race (Caucasian and Latin American)",
        "hispanic": "Latin American (Hispanic)",
        "lebanese": "Middle Eastern (Lebanese)",
        "marokko and caucasian": "Mixed Race (Moroccan and Caucasian)",
        "mixed race": "Mixed Race",
        "persian": "Middle Eastern (Persian)",
        "peru": "South American (Peruvian)",
        "sri lanka": "Asian (Sri Lankan)",
        "thai": "Asian (Thai)",
        "malagasy": "African (malagasy)",
        "ecuadorian": "South American (Ecuadorian)",
    }

    # Global mapping for "other" drug normalization (only for "class" == "OtherDrugs")
    DRUG_NORMALIZATION_MAP = {
        # Not Applicable / Unknown
        "Not applicable": "Not Applicable",
        "Unknown": "Unknown",
        "other": "Other",
        "Hormono-radiotherapy": "Other", # Combination therapy
        "Antirheumatikum": "Other", # Vague class
        "Galaktose": "Other", # Diagnostic/sweetener
        "Geon": "Other", # Unidentifiable
        "Tyuesyurt": "Other", # Unidentifiable
        "Ulovaxon": "Other", # Unidentifiable
        "Acide Fatigue": "Other", # Likely typo for Folic Acid, but ambiguous
        "Carmethin": "Other", # Unidentifiable
        "Prolin": "Other", # Amino Acid, not typically a drug in this context
        "Zytorix": "Other", # Unidentifiable
        "Acidium": "Other", # Ambiguous

        # Medical/Therapeutic Procedure
        "blood transfusion": "Medical/Therapeutic Procedure",
        "1 blood transfusion": "Medical/Therapeutic Procedure",
        "2 bloods transfusion for anemia": "Medical/Therapeutic Procedure",
        "2 bloods transfusions": "Medical/Therapeutic Procedure",
        "2 bloods transfusions due HB lower 82": "Medical/Therapeutic Procedure",
        "3 bloods transfusions": "Medical/Therapeutic Procedure",
        "1 blood transfusion due lower 89": "Medical/Therapeutic Procedure",
        "blood tranfusion": "Medical/Therapeutic Procedure",
        "Plasmapheresis": "Medical/Therapeutic Procedure",
        "plasmapheresis": "Medical/Therapeutic Procedure",
        "plasmapherisis": "Medical/Therapeutic Procedure",
        "plasmapherese  for initial disease": "Medical/Therapeutic Procedure",
        "PEX": "Medical/Therapeutic Procedure",
        "PEX plasmapheresis": "Medical/Therapeutic Procedure",
        "plasma exchange (PEX)": "Medical/Therapeutic Procedure",
        "Plasmapheresis 5 times": "Medical/Therapeutic Procedure",
        "plasmapherese for initial disease": "Medical/Therapeutic Procedure",
        "Immunadsorption": "Medical/Therapeutic Procedure",
        "Immunoadsorption": "Medical/Therapeutic Procedure",
        "Immunoabsorption": "Medical/Therapeutic Procedure",
        "ABO-IgG Immunabsorbtion": "Medical/Therapeutic Procedure",
        "ABO-IgG Adsorption": "Medical/Therapeutic Procedure",
        "AB0-Immunabsorbtion": "Medical/Therapeutic Procedure",
        "AB0-immunabsorbtion": "Medical/Therapeutic Procedure",
        "AB0-Immunadsorption": "Medical/Therapeutic Procedure",
        "AB0-Immunadsoption": "Medical/Therapeutic Procedure",
        "ABO-IgG-Absorbtion": "Medical/Therapeutic Procedure",
        "ABOi IgG Adsorption": "Medical/Therapeutic Procedure",
        "AB0-IgG Absorption": "Medical/Therapeutic Procedure",
        "IgG Adsorption": "Medical/Therapeutic Procedure",
        "Photopheresis": "Medical/Therapeutic Procedure",
        "photopheresis": "Medical/Therapeutic Procedure",
        "oxygen administration": "Medical/Therapeutic Procedure",
        "oxygene": "Medical/Therapeutic Procedure",
        "TLI": "Medical/Therapeutic Procedure", # Thoracic Lymphoid Irradiation
        "TLI in accordance with Swisstolerance StudyÂ” protocol": "Medical/Therapeutic Procedure",
        "Radiation therapy due to swisstolerance.CH study": "Medical/Therapeutic Procedure",
        "radiotherapy ttt for bone metastasis": "Medical/Therapeutic Procedure",
        "Radiotherapy": "Medical/Therapeutic Procedure",
        "Deflux injection": "Medical/Therapeutic Procedure",

        # --- Cardiovascular Agents ---
        "Antihypertensiva": "Antihypertensive (unspecified)",
        "Antihypertensive drugs": "Antihypertensive (unspecified)",
        "Hypertension Drug": "Antihypertensive (unspecified)",
        "Antihypertonikum": "Antihypertensive (unspecified)",
        "anti-hypertenseur (catapressan)": "Centrally Acting Agent", # More specific mapping
        "anti-hypertenseur (loniten )": "Vasodilator", # More specific mapping

        # ACE-Inhibitor
        "Lisinoprilum": "ACE-Inhibitor",
        "Captoprilum": "ACE-Inhibitor",
        "Ramiprilum": "ACE-Inhibitor",

        # Alpha-blocker
        "Alpha blocker": "Alpha-blocker",
        "Alpha-Blocker": "Alpha-blocker",
        "alpha-blocker": "Alpha-blocker",
        "alpha blocker": "Alpha-blocker",
        "alpha- Blocker": "Alpha-blocker",
        "Alpha-Antagonist": "Alpha-blocker",
        "Alphablocker": "Alpha-blocker",
        "alphablocker": "Alpha-blocker",
        "alpha- blocker:": "Alpha-blocker",
        "alpha-rezeptor blocker": "Alpha-blocker",
        "Alpha-1-Adrenorezeptoren-Blocker": "Alpha-blocker",
        "Doxazosin": "Alpha-blocker",
        "doxazosin": "Alpha-blocker",
        "Doxasosinum/Cardura": "Alpha-blocker",
        "Doxazosinum": "Alpha-blocker",
        "Doxazosinum / Cardura": "Alpha-blocker",
        "Doxazosin/Cardura": "Alpha-blocker",
        "Doxacosin": "Alpha-blocker", # Typo
        "Doxazesin": "Alpha-blocker", # Typo
        "doxacosinmesylat": "Alpha-blocker",
        "Docazosin": "Alpha-blocker", # Typo
        "Cardura": "Alpha-blocker",
        "cardura": "Alpha-blocker",
        "Cardura (Doxazosine)": "Alpha-blocker",
        "Cardura (Doxazosin)": "Alpha-blocker",
        "Doxazosin (Cardura)": "Alpha-blocker",
        "Cardura (Alpha-Blocker)": "Alpha-blocker",
        "Cardura (alpha-blocker)": "Alpha-blocker",
        "cardura (alpha-blocker)": "Alpha-blocker",
        "alpha blocker(cardura)": "Alpha-blocker",
        "Cardura, Alphablocker": "Alpha-blocker",
        "Tamsulosin": "Alpha-blocker",
        "tamsulosine (pradif)": "Alpha-blocker",
        "Tamsulosini (HCI) / Pradif": "Alpha-blocker",
        "Tamsulosini / Pradif": "Alpha-blocker",
        "Tamsulosini/Pradiv": "Alpha-blocker",
        "Tamsulosini hydrochlorid/Pradif": "Alpha-blocker",
        "Tamsulosini Hydrochloridum.": "Alpha-blocker",
        "Tamsulosion / Pradif": "Alpha-blocker",
        "Pradif": "Alpha-blocker",
        "pradif": "Alpha-blocker",
        "Pradif T (Tamsulosine)": "Alpha-blocker",
        "Pradif (Tamsulosine)": "Alpha-blocker",
        "Pradif (Alphablocker)": "Alpha-blocker",
        "Xatral": "Alpha-blocker",
        "Xatral Uno (Alfuzosine)": "Alpha-blocker",
        "Hytrin / alpha receptor blocker": "Alpha-blocker",
        "DUODART": "Alpha-blocker", # Dutasteride/Tamsulosin combo
        "Tamsulosini hydrochloridum": "Alpha-blocker",

        # ARB (Angiotensin II Receptor Blocker)
        "Candesartanum cilexetilum": "ARB",
        "Candesartan": "ARB",
        "Olmesartanmedoxomil": "ARB",
        "Losartan-Kalium": "ARB",
        
        # ARB/Diuretic Combo
        "ARB + diuretic (erdabyclore) et diuretic (torem)": "ARB/Diuretic Combo",
        "Candesartanum cilexetilum, Hydrochlorothiazidum / Atacand plus": "ARB/Diuretic Combo",
        "diuretic (micardis plus)": "ARB/Diuretic Combo",
        "micardis plus (plus diuretic)": "ARB/Diuretic Combo",
        "diuretic (co-aprovel)": "ARB/Diuretic Combo",
        "Votum plus (ARB- Duiretic)": "ARB/Diuretic Combo",
        "Sevikar HCT": "ARB/Diuretic Combo",
        "diuretic (voltum plus)": "ARB/Diuretic Combo",

        # Angiotensin Receptor-Neprilysin Inhibitor (ARNI)
        "Entresto": "Angiotensin Receptor-Neprilysin Inhibitor (ARNI)",
        "Sacubitril (Entresto)": "Angiotensin Receptor-Neprilysin Inhibitor (ARNI)",
        "Salubitril": "Angiotensin Receptor-Neprilysin Inhibitor (ARNI)", # Typo

        # Antiarrhythmic
        "Cordarone": "Antiarrhythmic",
        "cordarone (FA cardiversion )": "Antiarrhythmic",
        "Cordarone (amiodarone)": "Antiarrhythmic",
        "Cordarone (Amiodaron)": "Antiarrhythmic",
        "Amiodarone (Cordarone)": "Antiarrhythmic",
        "Amiodaron": "Antiarrhythmic",
        "amiodaron": "Antiarrhythmic",
        "Amiodaroni Hydrochloridum": "Antiarrhythmic",
        "anti-arrythmic": "Antiarrhythmic",
        "antiarythmie(digoxine)": "Antiarrhythmic",
        "Herzglykosid": "Antiarrhythmic",
        "cardiac glycosides": "Antiarrhythmic",
        "Tambocor": "Antiarrhythmic",
        "Tambocor (coeur)": "Antiarrhythmic",
        "Tambocar (antiarythmique)": "Antiarrhythmic",
        "Flecainid acetat (Tambocor)": "Antiarrhythmic",
        "Flecainid": "Antiarrhythmic",
        "Rhythmonorm": "Antiarrhythmic",
        "multaq": "Antiarrhythmic",
        "Multaq (antiarythmique)": "Antiarrhythmic",
        "Sotalol pour cardioversion": "Antiarrhythmic",

        # Beta-blocker
        "Beta blocker": "Beta-blocker",
        "Beta-Blocker i.v.": "Beta-blocker",
        "Propranololi hydrochl/Inderal": "Beta-blocker",

        # CCB (Calcium Channel Blocker)
        "Amlodipinum": "CCB",
        "Amlodipin": "CCB",
        "Lercanidipini hydrochloridum": "CCB",
        "Lercanidipini": "CCB",
        "calcium antagonist": "CCB",

        # Centrally Acting Agent
        "Central blocker": "Centrally Acting Agent",
        "central blocker": "Centrally Acting Agent",
        "Central-blocker": "Centrally Acting Agent",
        "Central blocker (catapressan )": "Centrally Acting Agent",
        "central blocker(catapressan)": "Centrally Acting Agent",
        "Central Blocker (Catapressan)": "Centrally Acting Agent",
        "Alpha-bloquer (Catapresan)": "Centrally Acting Agent", # Catapresan is central
        "central blocker (physiotens )": "Centrally Acting Agent",
        "Central blocker (physiotens)": "Centrally Acting Agent",
        "central Blocker (phisiotens)": "Centrally Acting Agent",
        "Physiotens (Central Adrenergic)": "Centrally Acting Agent",
        "Central Adrenergic Physiotens": "Centrally Acting Agent",
        "Alpha-Rezeptor-Agonist(Catapressan)": "Centrally Acting Agent",
        "Ã¡2-Rezeptor-Agonist": "Centrally Acting Agent",
        "Antisympathotonikum": "Centrally Acting Agent",
        "Moxonidinum / Physiotens": "Centrally Acting Agent",
        "Moxonidin / Physiotens": "Centrally Acting Agent",
        "Moxonidum/Physiotens": "Centrally Acting Agent",
        "Moxonidin / Physiotrans": "Centrally Acting Agent",
        "Moxonidinum": "Centrally Acting Agent",
        "Moxonidium": "Centrally Acting Agent",
        "Moxomidin": "Centrally Acting Agent", # Typo
        "Monoxidin": "Centrally Acting Agent", # Typo
        "Catapresan": "Centrally Acting Agent",
        "Methyldopa/Aldomet": "Centrally Acting Agent",

        # Diuretic
        "Diuretic": "Diuretic",
        "diuretic": "Diuretic",
        "diuretics": "Diuretic",
        "diuretiic": "Diuretic",
        "Diueretic": "Diuretic",
        "Diurectics": "Diuretic",
        "diurectics": "Diuretic",
        "diuretic 2 weeks": "Diuretic",
        "Diretikum": "Diuretic",
        "Diureticum": "Diuretic",
        "Diaretikum": "Diuretic",
        "DiurÃ©tique": "Diuretic",
        "Diuretic ( Torem )": "Diuretic",
        "diuretic (torem)": "Diuretic",
        "diuretic (Torem)": "Diuretic",
        "diuretic (lasix )": "Diuretic",
        "Lasix (diuretic )": "Diuretic",
        "diuretic ( Aldactone )": "Diuretic",
        "diuretics (aldactone )": "Diuretic",
        "diuretic (aldactone)": "Diuretic",
        "diuretic (inspra)": "Diuretic",
        "diuretic (comilorid )": "Diuretic",
        "diuretic (coversum combi)": "Diuretic", # Combo, but diuretic is mentioned
        "Valsartan (diuretic)": "Diuretic", # Valsartan is ARB, but specified as diuretic combo
        "diuretic (metolazone)": "Diuretic",
        "metolazone (diuretic)": "Diuretic",
        "Diuretic- Inspra, metolazole": "Diuretic",
        "Diuretic (Thiazid)": "Diuretic",
        "Torasemid": "Diuretic",
        "Torasemide.": "Diuretic",
        "Torasemidum": "Diuretic",
        "Torasemidium": "Diuretic",
        "Toresemidum": "Diuretic",
        "Torsemid / Torem": "Diuretic",
        "Torasemid / Torem": "Diuretic",
        "Toresemiid / Torem": "Diuretic",
        "Torasamied / Torem": "Diuretic",
        "Torsemid / Torsis": "Diuretic",
        "Torem": "Diuretic",
        "torem": "Diuretic",
        "Torfin": "Diuretic", # Brand for Torasemide
        "Spironolacton / Aldactone": "Diuretic",
        "Spironolacton / Xenalon": "Diuretic",
        "Sprironolactonum": "Diuretic", # Typo
        "Prolactone": "Diuretic", # Likely Spironolactone
        "Aldosteron Antagonist": "Diuretic",
        "Furosemidum.": "Diuretic",
        "Forosemid / Lasix": "Diuretic",
        "Furosemid / Lasic": "Diuretic",
        "Metalozon": "Diuretic",
        "Metolaznum": "Diuretic",
        "Thiazidum": "Diuretic",
        "Thiaziddiuretika": "Diuretic",
        "Thiaziddiuretikum": "Diuretic",
        "Chlorthalidone": "Diuretic",
        "clorthalidone": "Diuretic",
        "Chlorthiclidone": "Diuretic", # Typo
        "Chlortachidon": "Diuretic", # Typo
        "chlortalidon": "Diuretic", # Typo
        "Hydrchlorothiazid": "Diuretic",
        "Indapamid": "Diuretic",
        "indapamid": "Diuretic",
        "Judapamid": "Diuretic", # Typo
        "Indapenid": "Diuretic", # Typo
        "Indapamid retard": "Diuretic",
        "Esidrix": "Diuretic",
        "Inspra": "Diuretic",

        # Heart Rate Reducer (If Channel Inhibitor)
        "Ivabradine": "Heart Rate Reducer (If Channel Inhibitor)",
        "Procoralan (Ivabradin)": "Heart Rate Reducer (If Channel Inhibitor)",
        
        # Hemorheologic Agent
        "Pentoxifylin": "Hemorheologic Agent",
        "Pentoxifyllin": "Hemorheologic Agent",

        # Renin-Inhibitor
        "Renin-Inhibitor": "Renin-Inhibitor",
        "Renininhibitor": "Renin-Inhibitor",
        "Reninhemmer": "Renin-Inhibitor",
        "Reninrezeptor-Blocker(Rasilez)": "Renin-Inhibitor",
        "RenÃ¯n antagonist": "Renin-Inhibitor",
        "Aliskiren": "Renin-Inhibitor",
        "Aliskiren (Renin Hemmer)": "Renin-Inhibitor",
        "Rasilez": "Renin-Inhibitor",
        "Rasilez (Aliskirenum, inhibiteur de la rÃ©nine)": "Renin-Inhibitor",
        "Rasilez (AliskirÃ¨ne)": "Renin-Inhibitor",
        "aliskiren (Antihypertonikum)": "Renin-Inhibitor",
        "Aliskirenum": "Renin-Inhibitor",

        # Vasodilator
        "Minoxidil (Loniten)": "Vasodilator",
        "Minoxidilum / Loniten": "Vasodilator",
        "minoxidil": "Vasodilator",
        "Loniten (Minoxidil)": "Vasodilator",
        "Loniten (Vasodilatateur)": "Vasodilator",
        "Vasodilatateur (loniten )": "Vasodilator",
        "vaso-dilatateur ( loniten )": "Vasodilator",
        "loniten": "Vasodilator",
        "Vasodilatator": "Vasodilator",
        "Glyceroltrinitrat": "Vasodilator",
        "Deponite patch": "Vasodilator",
        "Molsidomine (corvaton retard)": "Vasodilator",
        "molsidomine": "Vasodilator",
        "Nicorandil": "Vasodilator",
        "Nicorandilum (Dancor)": "Vasodilator",
        "vasodilatateur coronarien": "Vasodilator",
        "Koronarvasodilatans": "Vasodilator",
        "Nipruss": "Vasodilator", # Sodium nitroprusside
        
        # Vasopressor / Sympathomimetic
        "Sympathomimetika": "Sympathomimetic",
        "Dobutaminum": "Sympathomimetic",
        "Dobutaminum / Dobutrex": "Sympathomimetic",
        "Dobutamin /Dobutrex": "Sympathomimetic",
        "Adrenalin": "Sympathomimetic",
        "Ephedrin": "Sympathomimetic",
        "Midodrin / Gutron": "Sympathomimetic",
        "Simdax": "Sympathomimetic", # Levosimendan
        "Levosimendan": "Sympathomimetic",
        
        # --- Neurology & Psychiatry Agents ---
        # Analgesic (Non-Opioid)
        "Paracetamolum/Dafalgan": "Analgesic (Non-Opioid)",
        "Paracetamol / Dafalgan": "Analgesic (Non-Opioid)",
        "Paracetamolium/Dafalgan": "Analgesic (Non-Opioid)",
        "Pyrazolunderivat": "Analgesic (Non-Opioid)",
        "Pyrazolonderivat/ Novalgin": "Analgesic (Non-Opioid)",
        "Metamizol / Novalgin": "Analgesic (Non-Opioid)",

        # Analgesic (Opioid)
        "Fentanyl": "Opioid Analgesic",
        "Analgeticum/Fentanyl": "Opioid Analgesic",
        "Opium/Ventanyl": "Opioid Analgesic", # Typo for Fentanyl
        "Tramadolhydrochlorid": "Opioid Analgesic",
        "Tramadolhydrochlorid / Tramal": "Opioid Analgesic",
        "Tramacol": "Opioid Analgesic", # Typo
        "methadone": "Opioid Analgesic",
        "Morphin (Sevre-Long)": "Opioid Analgesic",
        "Hydromorphon": "Opioid Analgesic",
        "Palladon": "Opioid Analgesic", # Hydromorphone
        "Oxycodon hydrochlorid": "Opioid Analgesic",
        "Targiu": "Opioid Analgesic", # Oxycodone/Naloxone
        "Buprenorphin / Temgesic": "Opioid Analgesic",

        # Antimigraine Agent
        "Zolmitriptan": "Antimigraine Agent",
        "Zolmitripan": "Antimigraine Agent", # Typo

        # Anxiolytic/Sedative
        "Anxiolit": "Anxiolytic/Sedative",
        "Diazepam (Valium)": "Anxiolytic/Sedative",
        "Diazepam/Valium": "Anxiolytic/Sedative",
        "Oxazepamum.": "Anxiolytic/Sedative",
        "Oxazepamum": "Anxiolytic/Sedative",
        "Flunitrazepamum": "Anxiolytic/Sedative",
        "Bromazepamum": "Anxiolytic/Sedative",
        "Bromazepanum": "Anxiolytic/Sedative",
        "Bromazepamum / Lexotanil": "Anxiolytic/Sedative",
        "Lorazepamum / Temesta": "Anxiolytic/Sedative",
        "Lorazepnaum/Temesta": "Anxiolytic/Sedative", # Typo
        "Xanax": "Anxiolytic/Sedative",
        "Xamax": "Anxiolytic/Sedative", # Typo
        "Tranxilium": "Anxiolytic/Sedative",
        "Imovane": "Anxiolytic/Sedative", # Zopiclone
        "Chloraldurat": "Anxiolytic/Sedative", # Chloral hydrate
        "Auxolit 15 mg": "Anxiolytic/Sedative", # Typo for Anxiolit? Oxazepam?
        
        # Anticonvulsant
        "Antiepileptic drug": "Anticonvulsant",
        "anti-epileptic": "Anticonvulsant",
        "Antiepileptic": "Anticonvulsant",
        "anti-Ã©pileptique ( keppra )": "Anticonvulsant",
        "Antiepileptikum": "Anticonvulsant",
        "anti-epileptique": "Anticonvulsant",
        "Levetiracetam (Keppra)": "Anticonvulsant",
        "Levetiracetam (anticonvulsants)": "Anticonvulsant",
        "keppra": "Anticonvulsant",
        "LÃ©vÃ©tiracetam": "Anticonvulsant",
        "Gabapentin / Neurontin": "Anticonvulsant",
        "Gabapentin": "Anticonvulsant",
        "Gabapentine": "Anticonvulsant",
        "neurontin": "Anticonvulsant",
        "Gaba pe-ti": "Anticonvulsant", # Gabapentin
        "Neurotin": "Anticonvulsant", # Typo for Neurontin
        "Lyrisa": "Anticonvulsant", # Typo for Lyrica
        "Valproat": "Anticonvulsant",
        "Natrii valproas": "Anticonvulsant",
        "Depakine chrono": "Anticonvulsant",
        "Valproat / Depakine chrono": "Anticonvulsant",
        "Orfiril": "Anticonvulsant", # Valproate
        "Lamotriginum": "Anticonvulsant",
        "Lamotrizin": "Anticonvulsant",
        "Lamictal (Lamotrigine)": "Anticonvulsant",
        "Oxcarbazebinum/Trileptal": "Anticonvulsant",
        "Phenhydan": "Anticonvulsant", # Phenytoin
        "Topiramat / Topamax": "Anticonvulsant",
        "Topiramat": "Anticonvulsant",
        "Topamax": "Anticonvulsant",
        "Rufinamid": "Anticonvulsant",

        # Antidepressant
        "Antidepressant drug": "Antidepressant",
        "Antidepressant": "Antidepressant",
        "anti-depressive treatment": "Antidepressant",
        "Tricyclic antidepressants": "Antidepressant",
        "tricyclic antridepressant": "Antidepressant", # Typo
        "Antidepressants SSRI": "Antidepressant",
        "serotonin reuptake inhibitors": "Antidepressant",
        "psycho-analeptique": "Antidepressant",
        "Mirtazapin": "Antidepressant",
        "MMirtazepin": "Antidepressant", # Typo
        "Mirazepin": "Antidepressant", # Typo
        "Mirtazepin / Remeron": "Antidepressant",
        "Remeron/ Antidepressivum": "Antidepressant",
        "Mitrazapinum": "Antidepressant",
        "Sertralinum": "Antidepressant",
        "Seralin": "Antidepressant", # Typo
        "Serfralin": "Antidepressant", # Typo
        "Cipralex (Escitalopram)": "Antidepressant",
        "Escitalopramum": "Antidepressant",
        "Ecitalopram": "Antidepressant", # Typo
        "Imipramini hydrochloridum": "Antidepressant",
        "Trazodon/Trittico": "Antidepressant",
        "Surmontil (Depression)": "Antidepressant",
        "Amitriptylin": "Antidepressant",
        "Doxepinum": "Antidepressant",
        "Bupropioni hydrochloridum": "Antidepressant",
        "Bupropion": "Antidepressant",
        "Venlafaxinum": "Antidepressant",
        "Venlaflaxin": "Antidepressant", # Typo
        "Venlafaxin": "Antidepressant", # Typo
        "Effexor": "Antidepressant", # Venlafaxine
        "Citalopranum / Citaoprami hydrobromidum": "Antidepressant",
        "Seropram": "Antidepressant", # Citalopram
        "Fluoxetinum": "Antidepressant",
        "Fluctine": "Antidepressant", # Fluoxetine
        "CImbalta": "Antidepressant", # Typo
        "Trimipran": "Antidepressant",
        "Trimipramini mesilas": "Antidepressant",
        "valdoxan": "Antidepressant", # Agomelatine
        "Vortioxetin": "Antidepressant",
        "Brintellix": "Antidepressant", # Vortioxetine
        "Paroxitin": "Antidepressant", # Paroxetine
        
        # Antiparkinson Agent
        "Levodopa Benserazid (Madopar)": "Antiparkinson Agent",
        "Levodopa + Benserazid (Madopar)": "Antiparkinson Agent",
        "dopar (Levodopa+Benserazid)": "Antiparkinson Agent",
        "Madorpar": "Antiparkinson Agent", # Typo
        "Levodopa + Carbidopa + Entacapon (Stalevo)": "Antiparkinson Agent",
        "Lerodopa": "Antiparkinson Agent", # Typo
        "Ropinirol": "Antiparkinson Agent",
        "Pramipexolum": "Antiparkinson Agent",
        "Pramipexal": "Antiparkinson Agent", # Typo
        "Pramiprexol": "Antiparkinson Agent", # Typo
        "Adartrel": "Antiparkinson Agent", # Ropinirole
        "Sifroc": "Antiparkinson Agent", # Pramipexole
        "Bromocriptine": "Antiparkinson Agent",
        
        # Antipsychotic
        "Antipsychotic drug": "Antipsychotic",
        "Neuroleptics": "Antipsychotic",
        "Neuroleptikum": "Antipsychotic",
        "Pipamperon": "Antipsychotic",
        "Pipamperonum": "Antipsychotic",
        "Pimpamperonum": "Antipsychotic", # Typo
        "Pipamperonum dihydrol": "Antipsychotic",
        "Pipamperonum dihydrochloridum": "Antipsychotic",
        "Pipamperonum (HCI)": "Antipsychotic",
        "Pipamperonum / Dipiperon": "Antipsychotic",
        "Quetiapin / Serequel": "Antipsychotic",
        "Levomepromazin": "Antipsychotic",
        "Levomeprovazinum / Nozinan": "Antipsychotic",
        "Haloperidol/^Haldol": "Antipsychotic",
        "Dopamin Antagonist": "Antipsychotic",
        "Risperdal (RispÃ©ridone)": "Antipsychotic",
        "Truxal": "Antipsychotic", # Chlorprothixene
        
        # Other CNS Agents
        "Ritaline (MÃ©thylphÃ©nidate)": "CNS Stimulant",
        "Exelon": "Cholinesterase Inhibitor", # Rivastigmine
        "Mestinon": "Cholinesterase Inhibitor", # Pyridostigmine
        "Piracetam": "Nootropic",
        "Limptar": "Nootropic", # Quinine/Ginkgo
        "Tebokan": "Nootropic", # Ginkgo Biloba
        "Myrtaven": "Nootropic", # Vincamine/Rutoside
        
        # --- Antimicrobials ---
        # Antibiotic
        "Uvamine Retard (Nitrofurantoin) en alternance avec le Bactrim.": "Antibiotic",
        "Uvamine R (NitrofurantoÃ¯ne) en alternance": "Antibiotic",
        "Uvamin R (NitrofurantoÃ¯ne) Alternatively.": "Antibiotic",
        "Uvamine R (NitrofurantoÃ¯ne) en alternance.": "Antibiotic",
        "Monuril (Fosfomycine) en alternance.": "Antibiotic",
        "Monuril (Fosfomycine) en alternance": "Antibiotic",
        "monuril (fosfomycine)": "Antibiotic",
        "Fosfomycin/Monuril": "Antibiotic",
        "Fosfomycin (Monuril) in alternance with Nitrofurantoin (Furadantin)": "Antibiotic",
        "Nitrofurantoin / Uvamin": "Antibiotic",
        "Nitrofurantoin (Uvamin)": "Antibiotic",
        "nitrofurantoine (uvamine)": "Antibiotic",
        "antiseptique (furandantine retard )": "Antibiotic",
        "Furandantine R (NitrofurantoÃ¯ne) en alternance.": "Antibiotic",
        "Furandantine R (NitrofurantoÃ¯ne) en alternance": "Antibiotic",
        "furandantin retard prophylaxie pour cystite": "Antibiotic",
        "Furadantine (Uvamine R)": "Antibiotic",
        "furandantin retad": "Antibiotic",
        "Furandantine retard": "Antibiotic",
        "furandantine": "Antibiotic",
        "furadantine": "Antibiotic",
        "nIFURANTIN": "Antibiotic", # Typo
        "uvamines": "Antibiotic",
        "uvamine retard": "Antibiotic",
        "NitrofurantoÃ¯ne": "Antibiotic",
        "nitrofurantoÃ¯ne": "Antibiotic",
        "nitrofurentoine": "Antibiotic", # Typo
        "Nitrofurantoin (Furadantin retard)": "Antibiotic",
        "Clindamycine": "Antibiotic",
        "Clindamycin (Dalacin)": "Antibiotic",
        "dalacin": "Antibiotic",
        "Clindamycin": "Antibiotic",
        "Clindamycinum": "Antibiotic",
        "Flucloxacillin": "Antibiotic",
        "Amikin (Amikacine)": "Antibiotic",
        "Teicoplaninum": "Antibiotic",
        "Daptomycin.": "Antibiotic",
        "Daptomycin": "Antibiotic",
        "azythromycine": "Antibiotic",
        "azithromycine": "Antibiotic",
        "Azythromycin": "Antibiotic",
        "Azithromicin": "Antibiotic",
        "Azithromycinum": "Antibiotic",
        "Azitromycin": "Antibiotic",
        "Macrolides": "Antibiotic",
        "Erythromycin": "Antibiotic",
        "Erythrocine": "Antibiotic",

        "Erythrocin (Erythromycine)": "Antibiotic",
        "Quinolone": "Antibiotic",
        "Chinolon": "Antibiotic",
        "Ciprofloxacine": "Antibiotic",
        "Ciprofloxan": "Antibiotic",
        "Cyprofloxacin": "Antibiotic",
        "Ciprofloxcain": "Antibiotic", # Typo
        "Norfloxacine": "Antibiotic",
        "penicilline": "Antibiotic",
        "Ospen (PÃ©nicilline V)": "Antibiotic",
        "Stabicillin (phenoxymethylpenicillin)": "Antibiotic",
        "ospen": "Antibiotic",
        "Amoxicillinum": "Antibiotic",
        "Amoxicillin": "Antibiotic",
        "amoxicilline": "Antibiotic",
        "Amoxillin": "Antibiotic",
        "co-amoxicillin": "Antibiotic",
        "Ceftazidimum": "Antibiotic",
        "Ceftazidime (Fortam)": "Antibiotic",
        "Ceftriaxonum": "Antibiotic",
        "Rocephin": "Antibiotic", # Ceftriaxone
        "Oftrioxel": "Antibiotic", # Ceftriaxone
        "cefluroxine": "Antibiotic",
        "cÃ©furoxime": "Antibiotic",
        "Carbapeneme": "Antibiotic",
        "Imvanz I / V , 3 X / Week": "Antibiotic", # Ertapenem
        "Ertapenem": "Antibiotic",
        "Vancomycine.": "Antibiotic",
        "Vancomycine": "Antibiotic",
        "Vancomycinum": "Antibiotic",
        "Vancomycine (vancocin)": "Antibiotic",
        "Polymyxin E": "Antibiotic",
        "polimyxin E": "Antibiotic",
        "Polimixine E/ Colistin": "Antibiotic",
        "Colistin": "Antibiotic",
        "aerosol colistin": "Antibiotic",
        "Doxycydinum": "Antibiotic",
        "Doxycyclin": "Antibiotic",
        "doxycicline": "Antibiotic",
        "Doxycycline": "Antibiotic",
        "Rifaximin": "Antibiotic",
        "Xifaxan (rifaximin)": "Antibiotic",
        "Xifafan (Rifamixine)": "Antibiotic",
        "Xifafan (Rifaximine)": "Antibiotic",
        "Xifafan": "Antibiotic",
        "Rifampicin": "Antibiotic",
        "Rifampicinum, Isoniazidum": "Antibiotic",
        "Rifampicine": "Antibiotic",
        "Pyrazinamidum": "Antibiotic",
        "Pyraminamide": "Antibiotic", # Typo
        "Isoniazid/Rimifon": "Antibiotic",
        "isoniazid": "Antibiotic",
        "Isoniazide pour TB en prophylaxie": "Antibiotic",
        "Tuberculostatic": "Antibiotic",
        "Ethambutol": "Antibiotic",
        "Garamycine": "Antibiotic", # Gentamicin
        "garamycine": "Antibiotic",
        "Gentamycine.": "Antibiotic",
        "Gentamnycine": "Antibiotic", # Typo
        "Sulfonamide": "Antibiotic",
        "Co-trimoxazole": "Antibiotic",
        "sulfadiazine (clindamycine )": "Antibiotic",
        "Linezolid": "Antibiotic",
        "Nivaquine": "Antiparasitic", # Chloroquine
        
        # Antifungal
        "Nystatinum-Zinci oxidum": "Antifungal",
        "Nystatinum Zinci oxidum": "Antifungal",
        "Nystatinmu/Multilind": "Antifungal",
        "Nystatinum/Multillind": "Antifungal",
        "Nystatinum-Zinci oxidum/Multilind": "Antifungal",
        "Mycostatine (Nystatine) Alternatively": "Antifungal",
        "nystatin": "Antifungal",
        "Amphotericinum B. Excipiens.": "Antifungal",
        "Antimycosique": "Antifungal",
        "Sporanox": "Antifungal", # Itraconazole
        "Anidulafungin": "Antifungal",
        "Mycolog topic": "Antifungal",
        
        # Antiparasitic
        "Mephaquin (MÃ©floquine)": "Antiparasitic",
        "Pyrimethaminum/Dapson": "Antiparasitic",
        "pyrimÃ©thamine": "Antiparasitic",
        "Albendazolum": "Antiparasitic",
        "Albendazol": "Antiparasitic",
        "Hydroxychloroquin (Plaquenil)": "Antiparasitic", # Also immunosuppressant
        "PlaquÃ©nil": "Antiparasitic",

        # Antiviral
        "Letermovir": "Antiviral",
        "letermovir": "Antiviral",
        "Intelence": "Antiviral", # Etravirine
        "Etravirine/Intelence": "Antiviral",
        "Vemlidy (Tenofovir alafenamid)": "Antiviral",
        "Tenofovir (Vemlidy)": "Antiviral",
        "Tenofovir (vemlidy)": "Antiviral",
        "Tenofovir alafenamid": "Antiviral",
        "Tenofovir alafenamid (Vemlidy)": "Antiviral",
        "Vemlidil": "Antiviral", # Typo
        "_Vemlidy (Tenofovir alafenamid)": "Antiviral",
        "Viread (TÃ©nofovir)": "Antiviral",
        "Valcanciclovir": "Antiviral", # Valganciclovir
        "Oseltamivir / Tamiflu": "Antiviral",
        "tameflu": "Antiviral", # Typo
        "Ribavirin": "Antiviral",
        "ribavirin": "Antiviral",
        "Copegus": "Antiviral", # Ribavirin
        "lamivudin": "Antiviral",
        "Daclasvir": "Antiviral",
        "Daclatasvir": "Antiviral",
        "Sovaldi": "Antiviral", # Sofosbuvir
        "Epclusa": "Antiviral", # Sofosbuvir/Velpatasvir
        "Harvoni (Anti HCV)": "Antiviral",
        "Hep-C-Therapy (Harvoni)": "Antiviral",
        "Enteclavir (barclude )": "Antiviral",
        "Entecavir / Baraclude": "Antiviral",
        "Baraclude (Entecavir)": "Antiviral",
        "Baraclude (Emtecavir)": "Antiviral", # Typo
        "baraclude": "Antiviral",
        "entecavir": "Antiviral",
        "AntÃ©cavir": "Antiviral", # Typo
        "Telbivudinum (Sebivo)": "Antiviral",
        "Sebivo (Telbivudine)": "Antiviral",
        "Adefovir (Hepsera)": "Antiviral",
        "anti-HIV treatment": "Antiviral",
        "Anti-HIV Treatment": "Antiviral",
        "antiviral HIV therapy": "Antiviral",
        "Virostatikum": "Antiviral",
        "antiviral Hep. B therapy": "Antiviral",
        "antiviral Hep.-C therapy": "Antiviral",
        "Integraseinhibitoren": "Antiviral",
        "NRTI, Virostatika/Retrovir": "Antiviral",
        "Celsentri": "Antiviral", # Maraviroc
        "Descory": "Antiviral", # Emtricitabine/Tenofovir
        "Descovy/Tivicay": "Antiviral",
        "Isenstress": "Antiviral", # Typo for Isentress (Raltegravir)
        "Isentress": "Antiviral",
        "Raltegravir": "Antiviral",
        "Pifeltro": "Antiviral", # Doravirine
        "Odefsey": "Antiviral",
        "Edurant": "Antiviral", # Rilpivirine
        "Edurant NB": "Antiviral",
        "Abacavir": "Antiviral",
        "Lamivudin / Abacavir / Dolutegravir": "Antiviral",
        "ATG (Lamivudin, Abacavir und Dolutegravir)": "Antiviral",
        "Dolutegravir": "Antiviral",
        "dolutegravir": "Antiviral",
        "Dolutegravier": "Antiviral", # Typo
        "anti-viraux": "Antiviral",
        "Remdesivir": "Antiviral",
        
        # --- Immunomodulators & Anti-inflammatories ---
        # Biologic/Immunosuppressant
        "Rituximab": "Biologic/Immunosuppressant",
        "rituximab": "Biologic/Immunosuppressant",
        "Rituximab (Mabthera)": "Biologic/Immunosuppressant",
        "Mabthera (Rituximab)": "Biologic/Immunosuppressant",
        "Mabthera": "Biologic/Immunosuppressant",
        "Rituximab (Rixathon)": "Biologic/Immunosuppressant",
        "Rituximab.": "Biologic/Immunosuppressant",
        "cure de rituximab": "Biologic/Immunosuppressant",
        "cure Rituximab": "Biologic/Immunosuppressant",
        "Cure Rituximab X4": "Biologic/Immunosuppressant",
        "cure Rituximab X 4": "Biologic/Immunosuppressant",
        "cure  Rituximab": "Biologic/Immunosuppressant",
        "Belatacept": "Biologic/Immunosuppressant",
        "belatacept": "Biologic/Immunosuppressant",
        "belatacept I/V": "Biologic/Immunosuppressant",
        "betalacept": "Biologic/Immunosuppressant", # Typo
        "Belatazept": "Biologic/Immunosuppressant", # Typo
        "Belactasept": "Biologic/Immunosuppressant", # Typo
        "Belataset 1 X / Month": "Biologic/Immunosuppressant",
        "BÃ©latacept  (I /V )": "Biologic/Immunosuppressant",
        "bÃ©latacpet (I / V )": "Biologic/Immunosuppressant",
        "BÃ©latacept ( I / V )": "Biologic/Immunosuppressant",
        "Belatacept 1 X / Month": "Biologic/Immunosuppressant",
        "Belatacept (Nujolix)": "Biologic/Immunosuppressant",
        "Nulojx": "Biologic/Immunosuppressant", # Typo for Nulojix
        "Eculizumab": "Biologic/Immunosuppressant",
        "Ã©culizumab (soliris ) I / V": "Biologic/Immunosuppressant",
        "Eculizimab I/V every 2 weeks": "Biologic/Immunosuppressant",
        "Soliris every 3 weeks": "Biologic/Immunosuppressant",
        "Eculizimab 1 X / Month": "Biologic/Immunosuppressant",
        "ATG/Eculizumab": "Biologic/Immunosuppressant",
        "Abatacept": "Biologic/Immunosuppressant",
        "Orencia": "Biologic/Immunosuppressant", # Abatacept
        "Adalimumab / Humira": "Biologic/Immunosuppressant",
        "Cyclophosphamid": "Biologic/Immunosuppressant",
        "Cyclophosphamide": "Biologic/Immunosuppressant",
        "Cyclophosphamid (Endoxan)": "Biologic/Immunosuppressant",
        "inter Endoxan Therapy": "Biologic/Immunosuppressant",
        "Etanercept s.c.": "Biologic/Immunosuppressant",
        "TNF-Inhibitor /Enbrel": "Biologic/Immunosuppressant",
        "Leflunomid/Arava": "Biologic/Immunosuppressant",
        "Thymoglobuline I/V": "Biologic/Immunosuppressant",
        "Thymoglobuline.": "Biologic/Immunosuppressant",
        "thymoglobuline  I / V": "Biologic/Immunosuppressant",
        "Immunglobulin anti-T-Lymphozyten human": "Biologic/Immunosuppressant",
        "Grafalon": "Biologic/Immunosuppressant", # Anti-thymocyte globulin
        "AEB 071": "Biologic/Immunosuppressant", # Sotrastaurin
        "Campath (Alemtuzumab)": "Biologic/Immunosuppressant",
        "Kineret 100mg": "Biologic/Immunosuppressant",
        "kineret": "Biologic/Immunosuppressant",
        "anakinra": "Biologic/Immunosuppressant",
        "Remicade": "Biologic/Immunosuppressant", # Infliximab
        "Actemra/Tocilizamab": "Biologic/Immunosuppressant",
        "stelara (ustekinumab)": "Biologic/Immunosuppressant",
        "Vedolizumab": "Biologic/Immunosuppressant",
        "Canakinumab": "Biologic/Immunosuppressant",
        "canakinumab": "Biologic/Immunosuppressant",
        "Interferon beta-1a": "Biologic/Immunosuppressant",

        # Corticosteroid
        "Dexamethasone.": "Corticosteroid",
        "dextamÃ©thasone": "Corticosteroid",
        "Infiltration of Dexamethasone.": "Corticosteroid",
        "Hydrocortison": "Corticosteroid",
        "Cortison": "Corticosteroid",
        "prednisone for side effect of cancer traitment": "Corticosteroid",
        "Prednisolonum": "Corticosteroid",
        "solumedrol": "Corticosteroid",
        "Solumedrol": "Corticosteroid",
        "SolumÃ©drol 500 mg./j.": "Cordinocosteroid",
        "solumedrol I / V": "Corticosteroid",
        "Solumedrol IV (5 bolus)": "Corticosteroid",
        "Solumedrol IV, 5 boluses": "Corticosteroid",
        "Methyl-prednisone I V": "Corticosteroid",
        "Budesonidum": "Corticosteroid",
        "Budesonid": "Corticosteroid",
        "Budenofalk 9 mg for 9 month": "Corticosteroid",
        "Florinet": "Corticosteroid", # Fludrocortisone
        "Glukokortikoid / Spiricort": "Corticosteroid",

        # Immunoglobulin Therapy (IVIG/SCIG)
        "IVIG": "Immunoglobulin Therapy (IVIG)",
        "IvIG": "Immunoglobulin Therapy (IVIG)",
        "iViG": "Immunoglobulin Therapy (IVIG)",
        "IGIV": "Immunoglobulin Therapy (IVIG)",
        "IV / IG": "Immunoglobulin Therapy (IVIG)",
        "I/V ig": "Immunoglobulin Therapy (IVIG)",
        "IV / Ig": "Immunoglobulin Therapy (IVIG)",
        "IG i/v": "Immunoglobulin Therapy (IVIG)",
        "s.c. IG": "Immunoglobulin Therapy (IVIG)",
        "IgG s.c.": "Immunoglobulin Therapy (IVIG)",
        "I V /  I G": "Immunoglobulin Therapy (IVIG)",
        "IV /IG treatment": "Immunoglobulin Therapy (IVIG)",
        "iv IG injection": "Immunoglobulin Therapy (IVIG)",
        "immunglobulin (ivIg)": "Immunoglobulin Therapy (IVIG)",
        "Cure IV / IG": "Immunoglobulin Therapy (IVIG)",
        "I/V ig cure": "Immunoglobulin Therapy (IVIG)",
        "Kiovig cure": "Immunoglobulin Therapy (IVIG)",
        "Kiovig I / V": "Immunoglobulin Therapy (IVIG)",
        "Kiovig I / V cure": "Immunoglobulin Therapy (IVIG)",
        "Kiovig cure X 3 days": "Immunoglobulin Therapy (IVIG)",
        "Kiovig X 3 Days": "Immunoglobulin Therapy (IVIG)",
        "Privigen": "Immunoglobulin Therapy (IVIG)",
        "Intratect": "Immunoglobulin Therapy (IVIG)",
        "IgG Privigen": "Immunoglobulin Therapy (IVIG)",
        "Immunglobulin (Zutectra)": "Immunoglobulin Therapy (IVIG)",
        "Cuvitru (Human normal immunoglobulin, subcutaneous)": "Immunoglobulin Therapy (IVIG)",
        "Cytomegalie-Immunglobulin human": "Immunoglobulin Therapy (IVIG)",
        "HÃ©patect": "Immunoglobulin Therapy (IVIG)", # Hepatitis B Immunoglobulin
        "hepatect": "Immunoglobulin Therapy (IVIG)",
        "Hepatitis-B-Immunglobulin": "Immunoglobulin Therapy (IVIG)",
        
        # Immunostimulant
        "Immunostimulants": "Immunostimulant",
        "Urovaxom": "Immunostimulant",

        # NSAID
        "Diclofenac / Voltaren": "NSAID",
        "Ibuprofen/Brufen": "NSAID",

        # --- Endocrine Agents ---
        # Antidiabetic (Biguanide/DPP-4 Inhibitor)
        "Janumet": "Antidiabetic (Combination)",
        "iardiance-Met": "Antidiabetic (Combination)", # Jardiance/Metformin
        
        # Antidiabetic - GLP-1 Receptor Agonist
        "GLP-1-Rezeptor-Agonisten": "GLP-1 Receptor Agonist",
        "GLP Analogon": "GLP-1 Receptor Agonist",
        "Injectable antidiabetic": "GLP-1 Receptor Agonist",
        "Semaglutide": "GLP-1 Receptor Agonist",
        "Semagluti": "GLP-1 Receptor Agonist", # Typo
        "ozempic": "GLP-1 Receptor Agonist",
        "Ozempic (semaglutid)": "GLP-1 Receptor Agonist",
        "Dulaglutid": "GLP-1 Receptor Agonist",
        "liraglutide (Victoza)": "GLP-1 Receptor Agonist",
        "Victtoza": "GLP-1 Receptor Agonist", # Typo
        "antidiabetic (victosa ) S / C": "GLP-1 Receptor Agonist",
        
        # Antidiabetic - SGLT2 Inhibitor
        "SGLT2 Inhibitor": "SGLT2 Inhibitor",
        "Jordiance": "SGLT2 Inhibitor", # Jardiance
        "jasdiance": "SGLT2 Inhibitor", # Typo
        "Daplagifozine": "SGLT2 Inhibitor", # Dapagliflozin
        "Forxiga": "SGLT2 Inhibitor", # Dapagliflozin
        
        # Antidiabetic (Other)
        "Orale antidiabetica": "Antidiabetic (Oral, unspecified)",
        "Linagliptin": "DPP-4 Inhibitor",
        "Trajenta": "DPP-4 Inhibitor", # Linagliptin
        "liaglipid": "DPP-4 Inhibitor", # Typo
        "Diamicron": "Sulfonylurea",

        # Antithyroid Agent
        "Neo-mercazole": "Antithyroid Agent",
        "neomercazole": "Antithyroid Agent",
        "nÃ©o-mercazole": "Antithyroid Agent",
        "Neomercazol: Basedow tt": "Antithyroid Agent",
        "Carbimazol / Neo Mercazol": "Antithyroid Agent",
        "Carbimazol / Neo-Mercazole": "Antithyroid Agent",
        "Thyrozol (hyperthyroidie )": "Antithyroid Agent",
        
        # Thyroid Hormone
        "thyroid hormone": "Thyroid Hormone",
        "Levothyroxin/Euthyrox": "Thyroid Hormone",
        "Levothyrosin/Euthyrox": "Thyroid Hormone",
        "Levothyroxin / Eltroxin": "Thyroid Hormone",
        "Levothyroxin-natrium / Eltroxin": "Thyroid Hormone",
        "Euthyrox (Levothyroxin)": "Thyroid Hormone",
        "Levothyroxinum natrium/ Eythyrox": "Thyroid Hormone",
        "(Levothyroxin natrium)": "Thyroid Hormone",
        "Levothyroxin / Euthyrax": "Thyroid Hormone", # Typo
        "Levothyroxinum natricum.": "Thyroid Hormone",
        "Euthyrox": "Thyroid Hormone",
        "Euythyrox": "Thyroid Hormone", # Typo
        "euthyrox / Eltroxine": "Thyroid Hormone",
        "etyrox": "Thyroid Hormone", # Typo
        "eltoxin": "Thyroid Hormone",
        "Elthyrox": "Thyroid Hormone", # Typo
        "Eltoxin": "Thyroid Hormone",
        "euthroxin": "Thyroid Hormone",
        "Thyrox": "Thyroid Hormone",

        # Hormonal Agent (Other)
        "Estriol": "Hormonal Agent",
        "Ovestin Ovula": "Hormonal Agent",
        "Estrogen Substitution": "Hormonal Agent",
        "Testosteron": "Hormonal Agent",
        "Testosteron-undecylat": "Hormonal Agent",
        "contraceptives": "Hormonal Agent",
        "Anticonceptivum": "Hormonal Agent",
        "Visanne": "Hormonal Agent", # Dienogest

        # --- Gastrointestinal Agents ---
        # Antacid
        "Alucol": "Antacid",
        
        # Antiemetic
        "Domperidon": "Antiemetic",
        "Domperidon/Motilium": "Antiemetic",
        "Metoclopramid (HCI)": "Antiemetic",
        "Metocolopramidi hydrochlorid": "Antiemetic",
        "Metoclopramid / Paspertin": "Antiemetic",
        "Metocroplamidi / Paspertin": "Antiemetic",
        "Tropisetron": "Antiemetic",
        "Tropisetron/Navoban": "Antiemetic",
        
        # Antiflatulent
        "Simeticon / Flatulex": "Antiflatulent",
        
        # Antispasmodic
        "Pinaveriumbromid": "Antispasmodic",
        "Scopolamini butylbromidi": "Antispasmodic",
        "Scopolamini butylbromidum": "Antispasmodic",
        "Scopolamini butylbromidum/Buscopan": "Antispasmodic",
        "anticholinergics": "Antispasmodic", # General but fits best here
        
        # Hepatobiliary Agent
        "acidum ursodeoxycholicum": "Hepatobiliary Agent",
        "Acidum ursodeoxycholicum": "Hepatobiliary Agent",
        "Ursodeoxycholic acid": "Hepatobiliary Agent",
        "Acidum ursodesoxycholicum": "Hepatobiliary Agent",

        # Laxative
        "Macrogolum": "Laxative",
        "Macrogolum  / Transipeg": "Laxative",
        "Macrogol, NaCl, Na-sulfat, Natriumhydrogencarbonat /  Transipeg": "Laxative",
        "Macrogol, NaCl, Na-sulfat, KCI, Natriumhydrogencarbonat / Transipeg": "Laxative",
        "Lactitol": "Laxative",
        "Lactitolum monohydricum": "Laxative",
        "Metamucil": "Laxative",
        "Natriumpicosulphat / Laxoberon": "Laxative",
        
        # Pancreatic Enzymes
        "Pancreatin": "Pancreatic Enzymes",
        
        # Probiotic
        "Saccharomyces boulardii lyophilisiert": "Probiotic",

        # Proton Pump Inhibitor (PPI)
        "Pantoprazolum / Zurcal": "Proton Pump Inhibitor (PPI)",
        "Pantoprazolum / Pantozol": "Proton Pump Inhibitor (PPI)",
        "Pantorprazol / Pantazol": "Proton Pump Inhibitor (PPI)",
        "Pantoprazol /Pantozol": "Proton Pump Inhibitor (PPI)",
        "Pantprazolum": "Proton Pump Inhibitor (PPI)",
        "Pantoprazolium": "Proton Pump Inhibitor (PPI)",
        "Pantorprazol": "Proton Pump Inhibitor (PPI)", # Typo
        "Pantoplazol": "Proton Pump Inhibitor (PPI)", # Typo
        "pantozol": "Proton Pump Inhibitor (PPI)",
        "Omeprazolum": "Proton Pump Inhibitor (PPI)",
        "Omeprazol / Omezol": "Proton Pump Inhibitor (PPI)",
        "Ozepranolum": "Proton Pump Inhibitor (PPI)", # Typo for Omeprazole
        "Esomep": "Proton Pump Inhibitor (PPI)",
        "Esomep 40 mg": "Proton Pump Inhibitor (PPI)",
        "esomeprazol": "Proton Pump Inhibitor (PPI)",
        "Esomepragel": "Proton Pump Inhibitor (PPI)", # Typo
        "Dexilaut": "Proton Pump Inhibitor (PPI)", # Dexlansoprazole
        "Rabeprazolum": "Proton Pump Inhibitor (PPI)",
        "Zantic": "H2 Receptor Blocker", # Ranitidine

        # --- Oncology Agents ---
        # Chemotherapy/Targeted Therapy
        "Chemotherapie": "Chemotherapy/Targeted Therapy",
        "Chemotheraphy": "Chemotherapy/Targeted Therapy",
        "chemotherapy": "Chemotherapy/Targeted Therapy",
        "CHOP (chemotherapy)": "Chemotherapy/Targeted Therapy",
        "FOLFOX (Chemo)": "Chemotherapy/Targeted Therapy",
        "FOLFIRI": "Chemotherapy/Targeted Therapy",
        "Immunochemotherapy R-CHOP-21": "Chemotherapy/Targeted Therapy",
        "Bendamustin": "Chemotherapy/Targeted Therapy",
        "Daratumumab (chemotherapy)": "Chemotherapy/Targeted Therapy",
        "daratumumab": "Chemotherapy/Targeted Therapy",
        "darzalex I/V": "Chemotherapy/Targeted Therapy",
        "Darzalex I/V": "Chemotherapy/Targeted Therapy",
        "Cetuximab (chemotherapy)": "Chemotherapy/Targeted Therapy",
        "Cetuximab": "Chemotherapy/Targeted Therapy",
        "Carboplatine (chemotherapy)": "Chemotherapy/Targeted Therapy",
        "Doxorubicin (Cytostatics)": "Chemotherapy/Targeted Therapy",
        "Doxorubicin Liposomal (chemotherapy)": "Chemotherapy/Targeted Therapy",
        "Cyclophosphamid (Cytostatics)": "Chemotherapy/Targeted Therapy",
        "Temodal (anti-nÃ©oplasique )": "Chemotherapy/Targeted Therapy",
        "cabozantimib": "Chemotherapy/Targeted Therapy",
        "Cabozantinib": "Chemotherapy/Targeted Therapy",
        "sorafenib": "Chemotherapy/Targeted Therapy",
        "lenvatinib": "Chemotherapy/Targeted Therapy",
        "Lenvatinib": "Chemotherapy/Targeted Therapy",
        "Gitluyro": "Chemotherapy/Targeted Therapy", # Gilteritinib
        "T-VEC I/V at 2 weeks": "Chemotherapy/Targeted Therapy", # Talimogene laherparepvec
        "Ibrance": "Chemotherapy/Targeted Therapy", # Palbociclib
        "Ipilimumab": "Chemotherapy/Targeted Therapy",
        "Ipilimumab aux 3 weeks": "Chemotherapy/Targeted Therapy",
        "Ipilimumab every 4 weeks": "Chemotherapy/Targeted Therapy",
        "Opdivo": "Chemotherapy/Targeted Therapy", # Nivolumab
        "Vilclad  TTT MyÃ©lome": "Chemotherapy/Targeted Therapy", # Velcade (Bortezomib)
        "Bortizomib TTT myÃ©lome": "Chemotherapy/Targeted Therapy",
        "kiprolis": "Chemotherapy/Targeted Therapy", # Kyprolis (Carfilzomib)
        "carfilzomib": "Chemotherapy/Targeted Therapy",
        "Panobinostat TTT myÃ©lome": "Chemotherapy/Targeted Therapy",
        "ixazomid": "Chemotherapy/Targeted Therapy",
        "Antineoplastika/Immunmodulator": "Chemotherapy/Targeted Therapy",
        "Chemotherapy (Carboplatin/Gemcitabin)": "Chemotherapy/Targeted Therapy",
        "Votubia /Everolimus": "Chemotherapy/Targeted Therapy",
        "Everolimus": "Chemotherapy/Targeted Therapy",
        "Abemaciclib": "Chemotherapy/Targeted Therapy",
        
        # Hormonal Therapy (Oncology)
        "Arimidex": "Hormonal Therapy (Oncology)",
        "Goserelin": "Hormonal Therapy (Oncology)",
        "Goserelin/Zoladex": "Hormonal Therapy (Oncology)",
        "Tamoxifenum": "Hormonal Therapy (Oncology)",
        "Tamixifencitrat": "Hormonal Therapy (Oncology)",
        "Fulvestrant": "Hormonal Therapy (Oncology)",
        "Firmagon": "Hormonal Therapy (Oncology)", # Degarelix
        "AROMASIN": "Hormonal Therapy (Oncology)", # Exemestane
        "taitement for cancer prostate": "Hormonal Therapy (Oncology)", # General description

        # --- Hematologic Agents ---
        # Anticoagulant
        "Liquemine": "Anticoagulant", # Heparin
        "Calciparine": "Anticoagulant", # Heparin
        "Apixaban": "Anticoagulant",
        "Eliquis": "Anticoagulant",
        "Eliquin": "Anticoagulant", # Typo
        "Fondaparinux-Natrium": "Anticoagulant",

        # Antiplatelet
        "Plavix": "Antiplatelet",
        "BRILIQUE": "Antiplatelet",

        # Erythropoiesis-Stimulating Agent (ESA)
        "EPO, Aranesp": "Erythropoiesis-Stimulating Agent (ESA)",
        "Dabepoetin alpha": "Erythropoiesis-Stimulating Agent (ESA)",
        "Darbepoetin alpha": "Erythropoiesis-Stimulating Agent (ESA)",
        "Darbepoetin beta": "Erythropoiesis-Stimulating Agent (ESA)",
        "Darbepoetin alfa": "Erythropoiesis-Stimulating Agent (ESA)",
        "Aranesp 1X/sem": "Erythropoiesis-Stimulating Agent (ESA)",
        "Darbepoetin alfa / Aranesp": "Erythropoiesis-Stimulating Agent (ESA)",
        "Epoetanum beta": "Erythropoiesis-Stimulating Agent (ESA)",
        "Epoetinum beta/ Recormon": "Erythropoiesis-Stimulating Agent (ESA)",
        "Epoetinum beta/Mircera": "Erythropoiesis-Stimulating Agent (ESA)",
        "PEG-Epoetin / Mircera": "Erythropoiesis-Stimulating Agent (ESA)",
        "PEG-Epoetin / Mircere": "Erythropoiesis-Stimulating Agent (ESA)", # Typo
        "Epoetin alfa / Eprex": "Erythropoiesis-Stimulating Agent (ESA)",

        # Colony-Stimulating Factor
        "Neupogen": "Colony-Stimulating Factor", # Filgrastim

        # Iron Supplement
        "Ferrum": "Iron Supplement",
        "Ferrum / Venofer": "Iron Supplement",
        "Venofer": "Iron Supplement",
        "ferinject": "Iron Supplement",
        "Ferinject": "Iron Supplement",
        "perfusion Ferinject": "Iron Supplement",
        "ferrinject i/v": "Iron Supplement",
        "Eisen / Ferinject": "Iron Supplement",
        "Ferinjekt/Pentamide": "Iron Supplement", # Ferinject part
        "Eisencarboxymaltose": "Iron Supplement",
        "Maltofer": "Iron Supplement",

        # --- Renal & Metabolic Agents ---
        # Bicarbonate Supplement
        "Hydrogencarbonas": "Bicarbonate Supplement",
        "Natrii hydrogencarbonas": "Bicarbonate Supplement",
        "sodiumbicarbonate": "Bicarbonate Supplement",
        "Natriumhydrogencarbonat": "Bicarbonate Supplement",
        "sodium bicarbonat": "Bicarbonate Supplement",
        "Natriumhydrogencarbonat /Nephrotrans": "Bicarbonate Supplement",
        "Natrii hydrogencarbonas / Nephrotrans": "Bicarbonate Supplement",
        "nephrotrans": "Bicarbonate Supplement",
        "Nephrotrans 500mg": "Bicarbonate Supplement",

        # Calcimimetic
        "Cinacalcet": "Calcimimetic",
        "Cinacalvet": "Calcimimetic", # Typo
        "Cinacalcef": "Calcimimetic", # Typo
        "Cinacalcet.": "Calcimimetic",
        "Cinacalcet)": "Calcimimetic",
        "Cinacalcet/Mimpara": "Calcimimetic",
        "Mimpara/cinacalcet": "Calcimimetic",
        
        # Metabolic Disorder Agent
        "Cystagon": "Metabolic Disorder Agent",
        "CYSTAGON": "Metabolic Disorder Agent",
        "Cystagon/ Cystamin": "Metabolic Disorder Agent",
        "Cysteamin": "Metabolic Disorder Agent",
        "Cysteamine": "Metabolic Disorder Agent",

        # Phosphate Binder
        "Ca Acetat": "Phosphate Binder",
        "Ca Azetat": "Phosphate Binder",
        "Ca - Azetat": "Phosphate Binder",
        "Ca/Acetat": "Phosphate Binder",
        "Ca- Acetat": "Phosphate Binder",
        "Calcium-Acetat": "Phosphate Binder",
        "calciumacetat": "Phosphate Binder",
        "Sevelamer/ Renagel": "Phosphate Binder",
        "Fosrenol": "Phosphate Binder", # Lanthanum carbonate
        "Phopsphocaps": "Phosphate Binder", # Typo
        "Phosphocaps": "Phosphate Binder",
        "Chloridhydroxid": "Phosphate Binder", # Sevelamer
        "Chloridhydroxid/Phosphonorm": "Phosphate Binder",
        "Calcifriat": "Phosphate Binder", # Calcium Acetate/Magnesium Carbonate
        
        # Potassium Binder
        "sorbisterit": "Potassium Binder",
        "Polystyrolsulfonat": "Potassium Binder",
        "Sulfoniertes Kopolymer": "Potassium Binder",
        "Kopolymer": "Potassium Binder",
        
        # Uric Acid Reducer
        "Allopurinol": "Uric Acid Reducer",
        "Alopurinol": "Uric Acid Reducer",
        "Allopurinl": "Uric Acid Reducer",
        "Allopupinol": "Uric Acid Reducer", # Typo
        "Allopurinnol": "Uric Acid Reducer", # Typo
        "all'opur": "Uric Acid Reducer",
        "Zylolic": "Uric Acid Reducer", # Allopurinol
        "Allopurinol / Zyloric": "Uric Acid Reducer",
        "Allopurinol (Urikostatikum)": "Uric Acid Reducer",
        "Adenuric": "Uric Acid Reducer",
        "Aduneric": "Uric Acid Reducer", # Typo
        "AdÃ©nuric": "Uric Acid Reducer",
        "AdÃ©nuric.": "Uric Acid Reducer",
        "Adenuric Febuxostat": "Uric Acid Reducer",
        "fasturtec iv (gut crisis)": "Uric Acid Reducer", # Rasburicase
        
        # Anti-gout Agent
        "Colchicin": "Anti-gout Agent",
        "Colchicine.": "Anti-gout Agent",
        "Colchin": "Anti-gout Agent", # Typo
        "Colchine": "Anti-gout Agent", # Typo
        
        # --- Respiratory & Allergy Agents ---
        # Antihistamine
        "Clemastin": "Antihistamine",
        "Clemastin/Tavegyl": "Antihistamine",
        "Clemastine/Tavegyl": "Antihistamine",
        "Antihystaminikum / Tavegyl": "Antihistamine",
        "Citrizinum/Cetrin": "Antihistamine",
        "Ceterizinum/Cetrin": "Antihistamine",
        "Xyzal": "Antihistamine",
        
        # Inhaled Corticosteroid/Bronchodilator
        "antiasthmatic drug": "Inhaled Corticosteroid/Bronchodilator",
        "Antiasthmatic": "Inhaled Corticosteroid/Bronchodilator",
        "Beta-2-Sympathomimetikum": "Inhaled Corticosteroid/Bronchodilator",
        "Bronchodilatans/ Ã¢ 2 -Sympathomimetikum": "Inhaled Corticosteroid/Bronchodilator",
        "Formoterol/Budesonid": "Inhaled Corticosteroid/Bronchodilator",
        "Budesonit, Formoterol / Symbicort": "Inhaled Corticosteroid/Bronchodilator",
        "Formoterol / Symbiocort": "Inhaled Corticosteroid/Bronchodilator",
        "Symbicort Spray": "Inhaled Corticosteroid/Bronchodilator",
        "Symbicort/ Dospir": "Inhaled Corticosteroid/Bronchodilator",
        "Symbicort; Ventolin": "Inhaled Corticosteroid/Bronchodilator",
        "Symbicort / Spriva": "Inhaled Corticosteroid/Bronchodilator", # Typo, should be Spiriva
        "Formoterolum": "Inhaled Corticosteroid/Bronchodilator",
        "Salmeterolum": "Inhaled Corticosteroid/Bronchodilator",
        "Formoterol / Foradil": "Inhaled Corticosteroid/Bronchodilator",
        "Formoterol / Oxis": "Inhaled Corticosteroid/Bronchodilator",
        "Tiotropium / Spiriva": "Inhaled Corticosteroid/Bronchodilator",
        "Tiotropium Bromide": "Inhaled Corticosteroid/Bronchodilator",
        "Ipratropiumbromid / Atrovent": "Inhaled Corticosteroid/Bronchodilator",
        "Ipramol": "Inhaled Corticosteroid/Bronchodilator", # Ipratropium/Salbutamol
        "Relvar Elipta": "Inhaled Corticosteroid/Bronchodilator",
        "Ultibro": "Inhaled Corticosteroid/Bronchodilator",
        "Enerzair": "Inhaled Corticosteroid/Bronchodilator",
        "Avamys": "Inhaled Corticosteroid/Bronchodilator", # Fluticasone furoate
        "Alvesco": "Inhaled Corticosteroid/Bronchodilator", # Ciclesonide
        "Anora": "Inhaled Corticosteroid/Bronchodilator",
        
        # Leukotriene Antagonist
        "Leukotriene antagonists (antiasthmatic)": "Leukotriene Antagonist",
        "Singulair": "Leukotriene Antagonist", # Montelukast
        "Lukair": "Leukotriene Antagonist", # Montelukast
        
        # --- Urological Agents ---
        # 5-alpha-reductase Inhibitor
        "Finasterid / Proscar": "5-alpha-reductase Inhibitor",
        
        # Anticholinergic (Bladder)
        "vesicare": "Anticholinergic (Bladder)",
        "Vesicare": "Anticholinergic (Bladder)",
        "Solifenacin / Vesicare": "Anticholinergic (Bladder)",
        "Trospiumchlorid / Spasmo-Urgenin Neo": "Anticholinergic (Bladder)",
        "Trospium chlorid / Spasmo-Urgenin": "Anticholinergic (Bladder)",
        "Trospiumchlorid/Spasmo Urgenin": "Anticholinergic (Bladder)",
        "Detrusitol": "Anticholinergic (Bladder)", # Tolterodine
        "Toviaz": "Anticholinergic (Bladder)", # Fesoterodine
        "Toviaz 4mg/tgl": "Anticholinergic (Bladder)",
        
        # PDE5 Inhibitor
        "Tadalafil (Cialis)": "PDE5 Inhibitor",
        "Tadalafil": "PDE5 Inhibitor",
        "Tadala": "PDE5 Inhibitor", # Typo
        "Adcurca": "PDE5 Inhibitor", # Tadalafil
        "sildenafil": "PDE5 Inhibitor",
        "Sildenofil": "PDE5 Inhibitor", # Typo
        
        # --- Lipid-Lowering Agents ---
        # Cholesterol Absorption Inhibitor
        "antihyperlipidemia ttt other than statin (ezetrol)": "Cholesterol Absorption Inhibitor",
        "ttt hyperlipidemia other than statin (Ezetrol)": "Cholesterol Absorption Inhibitor",
        "Cholesterinabsorptionshemmer": "Cholesterol Absorption Inhibitor",
        "Ezetimib.": "Cholesterol Absorption Inhibitor",
        "Ezetemib": "Cholesterol Absorption Inhibitor", # Typo
        "Zertimib": "Cholesterol Absorption Inhibitor", # Ezetimibe
        "azetidone": "Cholesterol Absorption Inhibitor", # Chemical class of Ezetimibe
        "EZETROL": "Cholesterol Absorption Inhibitor",

        # PCSK9 inhibitor
        "lipid-lowering agent (pravulent)": "PCSK9 inhibitor", # Praluent (Alirocumab)
        "Alirocumab": "PCSK9 inhibitor",
        "Alirocumab (praluent)": "PCSK9 inhibitor",

        # Statin
        "Fluvastatin/Lescol": "Statin",
        "Fluvastatin": "Statin",

        # --- Supplements & Minerals ---
        # Bone Density Agent
        "Bisphosphonat (Alendronat)": "Bone Density Agent",
        "Alendronat/Fosamax": "Bone Density Agent",
        "AlendronsÃ¤ure (Fosamax)": "Bone Density Agent",
        "Alendronate": "Bone Density Agent",
        "Alendronat": "Bone Density Agent",
        "Alendromate": "Bone Density Agent", # Typo
        "Alendromat": "Bone Density Agent", # Typo
        "actonel": "Bone Density Agent", # Risedronate
        "Bonivia": "Bone Density Agent",
        "Bonviva 1 X / 3 MOIS": "Bone Density Agent",
        "IbandronsÃ¤ure / Bonviva": "Bone Density Agent",
        "Pamidronat": "Bone Density Agent",
        "Pamidronasacidum": "Bone Density Agent",
        "Pamidronat / Aredia": "Bone Density Agent",
        "ZoledronsÃ¤ure": "Bone Density Agent",
        "acide zolodronique (Zeneta)": "Bone Density Agent",
        "Teriparatid (Forsteo)": "Bone Density Agent",

        # Calcium/Vitamin D Supplement
        "Cacitrol": "Calcium/Vitamin D Supplement",
        "Cacitriol (Rocaltrol)": "Calcium/Vitamin D Supplement",
        "rocaltrol": "Calcium/Vitamin D Supplement",
        "Nocaltol": "Calcium/Vitamin D Supplement", # Typo
        "Calcitriol / Rocaltrol": "Calcium/Vitamin D Supplement",
        "Synth. Calitriol": "Calcium/Vitamin D Supplement",
        "Synth. Calcitriol/Rocaltrol": "Calcium/Vitamin D Supplement",
        "Cynthetic calcitrol": "Calcium/Vitamin D Supplement",
        "Vit D / Rocaltrol": "Calcium/Vitamin D Supplement",
        "Vit. D / Rocaltrol": "Calcium/Vitamin D Supplement",
        "Paricalcitol/Zemplar": "Calcium/Vitamin D Supplement",
        "Paricalcitol / Zemplar": "Calcium/Vitamin D Supplement",
        "ViDe3": "Calcium/Vitamin D Supplement",
        "Vi De3": "Calcium/Vitamin D Supplement",
        "Vit D3": "Calcium/Vitamin D Supplement",
        "Vit D 3": "Calcium/Vitamin D Supplement",
        "VIT D3": "Calcium/Vitamin D Supplement",
        "Viamin D": "Calcium/Vitamin D Supplement", # Typo
        "Colecalciferol / Vi.D3": "Calcium/Vitamin D Supplement",
        "Colecalciferol Vit D3": "Calcium/Vitamin D Supplement",
        "Calcilum D3": "Calcium/Vitamin D Supplement",
        "Ca-Vit D3": "Calcium/Vitamin D Supplement",
        "Ca D3": "Calcium/Vitamin D Supplement",
        "Ca Vit 3": "Calcium/Vitamin D Supplement",
        "Calcium Vitamin D": "Calcium/Vitamin D Supplement",
        "Calcimofen D3": "Calcium/Vitamin D Supplement",
        "Calcimagone": "Calcium/Vitamin D Supplement",
        
        # Electrolyte Supplement
        "KCL DragÃ©es": "Electrolyte Supplement",
        "KCL Dragees": "Electrolyte Supplement",
        "Kalium Hausmann": "Electrolyte Supplement",
        "Laktat/Citrat /Magnegon": "Electrolyte Supplement", # Magnesium
        "Mg / Magnesium Diasporal": "Electrolyte Supplement",
        "Mg / Magnesiocard": "Electrolyte Supplement",
        "Natrium Chlorid": "Electrolyte Supplement",

        # Mineral Supplement
        "Zink Verla": "Mineral Supplement",
        
        # Joint Health Supplement
        "Condroxulf": "Joint Health Supplement", # Chondroitin sulfate
        
        # Vitamin Supplement
        "Acidum folicum": "Vitamin Supplement (B9)",
        "folic acid": "Vitamin Supplement (B9)",
        "calcium folinat": "Vitamin Supplement (B9)",
        "Vitamin B12": "Vitamin Supplement (B12)",
        "Cyanocobalaminum": "Vitamin Supplement (B12)",
        "Cianocobalaminum": "Vitamin Supplement (B12)",
        "Cyanoccobalaminum/Vitarubin": "Vitamin Supplement (B12)",
        "Vitarubin": "Vitamin Supplement (B12)",
        "Vitarubine": "Vitamin Supplement (B12)",
        "Vitamin B6": "Vitamin Supplement (B6)",
        "Vit B6": "Vitamin Supplement (B6)",
        "Pyridoxin": "Vitamin Supplement (B6)",
        "Vit B1,2,6/C/FolsÃ¤ure": "Vitamin Supplement (B-Complex)",
        "Vit B1,2,6/C/FolsÃ¤ure / Dialvit": "Vitamin Supplement (B-Complex)",
        "Vit.B Komplex / Becotal": "Vitamin Supplement (B-Complex)",
        "Nicotinamide": "Vitamin Supplement (B3)",
        "VItamin K/ Konakion": "Vitamin Supplement (K)",
        "Acetylcystein": "Mucolytic/Antidote", # N-acetylcysteine
        "Methionin / Acimethin": "Amino Acid Supplement",
        
        # --- Miscellaneous ---
        # Dermatological Agent
        "Neotigason": "Dermatological Agent", # Acitretin
        "Acitretin / Neotigason": "Dermatological Agent",
        "acitretin": "Dermatological Agent",
        "Isotretinoin": "Dermatological Agent",
        "Isotretinoine": "Dermatological Agent",
        "Priorin": "Dermatological Agent", # Hair loss supplement
        "Prionin": "Dermatological Agent", # Typo
        
        # Smoking Cessation Aid
        "Nicotin patch": "Smoking Cessation Aid",

        # Herbal/Dietary Supplement
        "Cimifemin": "Herbal/Dietary Supplement",
        
        # Vaccine
        "Vaccin H1N1": "Vaccine",
        "Meniingokokken-Impfung (Menveo)": "Vaccine",
    
        ###############
        # ADDED LATER #
        ###############

        # --- Cardiovascular Agents ---
        "Physiotens (Moxonidinum)": "Centrally Acting Agent",
        "Physiotens": "Centrally Acting Agent",
        "Central Blocker": "Centrally Acting Agent",
        "central blocker (physiotens)": "Centrally Acting Agent",
        "Physiotens (Central adrenergic)": "Centrally Acting Agent",
        "Moxonidin (Physiotens)": "Centrally Acting Agent",
        "Catapressan": "Centrally Acting Agent",
        "central-blocker": "Centrally Acting Agent",
        "Clonidin": "Centrally Acting Agent",
        "clonidin": "Centrally Acting Agent",
        "Clonidini hydrochloridum": "Centrally Acting Agent",
        "central blocker ( catapressan )": "Centrally Acting Agent",
        "alpha-blocker (catapressan )": "Centrally Acting Agent",
        "alpha blocker ( catapressan )": "Centrally Acting Agent",
        "alpha blocker ( Catapressan )": "Centrally Acting Agent",
        "Thiazid": "Diuretic",
        "Thiazide": "Diuretic",
        "Thiazideddiusetic": "Diuretic",
        "Metolazonum": "Diuretic",
        "Metolazon": "Diuretic",
        "Metalozonum": "Diuretic",
        "Metalozon": "Diuretic",
        "Metolazonum.": "Diuretic",
        "diuretic (metolazone)": "Diuretic",
        "Spironolactonum": "Diuretic",
        "Spironolacton": "Diuretic",
        "Aldactone": "Diuretic",
        "aldactone": "Diuretic",
        "diuretic ( aldactone )": "Diuretic",
        "diuretics (aldactone )": "Diuretic",
        "diuretic (spironolactone)": "Diuretic",
        "ALdactone": "Diuretic",
        "Spironolacton / Aldacton": "Diuretic",
        "Eplerenon": "Diuretic",
        "diuretic(inspra)": "Diuretic",
        "Furosemidum": "Diuretic",
        "Furosemid / Lasix": "Diuretic",
        "diuretic (lasix)": "Diuretic",
        "Furosemid/Lasix": "Diuretic",
        "Forosemic / Lasix": "Diuretic",
        "diuretic (torem )": "Diuretic",
        "diuretic (torasemid)": "Diuretic",
        "Toremisid": "Diuretic",
        "Torasemid/Torem": "Diuretic",
        "Toresemid": "Diuretic",
        "Torasamid": "Diuretic",
        "Torasemidum.": "Diuretic",
        "Hydrochlorothiazid": "Diuretic",
        "Hydrochlorothiazidum": "Diuretic",
        "Hydrochlorothiazide": "Diuretic",
        "Comilorid": "Diuretic",
        "diuretic (edarbyclore)": "ARB/Diuretic Combo",
        "Chlortalidon": "Diuretic",
        "Chlorthalidon": "Diuretic",
        "Diuretics": "Diuretic",
        "diuretic (torem)": "Diuretic",
        "Diuretic (torem)": "Diuretic",
        "Diuretic ( esidrex)": "Diuretic",
        "Diuretikum": "Diuretic",
        "Diuretika": "Diuretic",
        "Diuretic ( + )": "Diuretic",
        "Cordarone (Amiodarone)": "Antiarrhythmic",
        "cordarone": "Antiarrhythmic",
        "Amiodarone": "Antiarrhythmic",
        "amiodarone": "Antiarrhythmic",
        "Amiodaron (Cordarone)": "Antiarrhythmic",
        "Amiodaroni hydrochloridum": "Antiarrhythmic",
        "Amidarone": "Antiarrhythmic",
        "Amidaron": "Antiarrhythmic",
        "Amiodaron / Cordarone": "Antiarrhythmic",
        "Digoxin": "Antiarrhythmic",
        "Digoxine": "Antiarrhythmic",
        "Digoxinum": "Antiarrhythmic",
        "Labetalol": "Beta-blocker",
        "Atenololum": "Beta-blocker",
        "alpha-Blocker": "Alpha-blocker",
        "Alpha-blocker": "Alpha-blocker",
        "alpha-blocker (cardura)": "Alpha-blocker",
        "Alpha Blocker": "Alpha-blocker",
        "Doxazosin / Cardura": "Alpha-blocker",
        "Cardura (Doxazosin mesylate)": "Alpha-blocker",
        "alpha-blocker (cardura )": "Alpha-blocker",
        "Doxazosine (Cardura)": "Alpha-blocker",
        "Cardura (doxazosine)": "Alpha-blocker",
        "Tamsulosini Hydrochloridum": "Alpha-blocker",
        "Tamsulosini": "Alpha-blocker",
        "Tamsulosin / Pradif": "Alpha-blocker",
        "Tamsulosini hydrochlorid": "Alpha-blocker",
        "Chinazoline": "Alpha-blocker",
        "Duodart": "Alpha-blocker",
        "Duodart (DutastÃ©ride + tamsulosine)": "Alpha-blocker",
        "Nitroderm": "Vasodilator",
        "nitroderm patch": "Vasodilator",
        "Nitroglycerin": "Vasodilator",
        "Dancor": "Vasodilator",
        "Molsidomin": "Vasodilator",
        "Nitro": "Vasodilator",
        "Antihypertonikum / Loniten": "Vasodilator",
        "Corvaton": "Vasodilator",
        "Minoxidilum": "Vasodilator",
        "Opsumit": "Vasodilator",
        "Nitroderm patch": "Vasodilator",
        "Nitroglycerin (Nitroderm)": "Vasodilator",
        "Nicorandil (Dancor)": "Vasodilator",
        "Corvaton retard": "Vasodilator",
        "Nitroderm TTS": "Vasodilator",
        "Loniten (minoxidil)": "Vasodilator",
        "Minoxidil": "Vasodilator",
        "Loniten": "Vasodilator",
        "Noradrenalin": "Vasopressor",
        "Ranolazin": "Anti-anginal",
        "Adalat": "CCB",
        "Amloclipine": "CCB",
        "Rivaroxaban": "Anticoagulant",
        "Sintrom": "Anticoagulant",
        "Efient": "Antiplatelet",
        "Clopidogrel": "Antiplatelet",

        # --- Neurology & Psychiatry Agents ---
        "Lyrica": "Anticonvulsant",
        "Pregabalin": "Anticonvulsant",
        "Keppra": "Anticonvulsant",
        "Levetiracetam": "Anticonvulsant",
        "Antiepileptika": "Anticonvulsant",
        "Levetiracetam / Keppra": "Anticonvulsant",
        "Depakine": "Anticonvulsant",
        "Lamotrigin / Lamictal": "Anticonvulsant",
        "Vimpat": "Anticonvulsant",
        "Carbamazepin": "Anticonvulsant",
        "Lamictal": "Anticonvulsant",
        "Levitriactam": "Anticonvulsant",
        "Levetiracetam (keppra)": "Anticonvulsant",
        "Lamotrigin": "Anticonvulsant",
        "Aphenylbarbit": "Anticonvulsant",
        "Gabapenta": "Anticonvulsant",
        "Cipralex": "Antidepressant",
        "Citalopram": "Antidepressant",
        "Escitalopram": "Antidepressant",
        "Sertralin": "Antidepressant",
        "Antidepressivum": "Antidepressant",
        "Mirtazapinum": "Antidepressant",
        "Remeron": "Antidepressant",
        "Cymbalta": "Antidepressant",
        "Trittico": "Antidepressant",
        "Saroten": "Antidepressant",
        "Antidepressiva": "Antidepressant",
        "Mirtazapinum / Remeron": "Antidepressant",
        "Citalopramum": "Antidepressant",
        "citalopram": "Antidepressant",
        "Citalopranum": "Antidepressant",
        "Trazodon": "Antidepressant",
        "Antidepressive drug": "Antidepressant",
        "Deanxit": "Antidepressant",
        "Trimipramin": "Antidepressant",
        "escitalopram": "Antidepressant",
        "Mirtazepin": "Antidepressant",
        "Madopar": "Antiparkinson Agent",
        "Levodopum": "Antiparkinson Agent",
        "Pramipexol": "Antiparkinson Agent",
        "Sifrol": "Antiparkinson Agent",
        "Levodopa (Antiparkinson)": "Antiparkinson Agent",
        "Levodopa": "Antiparkinson Agent",
        "Alprazolam": "Anxiolytic/Sedative",
        "Lorazepamum": "Anxiolytic/Sedative",
        "Temesta": "Anxiolytic/Sedative",
        "Zolpidemtartrat": "Anxiolytic/Sedative",
        "Oxazepam": "Anxiolytic/Sedative",
        "Zolpidemi tartras": "Anxiolytic/Sedative",
        "Zolpidem / Stilnox": "Anxiolytic/Sedative",
        "Oxazepam / Seresta": "Anxiolytic/Sedative",
        "Zolpidem": "Anxiolytic/Sedative",
        "Urbanyl": "Anxiolytic/Sedative",
        "Zolpidem/Stilnox": "Anxiolytic/Sedative",
        "Quetiapin": "Antipsychotic",
        "Quetiapin / Seroquel": "Antipsychotic",
        "Olanzapin": "Antipsychotic",
        "Neuroleptika": "Antipsychotic",
        "Quetiapinum": "Antipsychotic",
        "Haloperidol": "Antipsychotic",
        "Seroquel": "Antipsychotic",
        "Abilify": "Antipsychotic",
        "Pipamperon / Dipiperon": "Antipsychotic",
        "Pipamperonum/Dipiperon": "Antipsychotic",
        "Paracetamolum": "Analgesic (Non-Opioid)",
        "Paracetamolum / Dafalgan": "Analgesic (Non-Opioid)",
        "Mephanol": "Analgesic (Non-Opioid)",
        "Pyrazolonderivat/Novalgin": "Analgesic (Non-Opioid)",
        "Pyrazolonderivat / Novalgin": "Analgesic (Non-Opioid)",
        "Pyrazolonderivat": "Analgesic (Non-Opioid)",
        "Nopilforte": "Analgesic (Non-Opioid)",
        "Methadon": "Opioid Analgesic",
        "Morphin": "Opioid Analgesic",
        "Temgesic": "Opioid Analgesic",
        "Betaserc": "Anti-vertigo Agent",

        # --- Antimicrobials ---
        "Monuril (Fosfomycine)": "Antibiotic",
        "Uvamine R (NitrofurantoÃ¯ne)": "Antibiotic",
        "Vancomycine (Vancocin)": "Antibiotic",
        "Vancomycin (Vancocin)": "Antibiotic",
        "Isoniazid": "Antibiotic",
        "Uvamine Retard (Nitrofurantoin)": "Antibiotic",
        "azithromycin": "Antibiotic",
        "Stabicilline <phÃ©noxymÃ©thylpÃ©nicilline (pÃ©nicilline V)>": "Antibiotic",
        "Azithromycin": "Antibiotic",
        "Ospen": "Antibiotic",
        "Polymyxine B + NÃ©omycine": "Antibiotic",
        "monuril": "Antibiotic",
        "Fosfomycin": "Antibiotic",
        "Monuril": "Antibiotic",
        "Isoniazide": "Antibiotic",
        "Gentamycine": "Antibiotic",
        "Polymixin E": "Antibiotic",
        "fosfomycine (monuril)": "Antibiotic",
        "Dapson/Pyrimethamin": "Antibiotic",
        "Clindamycine.": "Antibiotic",
        "Dapson": "Antibiotic",
        "Ciprofloxacin": "Antibiotic",
        "Vancomycin": "Antibiotic",
        "Penicilline benzathine": "Antibiotic",
        "nitrofurantoine": "Antibiotic",
        "Vancomycin (Vancocine)": "Antibiotic",
        "Azithromycine (Zithromax)": "Antibiotic",
        "Furandantine R (NitrofurantoÃ¯ne)": "Antibiotic",
        "Uvamine": "Antibiotic",
        "Zithromax": "Antibiotic",
        "fosfomycine": "Antibiotic",
        "Moxifloxacin": "Antibiotic",
        "Acitromycin": "Antibiotic",
        "Ethambutoli hydrochloridum": "Antibiotic",
        "Isonazid": "Antibiotic",
        "Isoniazidum": "Antibiotic",
        "Ospen (Phenoxymethylpenicillinum kalicum)": "Antibiotic",
        "uvamine": "Antibiotic",
        "Fosmomycin / Monuril": "Antibiotic",
        "Stabilline": "Antibiotic",
        "vancomycin": "Antibiotic",
        "polymixin E": "Antibiotic",
        "Furadantine": "Antibiotic",
        "Fosfomycinum": "Antibiotic",
        "furadentine": "Antibiotic",
        "B-Lactame": "Antibiotic",
        "Mycostatine (Nystatine)": "Antifungal",
        "Nystatinum/Multilind": "Antifungal",
        "Amphotericin": "Antifungal",
        "Lamivudine": "Antiviral",
        "Tivicay": "Antiviral",
        "Tamiflu": "Antiviral",
        "Baraclude": "Antiviral",
        "Entecavir": "Antiviral",
        "Tenofovir": "Antiviral",
        "Ziagen": "Antiviral",
        "Triumeq": "Antiviral",
        "Sofosbuvir": "Antiviral",
        "Ribavirine": "Antiviral",
        "Vemlidy": "Antiviral",

        # --- Immunomodulators & Anti-inflammatories ---
        "Belatacept (Nulojix)": "Biologic/Immunosuppressant",
        "Etanercept (Enbrel)": "Biologic/Immunosuppressant",
        "Enbrel": "Biologic/Immunosuppressant",
        "Etanercept": "Biologic/Immunosuppressant",
        "Enbrel (etanercept)": "Biologic/Immunosuppressant",
        "Leflunomid": "Biologic/Immunosuppressant",
        "Thymoglobulin": "Biologic/Immunosuppressant",
        "Thymoglobuline": "Biologic/Immunosuppressant",
        "ATG": "Biologic/Immunosuppressant",
        "Anakinra": "Biologic/Immunosuppressant",
        "cure Brentuximab I/V": "Biologic/Immunosuppressant",
        "cure Rithuximab I/V": "Biologic/Immunosuppressant",
        "Eculizumab (Soliris)": "Biologic/Immunosuppressant",
        "Soliris": "Biologic/Immunosuppressant",
        "eculizumab I / V": "Biologic/Immunosuppressant",
        "Leflunomid / Arava": "Biologic/Immunosuppressant",
        "etanercept": "Biologic/Immunosuppressant",
        "Hydroxychloroquin": "Biologic/Immunosuppressant",
        "Belatacept i.v.": "Biologic/Immunosuppressant",
        "belatacept (nulogix)": "Biologic/Immunosuppressant",
        "Etanerceptum": "Biologic/Immunosuppressant",
        "Rituximab IV": "Biologic/Immunosuppressant",
        "Etanercept injection": "Biologic/Immunosuppressant",
        "Anakinra (Kineret)": "Biologic/Immunosuppressant",
        "Infliximab": "Biologic/Immunosuppressant",
        "Soloris": "Biologic/Immunosuppressant",
        "Endoxon": "Biologic/Immunosuppressant",
        "Endoxan": "Biologic/Immunosuppressant",
        "Endoxan (Cyclophposphamide).": "Biologic/Immunosuppressant",
        "Cure IV / Ig": "Immunoglobulin Therapy (IVIG)",
        "ivIG": "Immunoglobulin Therapy (IVIG)",
        "I v / I g treatement": "Immunoglobulin Therapy (IVIG)",
        "IG/iv chronic humoral rejection": "Immunoglobulin Therapy (IVIG)",
        "Dexamethasone": "Corticosteroid",
        "DexamÃ©thasone": "Corticosteroid",
        "DexamÃ©thasone.": "Corticosteroid",
        "Glucocorticoid": "Corticosteroid",
        "Solumedrol IV": "Corticosteroid",
        "Fludrocortison": "Corticosteroid",
        "Spiricort": "Corticosteroid",
        "Spirocort": "Corticosteroid",
        "IV Solumedrol": "Corticosteroid",
        "Prednison": "Corticosteroid",
        "Hydrocartison": "Corticosteroid",
        "Florinef": "Corticosteroid",
        
        # --- Endocrine Agents ---
        "Semaglutid": "GLP-1 Receptor Agonist",
        "semaglutid": "GLP-1 Receptor Agonist",
        "Liraglutid": "GLP-1 Receptor Agonist",
        "Trulicity": "GLP-1 Receptor Agonist",
        "Ozempic": "GLP-1 Receptor Agonist",
        "Liraglutide": "GLP-1 Receptor Agonist",
        "Semaglutid s.c.": "GLP-1 Receptor Agonist",
        "liraglutid": "GLP-1 Receptor Agonist",
        "Semaglutid (Ozempic)": "GLP-1 Receptor Agonist",
        "Liraglutid (Victoza)": "GLP-1 Receptor Agonist",
        "Dulaglutide": "GLP-1 Receptor Agonist",
        "Dulaglutid (Trulicity)": "GLP-1 Receptor Agonist",
        "Victoza": "GLP-1 Receptor Agonist",
        "Glucagon-like-peptide 1": "GLP-1 Receptor Agonist",
        "Eltroxin": "Thyroid Hormone",
        "euthyrox": "Thyroid Hormone",
        "Levothyroxin-Natrium": "Thyroid Hormone",
        "Levothyroxin": "Thyroid Hormone",
        "Elthyroxin": "Thyroid Hormone",
        "Thyroxin": "Thyroid Hormone",
        "Euthyroxin": "Thyroid Hormone",
        "L-Thyroxin": "Thyroid Hormone",
        "levothyroxine": "Thyroid Hormone",
        "Jardiance": "SGLT2 Inhibitor",
        "Empagliflozine": "SGLT2 Inhibitor",
        "Methformin": "Biguanide",
        "Lantus": "Insulin",
        "Neomercazole": "Antithyroid Agent",
        "carbimazolum": "Antithyroid Agent",
        "Testoviron": "Hormonal Agent",
        
        # --- Gastrointestinal Agents ---
        "Pantoprazolum": "Proton Pump Inhibitor (PPI)",
        "Pantoprazol": "Proton Pump Inhibitor (PPI)",
        "Pantoprazol / Pantozol": "Proton Pump Inhibitor (PPI)",
        "Esomeprazolum": "Proton Pump Inhibitor (PPI)",
        "Pantozol": "Proton Pump Inhibitor (PPI)",
        "Esomeprazol / Nexium": "Proton Pump Inhibitor (PPI)",
        "Nexium": "Proton Pump Inhibitor (PPI)",
        "Pantoprazol / Pantazol": "Proton Pump Inhibitor (PPI)",
        "Esomeprazol": "Proton Pump Inhibitor (PPI)",
        "Pantoprazole": "Proton Pump Inhibitor (PPI)",
        "PPI": "Proton Pump Inhibitor (PPI)",
        "Pantozolum": "Proton Pump Inhibitor (PPI)",
        "Pantoprazole 40mg": "Proton Pump Inhibitor (PPI)",
        "Metoclopramidi/Paspertin": "Antiemetic",
        "Creon": "Pancreatic Enzymes",
        "Ranitidinum": "H2 Receptor Blocker",

        # --- Oncology Agents ---
        "Tamoxifen": "Hormonal Therapy (Oncology)",
        "Letrozol": "Hormonal Therapy (Oncology)",
        "Zoladex": "Hormonal Therapy (Oncology)",
        "Velcade": "Chemotherapy/Targeted Therapy",
        "Revlimid": "Chemotherapy/Targeted Therapy",
        "Mekinist": "Chemotherapy/Targeted Therapy",
        "Temodal (Cytostatic drug)": "Chemotherapy/Targeted Therapy",
        "cytostatic agents": "Chemotherapy/Targeted Therapy",
        "chemotherapy (Sorafenib)": "Chemotherapy/Targeted Therapy",
        "R-CHOP": "Chemotherapy/Targeted Therapy",
        
        # --- Hematologic Agents ---
        "Aranesp": "Erythropoiesis-Stimulating Agent (ESA)",
        "Darbepoetin / Aranesp": "Erythropoiesis-Stimulating Agent (ESA)",
        "Darbepoetin alpha/Aranesp": "Erythropoiesis-Stimulating Agent (ESA)",
        "Darbepoetin": "Erythropoiesis-Stimulating Agent (ESA)",
        "Darbepoetin alpha": "Erythropoiesis-Stimulating Agent (ESA)",
        "erythropoetin": "Erythropoiesis-Stimulating Agent (ESA)",
        "EPO (aranesp)": "Erythropoiesis-Stimulating Agent (ESA)",
        "Epoetunum beta": "Erythropoiesis-Stimulating Agent (ESA)",
        "Epoetum beta": "Erythropoiesis-Stimulating Agent (ESA)",
        "EPO": "Erythropoiesis-Stimulating Agent (ESA)",
        "Filgrastim": "Colony-Stimulating Factor",
        "Haemocomplettan": "Fibrinogen Concentrate",

        # --- Renal & Metabolic Agents ---
        "Mimpara": "Calcimimetic",
        "Cinacalcet": "Calcimimetic",
        "cinacalcet": "Calcimimetic",
        "Cinacalcet / Mimpara": "Calcimimetic",
        "Cinacalcet (Mimpara)": "Calcimimetic",
        "Mimpara (Cinacalcet)": "Calcimimetic",
        "Phoscaps": "Phosphate Binder",
        "Ca-Acetat": "Phosphate Binder",
        "Sevelamer / Renagel": "Phosphate Binder",
        "Allopurinolum": "Uric Acid Reducer",
        "Colchicine": "Anti-gout Agent",
        "Nephrotrans": "Bicarbonate Supplement",

        # --- Respiratory & Allergy Agents ---
        "Symbicort": "Inhaled Corticosteroid/Bronchodilator",
        "Spiriva": "Inhaled Corticosteroid/Bronchodilator",
        "Seretide": "Inhaled Corticosteroid/Bronchodilator",
        "Pulmicort": "Inhaled Corticosteroid/Bronchodilator",
        "Relvar": "Inhaled Corticosteroid/Bronchodilator",
        
        # --- Urological Agents ---
        "Proscar": "5-alpha-reductase Inhibitor",

        # --- Supplements & Minerals ---
        "Calcium D3": "Calcium/Vitamin D Supplement",
        "Calcimagon": "Calcium/Vitamin D Supplement",
        "Calcitriol": "Calcium/Vitamin D Supplement",
        "Colecalciferol": "Calcium/Vitamin D Supplement",
        "Vit. D": "Calcium/Vitamin D Supplement",
        "Rocaltrol": "Calcium/Vitamin D Supplement",
        "Vit. D3": "Calcium/Vitamin D Supplement",
        "Calcium": "Calcium/Vitamin D Supplement",
        "Calcium Sandoz D3": "Calcium/Vitamin D Supplement",
        
        ####################
        # ADDED EVEN LATER #
        ####################

        # Not Applicable / Unknown
        "other": "Other",

        # Medical/Therapeutic Procedure
        "AB0-IgG Adsorption": "Medical/Therapeutic Procedure",
        "ABO-Immunadsorption": "Medical/Therapeutic Procedure",
        "Plasmapheresis/PE": "Medical/Therapeutic Procedure",
        "Plasmaexchange": "Medical/Therapeutic Procedure",
        "Plasmapherisis 3 X / W": "Medical/Therapeutic Procedure",
        "IAD": "Medical/Therapeutic Procedure",
        "IADS": "Medical/Therapeutic Procedure",
        "AB0-Immunabsorption": "Medical/Therapeutic Procedure",
        "AB0-Immunadsorbtion": "Medical/Therapeutic Procedure",
        "Adsorption": "Medical/Therapeutic Procedure",
        "plasmapherese for initial disease": "Medical/Therapeutic Procedure",
        "recive 8 Plasmapheresis /PE": "Medical/Therapeutic Procedure",
        "Plasmapheresis /PE X 4": "Medical/Therapeutic Procedure",
        "plasmaphÃ©rÃ¨se X1": "Medical/Therapeutic Procedure",
        "Immunabsorbtion": "Medical/Therapeutic Procedure",
        "Glycosorb": "Medical/Therapeutic Procedure", # Part of immunoadsorption therapy
        "Photopherese": "Medical/Therapeutic Procedure",

        # --- Cardiovascular Agents ---
        "Physiotens (Moxonidinum)": "Centrally Acting Agent",
        "Physiotens": "Centrally Acting Agent",
        "Central Blocker": "Centrally Acting Agent",
        "central blocker (physiotens)": "Centrally Acting Agent",
        "central-blocker": "Centrally Acting Agent",
        "Central adrenergic (Physiotens)": "Centrally Acting Agent",
        "Moxonidin (Physiotens)": "Centrally Acting Agent",
        "Moxonidin": "Centrally Acting Agent",
        "Physiotens (moxonidine)": "Centrally Acting Agent",
        "Catapressan": "Centrally Acting Agent",
        "Clonidin": "Centrally Acting Agent",
        "clonidin": "Centrally Acting Agent",
        "Clonidini hydrochloridum": "Centrally Acting Agent",
        "Methyldopa": "Centrally Acting Agent",
        "central blocker ( catapressan )": "Centrally Acting Agent", # Catapresan is central, not alpha-blocker
        "alpha-blocker (catapressan )": "Centrally Acting Agent", # Mis-categorized in source text
        "alpha blocker ( catapressan )": "Centrally Acting Agent",
        "alpha blocker ( Catapressan )": "Centrally Acting Agent",
        "Thiazid": "Diuretic",
        "Thiazide": "Diuretic",
        "Thiazideddiusetic": "Diuretic", # Typo
        "Metolazonum": "Diuretic",
        "Metalozonum": "Diuretic",
        "Metolazon": "Diuretic",
        "diuretic (metolazone)": "Diuretic",
        "Spironolactonum": "Diuretic",
        "Spironolacton": "Diuretic",
        "Aldactone": "Diuretic",
        "aldactone": "Diuretic",
        "diuretic ( aldactone )": "Diuretic",
        "diuretics (aldactone )": "Diuretic",
        "diuretic (spironolactone)": "Diuretic",
        "Eplerenon": "Diuretic",
        "diuretic(inspra)": "Diuretic",
        "Furosemidum": "Diuretic",
        "Furosemid / Lasix": "Diuretic",
        "Furosemid": "Diuretic",
        "Forosemic / Lasix": "Diuretic",
        "diuretic (lasix)": "Diuretic",
        "diuretic (torem )": "Diuretic",
        "diuretic (torasemid)": "Diuretic",
        "Toremisid": "Diuretic", # Typo
        "Torasemid/Torem": "Diuretic",
        "Toresemid": "Diuretic",
        "Torasamid": "Diuretic",
        "Hydrochlorothiazid": "Diuretic",
        "Hydrochlorothiazidum": "Diuretic",
        "Hydrochlorothiazide": "Diuretic",
        "Comilorid": "Diuretic",
        "Chlortalidon": "Diuretic",
        "Chlorthalidon": "Diuretic",
        "Diuretics": "Diuretic",
        "Diuretic (torem)": "Diuretic",
        "Diuretic ( esidrex)": "Diuretic",
        "Diuretikum": "Diuretic",
        "Diuretika": "Diuretic",
        "Diuretic ( + )": "Diuretic",
        "Torasemid / Torsis": "Diuretic",
        "Diuretiz": "Diuretic", # Typo
        "Diuretics.": "Diuretic",
        "TorasÃ©mide": "Diuretic",
        "Cordarone (Amiodarone)": "Antiarrhythmic",
        "cordarone": "Antiarrhythmic",
        "Amiodarone": "Antiarrhythmic",
        "amiodarone": "Antiarrhythmic",
        "Amiodaron (Cordarone)": "Antiarrhythmic",
        "Amiodaroni hydrochloridum": "Antiarrhythmic",
        "Amidarone": "Antiarrhythmic",
        "Amidaron": "Antiarrhythmic",
        "Amiodaron / Cordarone": "Antiarrhythmic",
        "Digoxin": "Antiarrhythmic",
        "Digoxine": "Antiarrhythmic",
        "Digoxinum": "Antiarrhythmic",
        "alpha-Blocker": "Alpha-blocker",
        "Alpha-blocker": "Alpha-blocker",
        "alpha-blocker (cardura)": "Alpha-blocker",
        "alpha-blocker (cardura )": "Alpha-blocker",
        "Alpha Blocker": "Alpha-blocker",
        "Cardura (Doxazosin mesylate)": "Alpha-blocker",
        "Doxazosine (Cardura)": "Alpha-blocker",
        "Cardura (doxazosine)": "Alpha-blocker",
        "Tamsulosini Hydrochloridum": "Alpha-blocker",
        "Tamsulosini": "Alpha-blocker",
        "Tamsulosin / Pradif": "Alpha-blocker",
        "Tamsulosini hydrochlorid": "Alpha-blocker",
        "Chinazoline": "Alpha-blocker",
        "Duodart": "Alpha-blocker",
        "Duodart (DutastÃ©ride + tamsulosine)": "Alpha-blocker",
        "Doxazosin / Cardura": "Alpha-blocker",
        "Doxazosinum/Cardura": "Alpha-blocker",
        "Doxazosin (alpha-blocker)": "Alpha-blocker",
        "Cardura (Alphablocker)": "Alpha-blocker",
        "Cordura": "Alpha-blocker",
        "tamsulosine (prodif)": "Alpha-blocker",
        "Pradif T (Tamsulosine": "Alpha-blocker",
        "Alphabocker": "Alpha-blocker",
        "ARB + diuretic (erdabyclore)": "ARB/Diuretic Combo",
        "Labetalol": "Beta-blocker",
        "Atenololum": "Beta-blocker",
        "Nitroderm": "Vasodilator",
        "nitroderm patch": "Vasodilator",
        "Nitroglycerin": "Vasodilator",
        "Dancor": "Vasodilator",
        "Molsidomin": "Vasodilator",
        "Nitro": "Vasodilator",
        "Antihypertonikum / Loniten": "Vasodilator",
        "Corvaton": "Vasodilator",
        "Minoxidilum": "Vasodilator",
        "Opsumit": "Vasodilator",
        "Nitroderm patch": "Vasodilator",
        "Nitroglycerin (Nitroderm)": "Vasodilator",
        "Nicorandil (Dancor)": "Vasodilator",
        "Corvaton retard": "Vasodilator",
        "Nitroderm TTS": "Vasodilator",
        "Loniten (minoxidil)": "Vasodilator",
        "Minoxidil": "Vasodilator",
        "Loniten": "Vasodilator",
        "Noradrenalin": "Vasopressor",
        "Ranolazin": "Anti-anginal",
        "Adalat": "CCB",
        "Amloclipine": "CCB",
        "Rivaroxaban": "Anticoagulant",
        "Sintrom": "Anticoagulant",
        "Efient": "Antiplatelet",
        "Clopidogrel": "Antiplatelet",

        # --- Neurology & Psychiatry Agents ---
        "Lyrica": "Anticonvulsant",
        "Pregabalin": "Anticonvulsant",
        "Keppra": "Anticonvulsant",
        "Levetiracetam": "Anticonvulsant",
        "Antiepileptika": "Anticonvulsant",
        "Depakine": "Anticonvulsant",
        "Lamotrigin / Lamictal": "Anticonvulsant",
        "Vimpat": "Anticonvulsant",
        "Carbamazepin": "Anticonvulsant",
        "Lamictal": "Anticonvulsant",
        "Levitriactam": "Anticonvulsant", # Typo
        "Levetiracetam (keppra)": "Anticonvulsant",
        "Lamotrigin": "Anticonvulsant",
        "Aphenylbarbit": "Anticonvulsant",
        "Gabapenta": "Anticonvulsant", # Typo
        "Neurontin": "Anticonvulsant",
        "Cipralex": "Antidepressant",
        "Citalopram": "Antidepressant",
        "Escitalopram": "Antidepressant",
        "Sertralin": "Antidepressant",
        "Antidepressivum": "Antidepressant",
        "Mirtazapinum": "Antidepressant",
        "Remeron": "Antidepressant",
        "Cymbalta": "Antidepressant",
        "Trittico": "Antidepressant",
        "Saroten": "Antidepressant",
        "Antidepressiva": "Antidepressant",
        "Mirtazapinum / Remeron": "Antidepressant",
        "Citalopramum": "Antidepressant",
        "citalopram": "Antidepressant",
        "Citalopranum": "Antidepressant",
        "Trazodon": "Antidepressant",
        "Antidepressive drug": "Antidepressant",
        "Deanxit": "Antidepressant",
        "Trimipramin": "Antidepressant",
        "escitalopram": "Antidepressant",
        "Mirtazepin": "Antidepressant",
        "Madopar": "Antiparkinson Agent",
        "Levodopum": "Antiparkinson Agent",
        "Pramipexol": "Antiparkinson Agent",
        "Sifrol": "Antiparkinson Agent",
        "Levodopa (Antiparkinson)": "Antiparkinson Agent",
        "Levodopa": "Antiparkinson Agent",
        "Alprazolam": "Anxiolytic/Sedative",
        "Lorazepamum": "Anxiolytic/Sedative",
        "Temesta": "Anxiolytic/Sedative",
        "Zolpidemtartrat": "Anxiolytic/Sedative",
        "Oxazepam": "Anxiolytic/Sedative",
        "Zolpidemi tartras": "Anxiolytic/Sedative",
        "Zolpidem / Stilnox": "Anxiolytic/Sedative",
        "Oxazepam / Seresta": "Anxiolytic/Sedative",
        "Zolpidem": "Anxiolytic/Sedative",
        "Urbanyl": "Anxiolytic/Sedative",
        "Zolpidem/Stilnox": "Anxiolytic/Sedative",
        "Quetiapin": "Antipsychotic",
        "Quetiapin / Seroquel": "Antipsychotic",
        "Olanzapin": "Antipsychotic",
        "Neuroleptika": "Antipsychotic",
        "Quetiapinum": "Antipsychotic",
        "Haloperidol": "Antipsychotic",
        "Seroquel": "Antipsychotic",
        "Abilify": "Antipsychotic",
        "Pipamperon / Dipiperon": "Antipsychotic",
        "Pipamperonum/Dipiperon": "Antipsychotic",
        "Paracetamolum": "Analgesic (Non-Opioid)",
        "Paracetamolum / Dafalgan": "Analgesic (Non-Opioid)",
        "Mephanol": "Analgesic (Non-Opioid)",
        "Pyrazolonderivat/Novalgin": "Analgesic (Non-Opioid)",
        "Pyrazolonderivat / Novalgin": "Analgesic (Non-Opioid)",
        "Pyrazolonderivat": "Analgesic (Non-Opioid)",
        "Nopilforte": "Analgesic (Non-Opioid)",
        "Methadon": "Opioid Analgesic",
        "Morphin": "Opioid Analgesic",
        "Temgesic": "Opioid Analgesic",
        "Betaserc": "Anti-vertigo Agent",
        "Naloxon": "Opioid Antagonist",

        # --- Antimicrobials ---
        "Monuril (Fosfomycine)": "Antibiotic",
        "Uvamine R (NitrofurantoÃ¯ne)": "Antibiotic",
        "Vancomycine (Vancocin)": "Antibiotic",
        "Vancomycin (Vancocin)": "Antibiotic",
        "Isoniazid": "Antibiotic",
        "Uvamine Retard (Nitrofurantoin)": "Antibiotic",
        "azithromycin": "Antibiotic",
        "Stabicilline <phÃ©noxymÃ©thylpÃ©nicilline (pÃ©nicilline V)>": "Antibiotic",
        "Azithromycin": "Antibiotic",
        "Ospen": "Antibiotic",
        "Polymyxine B + NÃ©omycine": "Antibiotic",
        "monuril": "Antibiotic",
        "Fosfomycin": "Antibiotic",
        "Monuril": "Antibiotic",
        "Isoniazide": "Antibiotic",
        "Gentamycine": "Antibiotic",
        "Polymixin E": "Antibiotic",
        "fosfomycine (monuril)": "Antibiotic",
        "Dapson/Pyrimethamin": "Antibiotic",
        "Clindamycine.": "Antibiotic",
        "Dapson": "Antibiotic",
        "Ciprofloxacin": "Antibiotic",
        "Vancomycin": "Antibiotic",
        "Penicilline benzathine": "Antibiotic",
        "nitrofurantoine": "Antibiotic",
        "Vancomycin (Vancocine)": "Antibiotic",
        "Azithromycine (Zithromax)": "Antibiotic",
        "Furandantine R (NitrofurantoÃ¯ne)": "Antibiotic",
        "Uvamine": "Antibiotic",
        "Zithromax": "Antibiotic",
        "fosfomycine": "Antibiotic",
        "Moxifloxacin": "Antibiotic",
        "Acitromycin": "Antibiotic", # Typo
        "Ethambutoli hydrochloridum": "Antibiotic",
        "Isonazid": "Antibiotic",
        "Isoniazidum": "Antibiotic",
        "Ospen (Phenoxymethylpenicillinum kalicum)": "Antibiotic",
        "uvamine": "Antibiotic",
        "Fosmomycin / Monuril": "Antibiotic",
        "Stabilline": "Antibiotic",
        "vancomycin": "Antibiotic",
        "B-Lactame": "Antibiotic",
        "meropenem": "Antibiotic",
        "ceftriaxone": "Antibiotic",
        "phosphomycine": "Antibiotic",
        "colistin": "Antibiotic",
        "macrolide": "Antibiotic",
        "ceftazidine": "Antibiotic",
        "Bactrim": "Antibiotic",
        "Isoniazide for Tuberculosis": "Antibiotic",
        "Maxifloxacine for Tuberculosis": "Antibiotic",
        "Nifabutine for Tuberculosis": "Antibiotic",
        "Co Amoxicillin": "Antibiotic",
        "azitromycine": "Antibiotic",
        "Colistimethat / Colistin": "Antibiotic",
        "Azithromycin / Zithromax": "Antibiotic",
        "Mycostatine (Nystatine)": "Antifungal",
        "Nystatinum/Multilind": "Antifungal",
        "Nystatinum / Multilind": "Antifungal",
        "Amphotericin": "Antifungal",
        "Mycostatin": "Antifungal",
        "Amphotericin B": "Antifungal",
        "Nystatinum/7Multilind": "Antifungal",
        "Lamivudine": "Antiviral",
        "Tivicay": "Antiviral",
        "Tamiflu": "Antiviral",
        "Baraclude": "Antiviral",
        "Entecavir": "Antiviral",
        "Tenofovir": "Antiviral",
        "Ziagen": "Antiviral",
        "Triumeq": "Antiviral",
        "Sofosbuvir": "Antiviral",
        "Ribavirine": "Antiviral",
        "Vemlidy": "Antiviral",
        "Descovy": "Antiviral",
        "3TC": "Antiviral",
        "antiviral therapy": "Antiviral",
        "Emtriva": "Antiviral",
        "Kivexa": "Antiviral",
        "Stocrin": "Antiviral",
        "TTT HIV": "Antiviral",
        "viread": "Antiviral",
        "Zeffix": "Antiviral",
        "Asefovirdipivoxil": "Antiviral",
        "entÃ©cavir": "Antiviral",
        "baraclud": "Antiviral",
        "Famvir (famciclovir)": "Antiviral",
        "Dovato ( lamivudine + dolutÃ©gravir)": "Antiviral",
        "Trinmeq": "Antiviral",
        "Pentamidine": "Antiparasitic",

        # --- Immunomodulators & Anti-inflammatories ---
        "Etanercept (Enbrel)": "Biologic/Immunosuppressant",
        "Enbrel": "Biologic/Immunosuppressant",
        "Etanercept": "Biologic/Immunosuppressant",
        "Enbrel (etanercept)": "Biologic/Immunosuppressant",
        "Leflunomid": "Biologic/Immunosuppressant",
        "Thymoglobulin": "Biologic/Immunosuppressant",
        "Thymoglobuline": "Biologic/Immunosuppressant",
        "ATG": "Biologic/Immunosuppressant",
        "Anakinra": "Biologic/Immunosuppressant",
        "cure Brentuximab I/V": "Biologic/Immunosuppressant",
        "cure Rithuximab I/V": "Biologic/Immunosuppressant",
        "Eculizumab (Soliris)": "Biologic/Immunosuppressant",
        "Soliris": "Biologic/Immunosuppressant",
        "Leflunomid / Arava": "Biologic/Immunosuppressant",
        "etanercept": "Biologic/Immunosuppressant",
        "Hydroxychloroquin": "Biologic/Immunosuppressant",
        "Belatacept (Nulojix)": "Biologic/Immunosuppressant",
        "belatacept (nulogix)": "Biologic/Immunosuppressant",
        "eculizumab I / V": "Biologic/Immunosuppressant",
        "Etanerceptum": "Biologic/Immunosuppressant",
        "Rituximab IV": "Biologic/Immunosuppressant",
        "Etanercept injection": "Biologic/Immunosuppressant",
        "Anakinra (Kineret)": "Biologic/Immunosuppressant",
        "Infliximab": "Biologic/Immunosuppressant",
        "Soloris": "Biologic/Immunosuppressant", # Typo
        "Endoxon": "Biologic/Immunosuppressant", # Typo
        "Kineret": "Biologic/Immunosuppressant",
        "Ethanercept": "Biologic/Immunosuppressant", # Typo
        "Advagraf": "Biologic/Immunosuppressant",
        "Rituximab for initial desease": "Biologic/Immunosuppressant",
        "eculizumab": "Biologic/Immunosuppressant",
        "Betalacept": "Biologic/Immunosuppressant", # Typo
        "MabThera (Cytostatic drug)": "Biologic/Immunosuppressant",
        "Mercaptopurin": "Biologic/Immunosuppressant",
        "Tocilizumab": "Biologic/Immunosuppressant",
        "toculizumab": "Biologic/Immunosuppressant",
        "Endoxan": "Biologic/Immunosuppressant",
        "Endoxan (Cyclophposphamide).": "Biologic/Immunosuppressant",
        "Plaquenil": "Biologic/Immunosuppressant",
        "Cyclosporin": "Biologic/Immunosuppressant",
        "cure rituximab": "Biologic/Immunosuppressant",
        "IV / IG Cure": "Immunoglobulin Therapy (IVIG)",
        "ivIG": "Immunoglobulin Therapy (IVIG)",
        "I v / I g treatement": "Immunoglobulin Therapy (IVIG)",
        "IG/iv chronic humoral rejection": "Immunoglobulin Therapy (IVIG)",
        "Cure I V / I G": "Immunoglobulin Therapy (IVIG)",
        "IV / IG": "Immunoglobulin Therapy (IVIG)",
        "IG iv": "Immunoglobulin Therapy (IVIG)",
        "Kiovig IV / IG": "Immunoglobulin Therapy (IVIG)",
        "I V / I G": "Immunoglobulin Therapy (IVIG)",
        "Ivig": "Immunoglobulin Therapy (IVIG)",
        "IG / IV": "Immunoglobulin Therapy (IVIG)",
        "IG / iv X1": "Immunoglobulin Therapy (IVIG)",
        "IV /  IG for initial disaese": "Immunoglobulin Therapy (IVIG)",
        "KIOVIG  I / V": "Immunoglobulin Therapy (IVIG)",
        "Dexamethasone": "Corticosteroid",
        "DexamÃ©thasone": "Corticosteroid",
        "DexamÃ©thasone.": "Corticosteroid",
        "Glucocorticoid": "Corticosteroid",
        "Solumedrol IV": "Corticosteroid",
        "Fludrocortison": "Corticosteroid",
        "Spiricort": "Corticosteroid",
        "Spirocort": "Corticosteroid",
        "IV Solumedrol": "Corticosteroid",
        "Prednison": "Corticosteroid",
        "Hydrocartison": "Corticosteroid", # Typo
        "Florinef": "Corticosteroid",
        "Methyl-prednisone I / V": "Corticosteroid",

        # --- Endocrine Agents ---
        "Semaglutid": "GLP-1 Receptor Agonist",
        "semaglutid": "GLP-1 Receptor Agonist",
        "Liraglutid": "GLP-1 Receptor Agonist",
        "Trulicity": "GLP-1 Receptor Agonist",
        "Ozempic": "GLP-1 Receptor Agonist",
        "Liraglutide": "GLP-1 Receptor Agonist",
        "Semaglutid s.c.": "GLP-1 Receptor Agonist",
        "liraglutid": "GLP-1 Receptor Agonist",
        "Semaglutid (Ozempic)": "GLP-1 Receptor Agonist",
        "Liraglutid (Victoza)": "GLP-1 Receptor Agonist",
        "Dulaglutide": "GLP-1 Receptor Agonist",
        "Dulaglutid (Trulicity)": "GLP-1 Receptor Agonist",
        "Victoza": "GLP-1 Receptor Agonist",
        "Glucagon-like-peptide 1": "GLP-1 Receptor Agonist",
        "Eltroxin": "Thyroid Hormone",
        "euthyrox": "Thyroid Hormone",
        "Levothyroxin-Natrium": "Thyroid Hormone",
        "Levothyroxin": "Thyroid Hormone",
        "Elthyroxin": "Thyroid Hormone",
        "Thyroxin": "Thyroid Hormone",
        "Euthyroxin": "Thyroid Hormone",
        "L-Thyroxin": "Thyroid Hormone",
        "levothyroxine": "Thyroid Hormone",
        "Jardiance": "SGLT2 Inhibitor",
        "Empagliflozine": "SGLT2 Inhibitor",
        "Methformin": "Biguanide",
        "Lantus": "Insulin",
        "Neomercazole": "Antithyroid Agent",
        "neo-mercazole": "Antithyroid Agent",
        "carbimazolum": "Antithyroid Agent",
        "Neo-Mercazole": "Antithyroid Agent",
        "Testoviron": "Hormonal Agent",
        "Adrenocorticotropin": "Hormonal Agent",
        
        # --- Gastrointestinal Agents ---
        "Pantoprazolum": "Proton Pump Inhibitor (PPI)",
        "Pantoprazol": "Proton Pump Inhibitor (PPI)",
        "Pantoprazol / Pantozol": "Proton Pump Inhibitor (PPI)",
        "Esomeprazolum": "Proton Pump Inhibitor (PPI)",
        "Pantozol": "Proton Pump Inhibitor (PPI)",
        "Esomeprazol / Nexium": "Proton Pump Inhibitor (PPI)",
        "Nexium": "Proton Pump Inhibitor (PPI)",
        "Pantoprazol / Pantazol": "Proton Pump Inhibitor (PPI)",
        "Esomeprazol": "Proton Pump Inhibitor (PPI)",
        "Pantoprazole": "Proton Pump Inhibitor (PPI)",
        "PPI": "Proton Pump Inhibitor (PPI)",
        "Pantozolum": "Proton Pump Inhibitor (PPI)",
        "Pantoprazole 40mg": "Proton Pump Inhibitor (PPI)",
        "Pantopraxzo / Pantozol": "Proton Pump Inhibitor (PPI)",
        "Metoclopramidi/Paspertin": "Antiemetic",
        "Domperidon / Motilium": "Antiemetic",
        "Metoclopramidi hydrochlorid": "Antiemetic",
        "Metoclopramid": "Antiemetic",
        "Creon": "Pancreatic Enzymes",
        "Pancreatis pulvis": "Pancreatic Enzymes",
        "Pankreatin": "Pancreatic Enzymes",
        "Ranitidinum": "H2 Receptor Blocker",
        "Natriumpicosulfat/Laxoberon": "Laxative",
        "Natriumpicosulfat / Laxoberon": "Laxative",
        "Macrogolum/Transipeg": "Laxative",
        "Natriumpicosulfat": "Laxative",
        "Simeticon": "Antiflatulent",
        "Imodium": "Antidiarrheal",

        # --- Oncology Agents ---
        "Tamoxifen": "Hormonal Therapy (Oncology)",
        "Letrozol": "Hormonal Therapy (Oncology)",
        "Zoladex": "Hormonal Therapy (Oncology)",
        "Velcade": "Chemotherapy/Targeted Therapy",
        "Revlimid": "Chemotherapy/Targeted Therapy",
        "Mekinist": "Chemotherapy/Targeted Therapy",
        "Temodal (Cytostatic drug)": "Chemotherapy/Targeted Therapy",
        "cytostatic agents": "Chemotherapy/Targeted Therapy",
        "chemotherapy (Sorafenib)": "Chemotherapy/Targeted Therapy",
        "R-CHOP": "Chemotherapy/Targeted Therapy",
        
        # --- Hematologic Agents ---
        "Aranesp": "Erythropoiesis-Stimulating Agent (ESA)",
        "Darbepoetin / Aranesp": "Erythropoiesis-Stimulating Agent (ESA)",
        "Darbepoetin alpha/Aranesp": "Erythropoiesis-Stimulating Agent (ESA)",
        "Darbepoetin": "Erythropoiesis-Stimulating Agent (ESA)",
        "Darbepoetin alpha": "Erythropoiesis-Stimulating Agent (ESA)",
        "erythropoetin": "Erythropoiesis-Stimulating Agent (ESA)",
        "EPO (aranesp)": "Erythropoiesis-Stimulating Agent (ESA)",
        "Epoetunum beta": "Erythropoiesis-Stimulating Agent (ESA)",
        "Epoetum beta": "Erythropoiesis-Stimulating Agent (ESA)",
        "Erythropoetin": "Erythropoiesis-Stimulating Agent (ESA)",
        "Darbopoetin alpha": "Erythropoiesis-Stimulating Agent (ESA)",
        "EPO": "Erythropoiesis-Stimulating Agent (ESA)",
        "Filgrastim": "Colony-Stimulating Factor",
        "Haemocomplettan": "Fibrinogen Concentrate",
        "Eisen / Venofer": "Iron Supplement",
        "Ferinject 500mg": "Iron Supplement",
        "Perfusion Ferinject": "Iron Supplement",
        
        # --- Renal & Metabolic Agents ---
        "Mimpara": "Calcimimetic",
        "Cinacalcet": "Calcimimetic",
        "cinacalcet": "Calcimimetic",
        "Cinacalcet / Mimpara": "Calcimimetic",
        "Cinacalcet (Mimpara)": "Calcimimetic",
        "Cinacalcat/Mimpara": "Calcimimetic",
        "Phoscaps": "Phosphate Binder",
        "Ca-Acetat": "Phosphate Binder",
        "Sevelamer / Renagel": "Phosphate Binder",
        "Sevelamer/Renagel": "Phosphate Binder",
        "Lanthan / Fosrenol": "Phosphate Binder",
        "Sevelamer": "Phosphate Binder",
        "Phosphat": "Phosphate Binder",
        "Phosphat / Phoscap": "Phosphate Binder",
        "Calcium Acetat": "Phosphate Binder",
        "Calcium Azetat": "Phosphate Binder",
        "Renagel": "Phosphate Binder",
        "Calciumacetat 400mg": "Phosphate Binder",
        "Calcium / Calcium-Acetat": "Phosphate Binder",
        "Allopurinolum": "Uric Acid Reducer",
        "Zyloric": "Uric Acid Reducer",
        "allopurinol": "Uric Acid Reducer",
        "Febuxostat": "Uric Acid Reducer",
        "Zyloric (Allopurinol)": "Uric Acid Reducer",
        "ADENURIC": "Uric Acid Reducer",
        "Allopur": "Uric Acid Reducer",
        "FASTURTEC": "Uric Acid Reducer",
        "AdenÃ¼ric": "Uric Acid Reducer",
        "Urikostatika": "Uric Acid Reducer",
        "Colchicine": "Anti-gout Agent",
        "Colchizin": "Anti-gout Agent",
        "Cholchicin": "Anti-gout Agent",
        "Nephrotrans": "Bicarbonate Supplement",
        "Na-hydrogencarbonat / Nephrotrans": "Bicarbonate Supplement",
        "Natriumbicarbonat 1.2g": "Bicarbonate Supplement",
        "oral Bicarbonate: Nephrotrane": "Bicarbonate Supplement",
        "Natriumbicarbonat": "Bicarbonate Supplement",

        # --- Respiratory & Allergy Agents ---
        "Symbicort": "Inhaled Corticosteroid/Bronchodilator",
        "Spiriva": "Inhaled Corticosteroid/Bronchodilator",
        "Seretide": "Inhaled Corticosteroid/Bronchodilator",
        "Pulmicort": "Inhaled Corticosteroid/Bronchodilator",
        "Relvar": "Inhaled Corticosteroid/Bronchodilator",
        "Seritide": "Inhaled Corticosteroid/Bronchodilator", # Typo
        "Tiotropium": "Inhaled Corticosteroid/Bronchodilator",
        "Formoterol": "Inhaled Corticosteroid/Bronchodilator",
        "Atrovacant Spray": "Inhaled Corticosteroid/Bronchodilator",
        "SPIOLTO": "Inhaled Corticosteroid/Bronchodilator",
        "Dospir": "Inhaled Corticosteroid/Bronchodilator",
        "Antiasthmatikum": "Inhaled Corticosteroid/Bronchodilator",
        
        # --- Urological Agents ---
        "Proscar": "5-alpha-reductase Inhibitor",
        "Avodart": "5-alpha-reductase Inhibitor",
        "Bethanechol": "Cholinergic agonist",

        # --- Lipid-Lowering Agents ---
        "Ezetimib": "Cholesterol Absorption Inhibitor",
        "Ezetrol": "Cholesterol Absorption Inhibitor",
        "Ezetimibe": "Cholesterol Absorption Inhibitor",
        "Fluvastatinum": "Statin",
        "Atorvastatin": "Statin",

        # --- Supplements & Minerals ---
        "Prolia": "Bone Density Agent",
        "Denosumab (Prolia)": "Bone Density Agent",
        "Bisphosphonat": "Bone Density Agent",
        "Fosomab (Fosomax)": "Bone Density Agent",
        "Acidum ibandronicum ut natrii ibandronas hydricus / Bonviva": "Bone Density Agent",
        "Denosumab": "Bone Density Agent",
        "Calcium D3": "Calcium/Vitamin D Supplement",
        "Calcimagon": "Calcium/Vitamin D Supplement",
        "Calcitriol": "Calcium/Vitamin D Supplement",
        "Colecalciferol": "Calcium/Vitamin D Supplement",
        "Vit. D": "Calcium/Vitamin D Supplement",
        "Rocaltrol": "Calcium/Vitamin D Supplement",
        "Vit. D3": "Calcium/Vitamin D Supplement",
        "Calcium": "Calcium/Vitamin D Supplement",
        "Calcium Sandoz D3": "Calcium/Vitamin D Supplement",
        "Calcium Sandoz": "Calcium/Vitamin D Supplement",
        "Synthetic calcitrol": "Calcium/Vitamin D Supplement",
        "Synth. Calcitriol": "Calcium/Vitamin D Supplement",
        "Vit D": "Calcium/Vitamin D Supplement",
        "Vi-De": "Calcium/Vitamin D Supplement",
        "Calcitriol Vit D": "Calcium/Vitamin D Supplement",
        "Calcium Sandoz D3f": "Calcium/Vitamin D Supplement",
        "Synthetisches Calcitriol": "Calcium/Vitamin D Supplement",
        "Synth Calcitriol": "Calcium/Vitamin D Supplement",
        "Calcitril": "Calcium/Vitamin D Supplement", # Typo
        "Colecalciferol (Vit.D)": "Calcium/Vitamin D Supplement",
        "Paracalcitrol": "Calcium/Vitamin D Supplement",
        "Calcitiol / Rocaltrol": "Calcium/Vitamin D Supplement",
        "Calcitiol": "Calcium/Vitamin D Supplement",
        "Vit. D 3": "Calcium/Vitamin D Supplement",
        "Condrosulf": "Joint Health Supplement",
        "Phytomenadion/Konakion": "Vitamin Supplement (K)",
        "Phytomenadion": "Vitamin Supplement (K)",
        "Vit. B12": "Vitamin Supplement (B12)",
        "Acide Falique": "Vitamin Supplement (B9)",
        
        # --- Miscellaneous ---
        "Acicutan": "Dermatological Agent",
        "Aldara Creme (Imiquimod)": "Dermatological Agent",
        "Natriumthiosulfat": "Other",

        #########################
        # ADDED_EVEN EVEN LATER #
        #########################

        # --- Antimicrobials ---
        "Monuril (Fosfomycine) Alternatively.": "Antibiotic",
        "Fosfomycine (Monuril)": "Antibiotic",
        "Monuril (Fosfomycine) en alternance avec Uvamine R (NitrofurantoÃ¯ne)": "Antibiotic",
        "Uvamine R (NitrofurantoÃ¯ne) Alternatively.": "Antibiotic",
        "Monuril (fosfomycine)": "Antibiotic",
        "Uvamin R": "Antibiotic",
        "Fosfomycine": "Antibiotic",
        "phosphomycine": "Antibiotic", # Typo
        "Nitrofurantoine": "Antibiotic",
        "meronem": "Antibiotic", # Meropenem
        "ceftriaxone": "Antibiotic",
        "Bactrim": "Antibiotic",
        "azitromycine": "Antibiotic",
        "Azithromycine": "Antibiotic",
        "Azithromycin / Zithromax": "Antibiotic",
        "macrolide": "Antibiotic",
        "Polymyxin E": "Antibiotic",
        "Colistimethat / Colistin": "Antibiotic",
        "colistin": "Antibiotic",
        "vancomycine": "Antibiotic",
        "Co Amoxicillin": "Antibiotic",
        "Maxifloxacine for Tuberculosis": "Antibiotic", # Moxifloxacin
        "Isoniazide for Tuberculosis": "Antibiotic",
        "Nifabutine for Tuberculosis": "Antibiotic", # Rifabutin
        "Rifabutine": "Antibiotic",
        "Pyrimethamin": "Antiparasitic",
        "Pentamidin": "Antiparasitic",

        # --- Immunomodulators & Anti-inflammatories ---
        "Hepatect": "Immunoglobulin Therapy (IVIG)",
        "IV / IG Cure": "Immunoglobulin Therapy (IVIG)",
        "Kiovig IV / IG": "Immunoglobulin Therapy (IVIG)",
        "Cure I V / I G": "Immunoglobulin Therapy (IVIG)",
        "IV / IG": "Immunoglobulin Therapy (IVIG)",
        "cure IG/iv": "Immunoglobulin Therapy (IVIG)",
        "I V / I G": "Immunoglobulin Therapy (IVIG)",
        "IVig": "Immunoglobulin Therapy (IVIG)",
        "IG iv": "Immunoglobulin Therapy (IVIG)",
        "IV IG": "Immunoglobulin Therapy (IVIG)",
        "KIOVIG I / V": "Immunoglobulin Therapy (IVIG)",
        "cure de kiovig I/V": "Immunoglobulin Therapy (IVIG)",
        "cure de Kiovig I / V": "Immunoglobulin Therapy (IVIG)",
        "Tocilizumab": "Biologic/Immunosuppressant",
        "toculizumab": "Biologic/Immunosuppressant", # Typo
        "Plaquenil": "Biologic/Immunosuppressant",
        "Enbrel (Etanercept)": "Biologic/Immunosuppressant",
        "Soliris (Eculizumab) 5 times (once a week)": "Biologic/Immunosuppressant",
        "belatacept I / V": "Biologic/Immunosuppressant",
        "Thymoglobulines.": "Biologic/Immunosuppressant",
        "Cyclosporin": "Biologic/Immunosuppressant",
        "cure rituximab": "Biologic/Immunosuppressant",
        "Cure Rituximab X 4": "Biologic/Immunosuppressant",
        "Rituximab I/V": "Biologic/Immunosuppressant",
        "Eculizimab": "Biologic/Immunosuppressant",
        "Dexamethasone": "Corticosteroid",
        "Fludrocortison": "Corticosteroid",

        # --- Endocrine Agents ---
        "Eltroxine": "Thyroid Hormone",
        "elthroxine": "Thyroid Hormone", # Typo
        "Liraglutide": "GLP-1 Receptor Agonist",

        # --- Cardiovascular Agents ---
        "diuretic ( torem )": "Diuretic",
        "Esidrex": "Diuretic",
        "Adrenocorticotropin": "Hormonal Agent", # Also used for diagnostics
        "Ivabradine (Procoralan)": "Heart Rate Reducer (If Channel Inhibitor)",
        "Cardura (Alphablocker)": "Alpha-blocker",
        "alpha blocker ( Cardura )": "Alpha-blocker",
        
        # --- Renal & Metabolic Agents ---
        "Calcitrol": "Calcium/Vitamin D Supplement",
        "Calcimagon D3": "Calcium/Vitamin D Supplement",
        "Synt. Calcitriol": "Calcium/Vitamin D Supplement",
        "Vitamin D": "Calcium/Vitamin D Supplement",
        "Synthetic calcitriol": "Calcium/Vitamin D Supplement",
        "Calcium Sandoz": "Calcium/Vitamin D Supplement",
        "Natrii hydrocarbonas/Nephrotrans": "Bicarbonate Supplement",
        "Magnesiocard": "Electrolyte Supplement",
        "Kalium": "Electrolyte Supplement",
        "Phosphat / Phoscaps": "Phosphate Binder",
        "Zyloric": "Uric Acid Reducer",
        "allopurinol": "Uric Acid Reducer",
        
        # --- Hematologic Agents ---
        "Darbepoetin alpha / Aranesp": "Erythropoiesis-Stimulating Agent (ESA)",
        "aranesp": "Erythropoiesis-Stimulating Agent (ESA)",
        "Ferrum/Venofer": "Iron Supplement",
        
        # --- Gastrointestinal Agents ---
        "Esomeprazolum / Nexium": "Proton Pump Inhibitor (PPI)",
        "Xifaxan (Rifaximine)": "Antibiotic", # Also GI Agent
        "Natriumpicosulfat/Laxoberon": "Laxative",
        "Natriumpicosulfat / Laxoberon": "Laxative",
        "Macrogolum/Transipeg": "Laxative",
        "Natriumpicosulfat": "Laxative",
        
        # --- Antihistamines ---
        "Hydroxycin HCI": "Antihistamine",

        # --- Other/Miscellaneous ---
        "other": "Other",
        "Trospiumchlorid": "Anticholinergic (Bladder)",
        "Spasmourgenin": "Anticholinergic (Bladder)",
        "Trospiumchlorid/Spasmourgenin": "Anticholinergic (Bladder)",
        "Budesonid, Formoterol / Symbicort": "Inhaled Corticosteroid/Bronchodilator",
        "Seritide": "Inhaled Corticosteroid/Bronchodilator", # Typo
        "Mirtazapinum/Remeron": "Antidepressant",
        "Tizanidinum": "Muscle Relaxant",
        "Phytomenadion/Konakion": "Vitamin Supplement (K)",

        ##############################
        # ADDED EVEN EVEN EVEN LATER #
        ##############################
        
        # --- Cardiovascular Agents ---
        "Physiotens (Moxonidin)": "Centrally Acting Agent",
        "central blocker (catapressan )": "Centrally Acting Agent",
        "central blocker ( Physiotens )": "Centrally Acting Agent",
        "Central blocker (physiotens )": "Centrally Acting Agent",
        "Physioteno, 2mg": "Centrally Acting Agent",
        "Hygroton": "Diuretic",
        "Diuretic (spironolactone)": "Diuretic",
        "diuretic ( torem )": "Diuretic",
        "diuretic ( lasix )": "Diuretic",
        "Torem (diuretic)": "Diuretic",
        "Esidrex": "Diuretic",
        "Diuretic (metolazone)": "Diuretic",
        "Alpha-Blocker (Doxzosin)": "Alpha-blocker",
        "alpha blocker (cardura)": "Alpha-blocker",
        "Cardura, alphablocker": "Alpha-blocker",
        "alpha Blocker": "Alpha-blocker",
        "Procoralan": "Heart Rate Reducer (If Channel Inhibitor)",
        "Sildenafil": "PDE5 Inhibitor",
        "Aliskiron": "Renin-Inhibitor",
        "Renin-Inhibitor (Rasilez)": "Renin-Inhibitor",

        # --- Neurology & Psychiatry Agents ---
        "Methadone": "Opioid Analgesic",
        "Haldol": "Antipsychotic",
        "Paracetamolum/Daflagan": "Analgesic (Non-Opioid)",
        "Tizanidium": "Muscle Relaxant",

        # --- Antimicrobials ---
        "Uvamine R (NitrofurantoÃ¯ne) Alternatively": "Antibiotic",
        "Fosfomycin (Monuril)": "Antibiotic",
        "Monuril (Fosfomycine) alternately.": "Antibiotic",
        "Uvamine R (NitrofurantoÃ¯ne) alternately.": "Antibiotic",
        "Uvamine R (NitrofurantoÃ¯ne) en alternance avec Monuril.": "Antibiotic",
        "Monuril (Fosfomycine) en alternance avec Uvamine R.": "Antibiotic",
        "Monuril (Fosfomycine) Alternatively.": "Antibiotic",
        "Monuril (Fosfomycine) alternately": "Antibiotic",
        "Uvamine R (NitrofurantoÃ¯ne) alternately": "Antibiotic",
        "Uvamine R (NitrofurantoÃ¯ne), en alternance": "Antibiotic",
        "Monuril (Fosfomycine), en alternance": "Antibiotic",
        "Alternatively.Uvamine R (NitrofurantoÃ¯ne)": "Antibiotic",
        "Alternatively. Monuril (Fosfomycine)": "Antibiotic",
        "Monuril (Fosfomycine) (Alternatly)": "Antibiotic",
        "Uvamine R (NitrofurantoÃ¯ne) (Alternatly)": "Antibiotic",
        "Monuril (Fosfomycine) Alternatively": "Antibiotic",
        "Uvamin Retard (Nitrofurantoin) alternatively.": "Antibiotic",
        "Furandantine R (NitrofurantoÃ¯ne), alternatively.": "Antibiotic",
        "Furandantine R (NitrofurantoÃ¯ne),": "Antibiotic",
        "Nitrofurantoin": "Antibiotic",
        "Nitrofurantoine (uvamine)": "Antibiotic",
        "Uvamin": "Antibiotic",
        "Furadentine": "Antibiotic",
        "furadantin retad": "Antibiotic",
        "Nitrofurantin": "Antibiotic",
        "Furandantine": "Antibiotic",
        "fosfomycin": "Antibiotic",
        "Fosmomycine": "Antibiotic",
        "phosphomycine": "Antibiotic",
        "Stabicilline": "Antibiotic",
        "Vancomycin (vancocin)": "Antibiotic",
        "Polymyxin  E": "Antibiotic",
        "daptomycin": "Antibiotic",
        "Co Amoxicillin": "Antibiotic",
        "azythromycin": "Antibiotic",
        "isoniacid": "Antibiotic",
        "isoniazide": "Antibiotic",
        "Nystatinum / Multilind": "Antifungal",
        "Entecavir (Baraclude)": "Antiviral",
        "Zeffix (Lamivudine)": "Antiviral",
        "Viread": "Antiviral",
        "Harvoni": "Antiviral",
        "Felvire": "Antiviral",
        "BARACLUDE": "Antiviral",
        "3 TC": "Antiviral",
        "baraclud": "Antiviral",
        "entÃ©cavir": "Antiviral",
        "Pentamidine": "Antiparasitic",

        # --- Immunomodulators & Anti-inflammatories ---
        "belatacept  I / V": "Biologic/Immunosuppressant",
        "belatacept I / V 1 X / mois": "Biologic/Immunosuppressant",
        "Belatacept iv": "Biologic/Immunosuppressant",
        "Tocilizumab": "Biologic/Immunosuppressant",
        "Plaquenil": "Biologic/Immunosuppressant",
        "thymoglobuline": "Biologic/Immunosuppressant",
        "Cure Rituximab X 4": "Biologic/Immunosuppressant",
        "Immunoglobulin": "Immunoglobulin Therapy (IVIG)",
        "Hepatect": "Immunoglobulin Therapy (IVIG)",
        "IV / IG Cure": "Immunoglobulin Therapy (IVIG)",
        "Kiovig IV / IG": "Immunoglobulin Therapy (IVIG)",
        "Privigen/ IVIG": "Immunoglobulin Therapy (IVIG)",
        "cure de kiovig I/V": "Immunoglobulin Therapy (IVIG)",

        # --- Endocrine Agents ---
        "Levothyroxin natrium": "Thyroid Hormone",
        "Elthroxine": "Thyroid Hormone",
        "Elthroxin": "Thyroid Hormone",
        "Eltyroxin": "Thyroid Hormone",
        "Euthroxin": "Thyroid Hormone",
        "Neomercazol": "Antithyroid Agent",
        "Dulaglutid s.c.": "GLP-1 Receptor Agonist",

        # --- Lipid-Lowering Agents ---
        "Ezetimib": "Cholesterol Absorption Inhibitor",
        "Ezetrol": "Cholesterol Absorption Inhibitor",
        "Ezetrol (other antihyperlipidemic ttt)": "Cholesterol Absorption Inhibitor",
        "Ezetrol (antihyperlipidemia ttt other than statin)": "Cholesterol Absorption Inhibitor",
        "Alirocumab (Praluent)": "PCSK9 inhibitor",

        # --- Renal & Metabolic Agents ---
        "Sodiumbicarbonat (Nephrotrans)": "Bicarbonate Supplement",
        "Natrii hydrogemcarbonans": "Bicarbonate Supplement",

        # --- Urological Agents ---
        "Fesoterodin": "Anticholinergic (Bladder)",
        "Trospiumchlorid / Spasmo-Urgenin": "Anticholinergic (Bladder)",
        
        # --- Supplements & Minerals ---
        "Vitamin D3": "Calcium/Vitamin D Supplement",
        "Calcium D3 / Calcimagon": "Calcium/Vitamin D Supplement",
        "Synth.Calcitriol/Rocaltrol": "Calcium/Vitamin D Supplement",
        "Vitamin D*": "Calcium/Vitamin D Supplement",
        "Calcium Sandoz": "Calcium/Vitamin D Supplement",
        "Synt. Calcitriol/Rocaltrol": "Calcium/Vitamin D Supplement",
        "Paricalcitrol / Zemplar": "Calcium/Vitamin D Supplement",
        "Magnesium": "Electrolyte Supplement",
        "Cyanocobalaminum/Vitarubin": "Vitamin Supplement (B12)",
        "Vit B1,2,6/C/FolsÃ¤ure/Dialvit": "Vitamin Supplement (B-Complex)",
        
        # --- Miscellaneous ---
        "other": "Other",
        "Natriumthiosulfat": "Other", # Antidote
        "Immunostimulants": "Immunostimulant",
        "Bronchovaxom": "Immunostimulant",
        "immunstimulantic drug E.coli": "Immunostimulant",
        "Acide Fatigue": "Other", # Ambiguous
        "Carmethin": "Other",
        "Geon": "Other",
        "Tyuesyurt": "Other",
        "Ulovaxon": "Other",
        "Prolin": "Other",
        "Zytorix": "Other",
        "Acidium": "Other",
    
        #############
        # LAST ONES #
        #############

        # Medical/Therapeutic Procedure
        "Plasmapheresis.": "Medical/Therapeutic Procedure",
        "Plasmaphereses": "Medical/Therapeutic Procedure",
        "ABO-immunabsorbtion": "Medical/Therapeutic Procedure",
        "AB0-IgG-Adsorption": "Medical/Therapeutic Procedure",
        "plasmapherese for initial disease": "Medical/Therapeutic Procedure",

        # --- Antimicrobials ---
        "Uvamine R (NitrofurantoÃ¯ne) alternatively.": "Antibiotic",
        "Monuril (Fosfomycine) alternatively.": "Antibiotic",
        "Uvamine R (NitrofurantoÃ¯ne) alternatively": "Antibiotic",
        "Monuril (Fosfomycine) alternatively": "Antibiotic",
        "Monuril (Fosfomycin) Alternatively.": "Antibiotic",
        "Monuril (Fosfomycine), alternatively.": "Antibiotic",
        
        # --- Gastrointestinal Agents ---
        "Esomeprazolum/Nexium": "Proton Pump Inhibitor (PPI)",

        # --- Cardiovascular Agents ---
        "Lisoprinolum": "ACE-Inhibitor",
        
        # --- Respiratory & Allergy Agents ---
        "Clemastin / Tavegyl": "Antihistamine",
        "Telfast": "Antihistamine",
        "Seritide Spray": "Inhaled Corticosteroid/Bronchodilator",

        # --- Renal & Metabolic Agents ---
        "Calcium / Ca-Acetat": "Phosphate Binder",

        # --- Supplements & Minerals ---
        "Vit. B 12": "Vitamin Supplement (B12)",
        
        # --- Miscellaneous ---
        "Scopolamin (Augentropfen)": "Anticholinergic (Ophthalmic)",
    }