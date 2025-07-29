import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict


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

    # Some key variables to build the data (used for drug df only for now)
    START_TIME_IMPUTATION_STRATEGY = "aggressive"  # None, "normal", "aggressive", "remove"
    STOP_TIME_IMPUTATION_STRATEGY = "remove"  # None, "normal", "aggressive", "remove" 
    OTHER_DRUG_TYPE_MAPPING_STRATEGY = "coarse"  # None, "normal", "coarse"
    
    # To use for removing missing or invalid values
    VALID_DATE_RANGE = (pd.Timestamp("1900-01-01"), pd.Timestamp("2030-01-01"))
    NAN_LIKE_DATES = (pd.NaT, pd.Timestamp("9999-01-01"), pd.Timestamp("2000-01-01"))
    NAN_LIKE_NUMBERS = (np.nan, pd.NA, -555.0, -666.0, -777.0, -888.0, -999.0)
    NAN_LIKE_CATEGORIES = (
        "NaN", "nan", "Nan", pd.NA, np.nan, "NA in FUP", -999.0,  # "Unknown"
        "Global consent refused", "Refused", "Not done", "Not applicable",
    )

    # Mapping for reference transplant center
    REFERENCE_CENTER_NORMALIZATION_MAP = {
        "USZ": "University Hospital Zurich",
        "USB": "University Hospital Basel",
        "CHUV": "Lausanne University Hospital",
        "BE": "Bern University Hospital",
        "HUG": "Geneva University Hospital",
        "SG": "St. Gallen Hospital",
    }

    # Mapping for patient ethnicity normalization
    INVERSE_ETHNICITY_NORMALIZATION_MAP = {
        'African': ['african'],
        'African (Mauritian)': ['mauritian'],
        'African (malagasy)': ['malagasy'],
        'American Indian': ['american indian', 'amerindien'],
        'Asian': ['asian'],
        'Asian (Indian)': ['india'],
        'Asian (Mongolian)': ['mongolian'],
        'Asian (Sri Lankan)': ['sri lanka'],
        'Asian (Tamil)': ['tamil', 'thamil'],
        'Asian (Thai)': ['thai'],
        'Caucasian': ['caucasian'],
        'Latin American': ['latin american', 'latino american'],
        'Latin American (Dominican Republic)': ['dominican republic'],
        'Latin American (Hispanic)': ['hispanic'],
        'Middle Eastern (Arab)': ['arab', 'arabic'],
        'Middle Eastern (Kurdish)': ['kurde'],
        'Middle Eastern (Lebanese)': ['lebanese'],
        'Middle Eastern (Persian)': ['persian'],
        'Middle Eastern (Syrian)': ['syria'],
        'Middle Eastern (Turkish)': ['turkey', 'turkish'],
        'Middle Eastern (Yemeni)': ['jemen'],
        'Mixed Race': ['mixed race'],
        'Mixed Race (Caucasian and African)': ['caucasian and africain', 'african/caucasian'],
        'Mixed Race (Caucasian and Arab)': ['caucasian and arabic'],
        'Mixed Race (Caucasian and Latin American)': ['caucasian and latino', 'caucasian and latino american'],
        'Mixed Race (Moroccan and Caucasian)': ['marokko and caucasian'],
        'Mixed Race (Syrian, Saudi, Caucasian)': ['syrian, saoudian, caucasian'],
        'North African': ['north africa'],
        'North African (Moroccan)': ['moroccan'],
        'North African (Tunisian)': ['tunesier', 'tunisia', 'tunisian'],
        'South American': ['amã©rique du sud', 'amérique du sud', 'south american'],
        'South American (American Indian)': ['south american (american indian)'],
        'South American (Bolivian)': ['bolivian'],
        'South American (Brazilian)': ['brasil', 'brazil', 'brazilian'],
        'South American (Chilean)': ['chile', 'sud american (chili)'],
        'South American (Ecuadorian)': ['ecuadorian'],
        'South American (Peruvian)': ['peru'],
        'Unknown': ['unknown'],
    }

    # Ethnicity mapping as key = ethnicity value in the raw table -> ethnicity used by ML models
    ETHNICITY_NORMALIZATION_MAP = {k: v for v, ks in INVERSE_ETHNICITY_NORMALIZATION_MAP.items() for k in ks}

    # Mapping for "other" drug normalization (only for "class" == "OtherDrugs")
    INVERSE_DRUG_NORMALIZATION_MAP = {
        'Antibiotic': ['Uvamine Retard (Nitrofurantoin) en alternance avec le Bactrim.', 'Uvamine R (NitrofurantoÃ¯ne) en alternance', 'Uvamin R (NitrofurantoÃ¯ne) Alternatively.', 'Uvamine R (NitrofurantoÃ¯ne) en alternance.', 'Monuril (Fosfomycine) en alternance.', 'Monuril (Fosfomycine) en alternance', 'monuril (fosfomycine)', 'Fosfomycin/Monuril', 'Fosfomycin (Monuril) in alternance with Nitrofurantoin (Furadantin)', 'Nitrofurantoin / Uvamin', 'Nitrofurantoin (Uvamin)', 'nitrofurantoine (uvamine)', 'antiseptique (furandantine retard )', 'Furandantine R (NitrofurantoÃ¯ne) en alternance.', 'Furandantine R (NitrofurantoÃ¯ne) en alternance', 'furandantin retard prophylaxie pour cystite', 'Furadantine (Uvamine R)', 'furandantin retad', 'Furandantine retard', 'furandantine', 'furadantine', 'nIFURANTIN', 'uvamines', 'uvamine retard', 'NitrofurantoÃ¯ne', 'nitrofurantoÃ¯ne', 'nitrofurentoine', 'Nitrofurantoin (Furadantin retard)', 'Clindamycine', 'Clindamycin (Dalacin)', 'dalacin', 'Clindamycin', 'Clindamycinum', 'Flucloxacillin', 'Amikin (Amikacine)', 'Teicoplaninum', 'Daptomycin.', 'Daptomycin', 'azythromycine', 'azithromycine', 'Azythromycin', 'Azithromicin', 'Azithromycinum', 'Azitromycin', 'Macrolides', 'Erythromycin', 'Erythrocine', 'Erythrocin (Erythromycine)', 'Quinolone', 'Chinolon', 'Ciprofloxacine', 'Ciprofloxan', 'Cyprofloxacin', 'Ciprofloxcain', 'Norfloxacine', 'penicilline', 'Ospen (PÃ©nicilline V)', 'Stabicillin (phenoxymethylpenicillin)', 'ospen', 'Amoxicillinum', 'Amoxicillin', 'amoxicilline', 'Amoxillin', 'co-amoxicillin', 'Ceftazidimum', 'Ceftazidime (Fortam)', 'Ceftriaxonum', 'Rocephin', 'Oftrioxel', 'cefluroxine', 'cÃ©furoxime', 'Carbapeneme', 'Imvanz I / V , 3 X / Week', 'Ertapenem', 'Vancomycine.', 'Vancomycine', 'Vancomycinum', 'Vancomycine (vancocin)', 'Polymyxin E', 'polimyxin E', 'Polimixine E/ Colistin', 'Colistin', 'aerosol colistin', 'Doxycydinum', 'Doxycyclin', 'doxycicline', 'Doxycycline', 'Rifaximin', 'Xifaxan (rifaximin)', 'Xifafan (Rifamixine)', 'Xifafan (Rifaximine)', 'Xifafan', 'Rifampicin', 'Rifampicinum, Isoniazidum', 'Rifampicine', 'Pyrazinamidum', 'Pyraminamide', 'Isoniazid/Rimifon', 'isoniazid', 'Isoniazide pour TB en prophylaxie', 'Tuberculostatic', 'Ethambutol', 'Garamycine', 'garamycine', 'Gentamycine.', 'Gentamnycine', 'Sulfonamide', 'Co-trimoxazole', 'sulfadiazine (clindamycine )', 'Linezolid', 'Monuril (Fosfomycine)', 'Uvamine R (NitrofurantoÃ¯ne)', 'Vancomycine (Vancocin)', 'Vancomycin (Vancocin)', 'Isoniazid', 'Uvamine Retard (Nitrofurantoin)', 'azithromycin', 'Stabicilline <phÃ©noxymÃ©thylpÃ©nicilline (pÃ©nicilline V)>', 'Azithromycin', 'Ospen', 'Polymyxine B + NÃ©omycine', 'monuril', 'Fosfomycin', 'Monuril', 'Isoniazide', 'Gentamycine', 'Polymixin E', 'fosfomycine (monuril)', 'Dapson/Pyrimethamin', 'Clindamycine.', 'Dapson', 'Ciprofloxacin', 'Vancomycin', 'Penicilline benzathine', 'nitrofurantoine', 'Vancomycin (Vancocine)', 'Azithromycine (Zithromax)', 'Furandantine R (NitrofurantoÃ¯ne)', 'Uvamine', 'Zithromax', 'fosfomycine', 'Moxifloxacin', 'Acitromycin', 'Ethambutoli hydrochloridum', 'Isonazid', 'Isoniazidum', 'Ospen (Phenoxymethylpenicillinum kalicum)', 'uvamine', 'Fosmomycin / Monuril', 'Stabilline', 'vancomycin', 'polymixin E', 'Furadantine', 'Fosfomycinum', 'furadentine', 'B-Lactame', 'meropenem', 'ceftriaxone', 'phosphomycine', 'colistin', 'macrolide', 'ceftazidine', 'Bactrim', 'Isoniazide for Tuberculosis', 'Maxifloxacine for Tuberculosis', 'Nifabutine for Tuberculosis', 'Co Amoxicillin', 'azitromycine', 'Colistimethat / Colistin', 'Azithromycin / Zithromax', 'Monuril (Fosfomycine) Alternatively.', 'Fosfomycine (Monuril)', 'Monuril (Fosfomycine) en alternance avec Uvamine R (NitrofurantoÃ¯ne)', 'Uvamine R (NitrofurantoÃ¯ne) Alternatively.', 'Monuril (fosfomycine)', 'Uvamin R', 'Fosfomycine', 'Nitrofurantoine', 'meronem', 'Azithromycine', 'vancomycine', 'Rifabutine', 'Xifaxan (Rifaximine)', 'Uvamine R (NitrofurantoÃ¯ne) Alternatively', 'Fosfomycin (Monuril)', 'Monuril (Fosfomycine) alternately.', 'Uvamine R (NitrofurantoÃ¯ne) alternately.', 'Uvamine R (NitrofurantoÃ¯ne) en alternance avec Monuril.', 'Monuril (Fosfomycine) en alternance avec Uvamine R.', 'Monuril (Fosfomycine) alternately', 'Uvamine R (NitrofurantoÃ¯ne) alternately', 'Uvamine R (NitrofurantoÃ¯ne), en alternance', 'Monuril (Fosfomycine), en alternance', 'Alternatively.Uvamine R (NitrofurantoÃ¯ne)', 'Alternatively. Monuril (Fosfomycine)', 'Monuril (Fosfomycine) (Alternatly)', 'Uvamine R (NitrofurantoÃ¯ne) (Alternatly)', 'Monuril (Fosfomycine) Alternatively', 'Uvamin Retard (Nitrofurantoin) alternatively.', 'Furandantine R (NitrofurantoÃ¯ne), alternatively.', 'Furandantine R (NitrofurantoÃ¯ne),', 'Nitrofurantoin', 'Nitrofurantoine (uvamine)', 'Uvamin', 'Furadentine', 'furadantin retad', 'Nitrofurantin', 'Furandantine', 'fosfomycin', 'Fosmomycine', 'Stabicilline', 'Vancomycin (vancocin)', 'Polymyxin  E', 'daptomycin', 'azythromycin', 'isoniacid', 'isoniazide', 'Uvamine R (NitrofurantoÃ¯ne) alternatively.', 'Monuril (Fosfomycine) alternatively.', 'Uvamine R (NitrofurantoÃ¯ne) alternatively', 'Monuril (Fosfomycine) alternatively', 'Monuril (Fosfomycin) Alternatively.', 'Monuril (Fosfomycine), alternatively.'],
        'Biologic/Immunosuppressant': ['Rituximab', 'rituximab', 'Rituximab (Mabthera)', 'Mabthera (Rituximab)', 'Mabthera', 'Rituximab (Rixathon)', 'Rituximab.', 'cure de rituximab', 'cure Rituximab', 'Cure Rituximab X4', 'cure Rituximab X 4', 'cure  Rituximab', 'Belatacept', 'belatacept', 'belatacept I/V', 'betalacept', 'Belatazept', 'Belactasept', 'Belataset 1 X / Month', 'BÃ©latacept  (I /V )', 'bÃ©latacpet (I / V )', 'BÃ©latacept ( I / V )', 'Belatacept 1 X / Month', 'Belatacept (Nujolix)', 'Nulojx', 'Eculizumab', 'Ã©culizumab (soliris ) I / V', 'Eculizimab I/V every 2 weeks', 'Soliris every 3 weeks', 'Eculizimab 1 X / Month', 'ATG/Eculizumab', 'Abatacept', 'Orencia', 'Adalimumab / Humira', 'Cyclophosphamid', 'Cyclophosphamide', 'Cyclophosphamid (Endoxan)', 'inter Endoxan Therapy', 'Etanercept s.c.', 'TNF-Inhibitor /Enbrel', 'Leflunomid/Arava', 'Thymoglobuline I/V', 'Thymoglobuline.', 'thymoglobuline  I / V', 'Immunglobulin anti-T-Lymphozyten human', 'Grafalon', 'AEB 071', 'Campath (Alemtuzumab)', 'Kineret 100mg', 'kineret', 'anakinra', 'Remicade', 'Actemra/Tocilizamab', 'stelara (ustekinumab)', 'Vedolizumab', 'Canakinumab', 'canakinumab', 'Interferon beta-1a', 'Belatacept (Nulojix)', 'Etanercept (Enbrel)', 'Enbrel', 'Etanercept', 'Enbrel (etanercept)', 'Leflunomid', 'Thymoglobulin', 'Thymoglobuline', 'ATG', 'Anakinra', 'cure Brentuximab I/V', 'cure Rithuximab I/V', 'Eculizumab (Soliris)', 'Soliris', 'eculizumab I / V', 'Leflunomid / Arava', 'etanercept', 'Hydroxychloroquin', 'Belatacept i.v.', 'belatacept (nulogix)', 'Etanerceptum', 'Rituximab IV', 'Etanercept injection', 'Anakinra (Kineret)', 'Infliximab', 'Soloris', 'Endoxon', 'Endoxan', 'Endoxan (Cyclophposphamide).', 'Kineret', 'Ethanercept', 'Advagraf', 'Rituximab for initial desease', 'eculizumab', 'Betalacept', 'MabThera (Cytostatic drug)', 'Mercaptopurin', 'Tocilizumab', 'toculizumab', 'Plaquenil', 'Cyclosporin', 'cure rituximab', 'Enbrel (Etanercept)', 'Soliris (Eculizumab) 5 times (once a week)', 'belatacept I / V', 'Thymoglobulines.', 'Cure Rituximab X 4', 'Rituximab I/V', 'Eculizimab', 'belatacept  I / V', 'belatacept I / V 1 X / mois', 'Belatacept iv', 'thymoglobuline'],
        'Diuretic': ['Diuretic', 'diuretic', 'diuretics', 'diuretiic', 'Diueretic', 'Diurectics', 'diurectics', 'diuretic 2 weeks', 'Diretikum', 'Diureticum', 'Diaretikum', 'DiurÃ©tique', 'Diuretic ( Torem )', 'diuretic (torem)', 'diuretic (Torem)', 'diuretic (lasix )', 'Lasix (diuretic )', 'diuretic ( Aldactone )', 'diuretics (aldactone )', 'diuretic (aldactone)', 'diuretic (inspra)', 'diuretic (comilorid )', 'diuretic (coversum combi)', 'Valsartan (diuretic)', 'diuretic (metolazone)', 'metolazone (diuretic)', 'Diuretic- Inspra, metolazole', 'Diuretic (Thiazid)', 'Torasemid', 'Torasemide.', 'Torasemidum', 'Torasemidium', 'Toresemidum', 'Torsemid / Torem', 'Torasemid / Torem', 'Toresemiid / Torem', 'Torasamied / Torem', 'Torsemid / Torsis', 'Torem', 'torem', 'Torfin', 'Spironolacton / Aldactone', 'Spironolacton / Xenalon', 'Sprironolactonum', 'Prolactone', 'Aldosteron Antagonist', 'Furosemidum.', 'Forosemid / Lasix', 'Furosemid / Lasic', 'Metalozon', 'Metolaznum', 'Thiazidum', 'Thiaziddiuretika', 'Thiaziddiuretikum', 'Chlorthalidone', 'clorthalidone', 'Chlorthiclidone', 'Chlortachidon', 'chlortalidon', 'Hydrchlorothiazid', 'Indapamid', 'indapamid', 'Judapamid', 'Indapenid', 'Indapamid retard', 'Esidrix', 'Inspra', 'Thiazid', 'Thiazide', 'Thiazideddiusetic', 'Metolazonum', 'Metolazon', 'Metalozonum', 'Metolazonum.', 'Spironolactonum', 'Spironolacton', 'Aldactone', 'aldactone', 'diuretic ( aldactone )', 'diuretic (spironolactone)', 'ALdactone', 'Spironolacton / Aldacton', 'Eplerenon', 'diuretic(inspra)', 'Furosemidum', 'Furosemid / Lasix', 'diuretic (lasix)', 'Furosemid/Lasix', 'Forosemic / Lasix', 'diuretic (torem )', 'diuretic (torasemid)', 'Toremisid', 'Torasemid/Torem', 'Toresemid', 'Torasamid', 'Torasemidum.', 'Hydrochlorothiazid', 'Hydrochlorothiazidum', 'Hydrochlorothiazide', 'Comilorid', 'Chlortalidon', 'Chlorthalidon', 'Diuretics', 'Diuretic (torem)', 'Diuretic ( esidrex)', 'Diuretikum', 'Diuretika', 'Diuretic ( + )', 'Furosemid', 'Torasemid / Torsis', 'Diuretiz', 'Diuretics.', 'TorasÃ©mide', 'diuretic ( torem )', 'Esidrex', 'Hygroton', 'Diuretic (spironolactone)', 'diuretic ( lasix )', 'Torem (diuretic)', 'Diuretic (metolazone)'],
        'Alpha-blocker': ['Alpha blocker', 'Alpha-Blocker', 'alpha-blocker', 'alpha blocker', 'alpha- Blocker', 'Alpha-Antagonist', 'Alphablocker', 'alphablocker', 'alpha- blocker:', 'alpha-rezeptor blocker', 'Alpha-1-Adrenorezeptoren-Blocker', 'Doxazosin', 'doxazosin', 'Doxasosinum/Cardura', 'Doxazosinum', 'Doxazosinum / Cardura', 'Doxazosin/Cardura', 'Doxacosin', 'Doxazesin', 'doxacosinmesylat', 'Docazosin', 'Cardura', 'cardura', 'Cardura (Doxazosine)', 'Cardura (Doxazosin)', 'Doxazosin (Cardura)', 'Cardura (Alpha-Blocker)', 'Cardura (alpha-blocker)', 'cardura (alpha-blocker)', 'alpha blocker(cardura)', 'Cardura, Alphablocker', 'Tamsulosin', 'tamsulosine (pradif)', 'Tamsulosini (HCI) / Pradif', 'Tamsulosini / Pradif', 'Tamsulosini/Pradiv', 'Tamsulosini hydrochlorid/Pradif', 'Tamsulosini Hydrochloridum.', 'Tamsulosion / Pradif', 'Pradif', 'pradif', 'Pradif T (Tamsulosine)', 'Pradif (Tamsulosine)', 'Pradif (Alphablocker)', 'Xatral', 'Xatral Uno (Alfuzosine)', 'Hytrin / alpha receptor blocker', 'DUODART', 'Tamsulosini hydrochloridum', 'alpha-Blocker', 'Alpha-blocker', 'alpha-blocker (cardura)', 'Alpha Blocker', 'Doxazosin / Cardura', 'Cardura (Doxazosin mesylate)', 'alpha-blocker (cardura )', 'Doxazosine (Cardura)', 'Cardura (doxazosine)', 'Tamsulosini Hydrochloridum', 'Tamsulosini', 'Tamsulosin / Pradif', 'Tamsulosini hydrochlorid', 'Chinazoline', 'Duodart', 'Duodart (DutastÃ©ride + tamsulosine)', 'Doxazosinum/Cardura', 'Doxazosin (alpha-blocker)', 'Cardura (Alphablocker)', 'Cordura', 'tamsulosine (prodif)', 'Pradif T (Tamsulosine', 'Alphabocker', 'alpha blocker ( Cardura )', 'Alpha-Blocker (Doxzosin)', 'alpha blocker (cardura)', 'Cardura, alphablocker', 'alpha Blocker'],
        'Antiviral': ['Letermovir', 'letermovir', 'Intelence', 'Etravirine/Intelence', 'Vemlidy (Tenofovir alafenamid)', 'Tenofovir (Vemlidy)', 'Tenofovir (vemlidy)', 'Tenofovir alafenamid', 'Tenofovir alafenamid (Vemlidy)', 'Vemlidil', '_Vemlidy (Tenofovir alafenamid)', 'Viread (TÃ©nofovir)', 'Valcanciclovir', 'Oseltamivir / Tamiflu', 'tameflu', 'Ribavirin', 'ribavirin', 'Copegus', 'lamivudin', 'Daclasvir', 'Daclatasvir', 'Sovaldi', 'Epclusa', 'Harvoni (Anti HCV)', 'Hep-C-Therapy (Harvoni)', 'Enteclavir (barclude )', 'Entecavir / Baraclude', 'Baraclude (Entecavir)', 'Baraclude (Emtecavir)', 'baraclude', 'entecavir', 'AntÃ©cavir', 'Telbivudinum (Sebivo)', 'Sebivo (Telbivudine)', 'Adefovir (Hepsera)', 'anti-HIV treatment', 'Anti-HIV Treatment', 'antiviral HIV therapy', 'Virostatikum', 'antiviral Hep. B therapy', 'antiviral Hep.-C therapy', 'Integraseinhibitoren', 'NRTI, Virostatika/Retrovir', 'Celsentri', 'Descory', 'Descovy/Tivicay', 'Isenstress', 'Isentress', 'Raltegravir', 'Pifeltro', 'Odefsey', 'Edurant', 'Edurant NB', 'Abacavir', 'Lamivudin / Abacavir / Dolutegravir', 'ATG (Lamivudin, Abacavir und Dolutegravir)', 'Dolutegravir', 'dolutegravir', 'Dolutegravier', 'anti-viraux', 'Remdesivir', 'Lamivudine', 'Tivicay', 'Tamiflu', 'Baraclude', 'Entecavir', 'Tenofovir', 'Ziagen', 'Triumeq', 'Sofosbuvir', 'Ribavirine', 'Vemlidy', 'Descovy', '3TC', 'antiviral therapy', 'Emtriva', 'Kivexa', 'Stocrin', 'TTT HIV', 'viread', 'Zeffix', 'Asefovirdipivoxil', 'entÃ©cavir', 'baraclud', 'Famvir (famciclovir)', 'Dovato ( lamivudine + dolutÃ©gravir)', 'Trinmeq', 'Entecavir (Baraclude)', 'Zeffix (Lamivudine)', 'Viread', 'Harvoni', 'Felvire', 'BARACLUDE', '3 TC'],
        'Medical/Therapeutic Procedure': ['blood transfusion', '1 blood transfusion', '2 bloods transfusion for anemia', '2 bloods transfusions', '2 bloods transfusions due HB lower 82', '3 bloods transfusions', '1 blood transfusion due lower 89', 'blood tranfusion', 'Plasmapheresis', 'plasmapheresis', 'plasmapherisis', 'plasmapherese  for initial disease', 'PEX', 'PEX plasmapheresis', 'plasma exchange (PEX)', 'Plasmapheresis 5 times', 'plasmapherese for initial disease', 'Immunadsorption', 'Immunoadsorption', 'Immunoabsorption', 'ABO-IgG Immunabsorbtion', 'ABO-IgG Adsorption', 'AB0-Immunabsorbtion', 'AB0-immunabsorbtion', 'AB0-Immunadsorption', 'AB0-Immunadsoption', 'ABO-IgG-Absorbtion', 'ABOi IgG Adsorption', 'AB0-IgG Absorption', 'IgG Adsorption', 'Photopheresis', 'photopheresis', 'oxygen administration', 'oxygene', 'TLI', 'TLI in accordance with Swisstolerance StudyÂ” protocol', 'Radiation therapy due to swisstolerance.CH study', 'radiotherapy ttt for bone metastasis', 'Radiotherapy', 'Deflux injection', 'AB0-IgG Adsorption', 'ABO-Immunadsorption', 'Plasmapheresis/PE', 'Plasmaexchange', 'Plasmapherisis 3 X / W', 'IAD', 'IADS', 'AB0-Immunabsorption', 'AB0-Immunadsorbtion', 'Adsorption', 'recive 8 Plasmapheresis /PE', 'Plasmapheresis /PE X 4', 'plasmaphÃ©rÃ¨se X1', 'Immunabsorbtion', 'Glycosorb', 'Photopherese', 'Plasmapheresis.', 'Plasmaphereses', 'ABO-immunabsorbtion', 'AB0-IgG-Adsorption'],
        'Centrally Acting Agent': ['anti-hypertenseur (catapressan)', 'Central blocker', 'central blocker', 'Central-blocker', 'Central blocker (catapressan )', 'central blocker(catapressan)', 'Central Blocker (Catapressan)', 'Alpha-bloquer (Catapresan)', 'central blocker (physiotens )', 'Central blocker (physiotens)', 'central Blocker (phisiotens)', 'Physiotens (Central Adrenergic)', 'Central Adrenergic Physiotens', 'Alpha-Rezeptor-Agonist(Catapressan)', 'Ã¡2-Rezeptor-Agonist', 'Antisympathotonikum', 'Moxonidinum / Physiotens', 'Moxonidin / Physiotens', 'Moxonidum/Physiotens', 'Moxonidin / Physiotrans', 'Moxonidinum', 'Moxonidium', 'Moxomidin', 'Monoxidin', 'Catapresan', 'Methyldopa/Aldomet', 'Physiotens (Moxonidinum)', 'Physiotens', 'Central Blocker', 'central blocker (physiotens)', 'Physiotens (Central adrenergic)', 'Moxonidin (Physiotens)', 'Catapressan', 'central-blocker', 'Clonidin', 'clonidin', 'Clonidini hydrochloridum', 'central blocker ( catapressan )', 'alpha-blocker (catapressan )', 'alpha blocker ( catapressan )', 'alpha blocker ( Catapressan )', 'Central adrenergic (Physiotens)', 'Moxonidin', 'Physiotens (moxonidine)', 'Methyldopa', 'Physiotens (Moxonidin)', 'central blocker (catapressan )', 'central blocker ( Physiotens )', 'Central blocker (physiotens )', 'Physioteno, 2mg'],
        'Antidepressant': ['Antidepressant drug', 'Antidepressant', 'anti-depressive treatment', 'Tricyclic antidepressants', 'tricyclic antridepressant', 'Antidepressants SSRI', 'serotonin reuptake inhibitors', 'psycho-analeptique', 'Mirtazapin', 'MMirtazepin', 'Mirazepin', 'Mirtazepin / Remeron', 'Remeron/ Antidepressivum', 'Mitrazapinum', 'Sertralinum', 'Seralin', 'Serfralin', 'Cipralex (Escitalopram)', 'Escitalopramum', 'Ecitalopram', 'Imipramini hydrochloridum', 'Trazodon/Trittico', 'Surmontil (Depression)', 'Amitriptylin', 'Doxepinum', 'Bupropioni hydrochloridum', 'Bupropion', 'Venlafaxinum', 'Venlaflaxin', 'Venlafaxin', 'Effexor', 'Citalopranum / Citaoprami hydrobromidum', 'Seropram', 'Fluoxetinum', 'Fluctine', 'CImbalta', 'Trimipran', 'Trimipramini mesilas', 'valdoxan', 'Vortioxetin', 'Brintellix', 'Paroxitin', 'Cipralex', 'Citalopram', 'Escitalopram', 'Sertralin', 'Antidepressivum', 'Mirtazapinum', 'Remeron', 'Cymbalta', 'Trittico', 'Saroten', 'Antidepressiva', 'Mirtazapinum / Remeron', 'Citalopramum', 'citalopram', 'Citalopranum', 'Trazodon', 'Antidepressive drug', 'Deanxit', 'Trimipramin', 'escitalopram', 'Mirtazepin', 'Mirtazapinum/Remeron'],
        'Calcium/Vitamin D Supplement': ['Cacitrol', 'Cacitriol (Rocaltrol)', 'rocaltrol', 'Nocaltol', 'Calcitriol / Rocaltrol', 'Synth. Calitriol', 'Synth. Calcitriol/Rocaltrol', 'Cynthetic calcitrol', 'Vit D / Rocaltrol', 'Vit. D / Rocaltrol', 'Paricalcitol/Zemplar', 'Paricalcitol / Zemplar', 'ViDe3', 'Vi De3', 'Vit D3', 'Vit D 3', 'VIT D3', 'Viamin D', 'Colecalciferol / Vi.D3', 'Colecalciferol Vit D3', 'Calcilum D3', 'Ca-Vit D3', 'Ca D3', 'Ca Vit 3', 'Calcium Vitamin D', 'Calcimofen D3', 'Calcimagone', 'Calcium D3', 'Calcimagon', 'Calcitriol', 'Colecalciferol', 'Vit. D', 'Rocaltrol', 'Vit. D3', 'Calcium', 'Calcium Sandoz D3', 'Calcium Sandoz', 'Synthetic calcitrol', 'Synth. Calcitriol', 'Vit D', 'Vi-De', 'Calcitriol Vit D', 'Calcium Sandoz D3f', 'Synthetisches Calcitriol', 'Synth Calcitriol', 'Calcitril', 'Colecalciferol (Vit.D)', 'Paracalcitrol', 'Calcitiol / Rocaltrol', 'Calcitiol', 'Vit. D 3', 'Calcitrol', 'Calcimagon D3', 'Synt. Calcitriol', 'Vitamin D', 'Synthetic calcitriol', 'Vitamin D3', 'Calcium D3 / Calcimagon', 'Synth.Calcitriol/Rocaltrol', 'Vitamin D*', 'Synt. Calcitriol/Rocaltrol', 'Paricalcitrol / Zemplar'],
        'Chemotherapy/Targeted Therapy': ['Chemotherapie', 'Chemotheraphy', 'chemotherapy', 'CHOP (chemotherapy)', 'FOLFOX (Chemo)', 'FOLFIRI', 'Immunochemotherapy R-CHOP-21', 'Bendamustin', 'Daratumumab (chemotherapy)', 'daratumumab', 'darzalex I/V', 'Darzalex I/V', 'Cetuximab (chemotherapy)', 'Cetuximab', 'Carboplatine (chemotherapy)', 'Doxorubicin (Cytostatics)', 'Doxorubicin Liposomal (chemotherapy)', 'Cyclophosphamid (Cytostatics)', 'Temodal (anti-nÃ©oplasique )', 'cabozantimib', 'Cabozantinib', 'sorafenib', 'lenvatinib', 'Lenvatinib', 'Gitluyro', 'T-VEC I/V at 2 weeks', 'Ibrance', 'Ipilimumab', 'Ipilimumab aux 3 weeks', 'Ipilimumab every 4 weeks', 'Opdivo', 'Vilclad  TTT MyÃ©lome', 'Bortizomib TTT myÃ©lome', 'kiprolis', 'carfilzomib', 'Panobinostat TTT myÃ©lome', 'ixazomid', 'Antineoplastika/Immunmodulator', 'Chemotherapy (Carboplatin/Gemcitabin)', 'Votubia /Everolimus', 'Everolimus', 'Abemaciclib', 'Velcade', 'Revlimid', 'Mekinist', 'Temodal (Cytostatic drug)', 'cytostatic agents', 'chemotherapy (Sorafenib)', 'R-CHOP'],
        'Immunoglobulin Therapy (IVIG)': ['IVIG', 'IvIG', 'iViG', 'IGIV', 'IV / IG', 'I/V ig', 'IV / Ig', 'IG i/v', 's.c. IG', 'IgG s.c.', 'I V /  I G', 'IV /IG treatment', 'iv IG injection', 'immunglobulin (ivIg)', 'Cure IV / IG', 'I/V ig cure', 'Kiovig cure', 'Kiovig I / V', 'Kiovig I / V cure', 'Kiovig cure X 3 days', 'Kiovig X 3 Days', 'Privigen', 'Intratect', 'IgG Privigen', 'Immunglobulin (Zutectra)', 'Cuvitru (Human normal immunoglobulin, subcutaneous)', 'Cytomegalie-Immunglobulin human', 'HÃ©patect', 'hepatect', 'Hepatitis-B-Immunglobulin', 'Cure IV / Ig', 'ivIG', 'I v / I g treatement', 'IG/iv chronic humoral rejection', 'IV / IG Cure', 'Cure I V / I G', 'IG iv', 'Kiovig IV / IG', 'I V / I G', 'Ivig', 'IG / IV', 'IG / iv X1', 'IV /  IG for initial disaese', 'KIOVIG  I / V', 'Hepatect', 'cure IG/iv', 'IVig', 'IV IG', 'KIOVIG I / V', 'cure de kiovig I/V', 'cure de Kiovig I / V', 'Immunoglobulin', 'Privigen/ IVIG'],
        'Anticonvulsant': ['Antiepileptic drug', 'anti-epileptic', 'Antiepileptic', 'anti-Ã©pileptique ( keppra )', 'Antiepileptikum', 'anti-epileptique', 'Levetiracetam (Keppra)', 'Levetiracetam (anticonvulsants)', 'keppra', 'LÃ©vÃ©tiracetam', 'Gabapentin / Neurontin', 'Gabapentin', 'Gabapentine', 'neurontin', 'Gaba pe-ti', 'Neurotin', 'Lyrisa', 'Valproat', 'Natrii valproas', 'Depakine chrono', 'Valproat / Depakine chrono', 'Orfiril', 'Lamotriginum', 'Lamotrizin', 'Lamictal (Lamotrigine)', 'Oxcarbazebinum/Trileptal', 'Phenhydan', 'Topiramat / Topamax', 'Topiramat', 'Topamax', 'Rufinamid', 'Lyrica', 'Pregabalin', 'Keppra', 'Levetiracetam', 'Antiepileptika', 'Levetiracetam / Keppra', 'Depakine', 'Lamotrigin / Lamictal', 'Vimpat', 'Carbamazepin', 'Lamictal', 'Levitriactam', 'Levetiracetam (keppra)', 'Lamotrigin', 'Aphenylbarbit', 'Gabapenta', 'Neurontin'],
        'Inhaled Corticosteroid/Bronchodilator': ['antiasthmatic drug', 'Antiasthmatic', 'Beta-2-Sympathomimetikum', 'Bronchodilatans/ Ã¢ 2 -Sympathomimetikum', 'Formoterol/Budesonid', 'Budesonit, Formoterol / Symbicort', 'Formoterol / Symbiocort', 'Symbicort Spray', 'Symbicort/ Dospir', 'Symbicort; Ventolin', 'Symbicort / Spriva', 'Formoterolum', 'Salmeterolum', 'Formoterol / Foradil', 'Formoterol / Oxis', 'Tiotropium / Spiriva', 'Tiotropium Bromide', 'Ipratropiumbromid / Atrovent', 'Ipramol', 'Relvar Elipta', 'Ultibro', 'Enerzair', 'Avamys', 'Alvesco', 'Anora', 'Symbicort', 'Spiriva', 'Seretide', 'Pulmicort', 'Relvar', 'Seritide', 'Tiotropium', 'Formoterol', 'Atrovacant Spray', 'SPIOLTO', 'Dospir', 'Antiasthmatikum', 'Budesonid, Formoterol / Symbicort', 'Seritide Spray'],
        'Vasodilator': ['anti-hypertenseur (loniten )', 'Minoxidil (Loniten)', 'Minoxidilum / Loniten', 'minoxidil', 'Loniten (Minoxidil)', 'Loniten (Vasodilatateur)', 'Vasodilatateur (loniten )', 'vaso-dilatateur ( loniten )', 'loniten', 'Vasodilatator', 'Glyceroltrinitrat', 'Deponite patch', 'Molsidomine (corvaton retard)', 'molsidomine', 'Nicorandil', 'Nicorandilum (Dancor)', 'vasodilatateur coronarien', 'Koronarvasodilatans', 'Nipruss', 'Nitroderm', 'nitroderm patch', 'Nitroglycerin', 'Dancor', 'Molsidomin', 'Nitro', 'Antihypertonikum / Loniten', 'Corvaton', 'Minoxidilum', 'Opsumit', 'Nitroderm patch', 'Nitroglycerin (Nitroderm)', 'Nicorandil (Dancor)', 'Corvaton retard', 'Nitroderm TTS', 'Loniten (minoxidil)', 'Minoxidil', 'Loniten'],
        'Antiarrhythmic': ['Cordarone', 'cordarone (FA cardiversion )', 'Cordarone (amiodarone)', 'Cordarone (Amiodaron)', 'Amiodarone (Cordarone)', 'Amiodaron', 'amiodaron', 'Amiodaroni Hydrochloridum', 'anti-arrythmic', 'antiarythmie(digoxine)', 'Herzglykosid', 'cardiac glycosides', 'Tambocor', 'Tambocor (coeur)', 'Tambocar (antiarythmique)', 'Flecainid acetat (Tambocor)', 'Flecainid', 'Rhythmonorm', 'multaq', 'Multaq (antiarythmique)', 'Sotalol pour cardioversion', 'Cordarone (Amiodarone)', 'cordarone', 'Amiodarone', 'amiodarone', 'Amiodaron (Cordarone)', 'Amiodaroni hydrochloridum', 'Amidarone', 'Amidaron', 'Amiodaron / Cordarone', 'Digoxin', 'Digoxine', 'Digoxinum'],
        'Corticosteroid': ['Dexamethasone.', 'dextamÃ©thasone', 'Infiltration of Dexamethasone.', 'Hydrocortison', 'Cortison', 'SolumÃ©drol 500 mg./j.', 'prednisone for side effect of cancer traitment', 'Prednisolonum', 'solumedrol', 'Solumedrol', 'solumedrol I / V', 'Solumedrol IV (5 bolus)', 'Solumedrol IV, 5 boluses', 'Methyl-prednisone I V', 'Budesonidum', 'Budesonid', 'Budenofalk 9 mg for 9 month', 'Florinet', 'Glukokortikoid / Spiricort', 'Dexamethasone', 'DexamÃ©thasone', 'DexamÃ©thasone.', 'Glucocorticoid', 'Solumedrol IV', 'Fludrocortison', 'Spiricort', 'Spirocort', 'IV Solumedrol', 'Prednison', 'Hydrocartison', 'Florinef', 'Methyl-prednisone I / V'],
        'Thyroid Hormone': ['thyroid hormone', 'Levothyroxin/Euthyrox', 'Levothyrosin/Euthyrox', 'Levothyroxin / Eltroxin', 'Levothyroxin-natrium / Eltroxin', 'Euthyrox (Levothyroxin)', 'Levothyroxinum natrium/ Eythyrox', '(Levothyroxin natrium)', 'Levothyroxin / Euthyrax', 'Levothyroxinum natricum.', 'Euthyrox', 'Euythyrox', 'euthyrox / Eltroxine', 'etyrox', 'eltoxin', 'Elthyrox', 'Eltoxin', 'euthroxin', 'Thyrox', 'Eltroxin', 'euthyrox', 'Levothyroxin-Natrium', 'Levothyroxin', 'Elthyroxin', 'Thyroxin', 'Euthyroxin', 'L-Thyroxin', 'levothyroxine', 'Eltroxine', 'elthroxine', 'Levothyroxin natrium', 'Elthroxine', 'Elthroxin', 'Eltyroxin', 'Euthroxin'],
        'Proton Pump Inhibitor (PPI)': ['Pantoprazolum / Zurcal', 'Pantoprazolum / Pantozol', 'Pantorprazol / Pantazol', 'Pantoprazol /Pantozol', 'Pantprazolum', 'Pantoprazolium', 'Pantorprazol', 'Pantoplazol', 'pantozol', 'Omeprazolum', 'Omeprazol / Omezol', 'Ozepranolum', 'Esomep', 'Esomep 40 mg', 'esomeprazol', 'Esomepragel', 'Dexilaut', 'Rabeprazolum', 'Pantoprazolum', 'Pantoprazol', 'Pantoprazol / Pantozol', 'Esomeprazolum', 'Pantozol', 'Esomeprazol / Nexium', 'Nexium', 'Pantoprazol / Pantazol', 'Esomeprazol', 'Pantoprazole', 'PPI', 'Pantozolum', 'Pantoprazole 40mg', 'Pantopraxzo / Pantozol', 'Esomeprazolum / Nexium', 'Esomeprazolum/Nexium'],
        'Erythropoiesis-Stimulating Agent (ESA)': ['EPO, Aranesp', 'Dabepoetin alpha', 'Darbepoetin alpha', 'Darbepoetin beta', 'Darbepoetin alfa', 'Aranesp 1X/sem', 'Darbepoetin alfa / Aranesp', 'Epoetanum beta', 'Epoetinum beta/ Recormon', 'Epoetinum beta/Mircera', 'PEG-Epoetin / Mircera', 'PEG-Epoetin / Mircere', 'Epoetin alfa / Eprex', 'Aranesp', 'Darbepoetin / Aranesp', 'Darbepoetin alpha/Aranesp', 'Darbepoetin', 'erythropoetin', 'EPO (aranesp)', 'Epoetunum beta', 'Epoetum beta', 'EPO', 'Erythropoetin', 'Darbopoetin alpha', 'Darbepoetin alpha / Aranesp', 'aranesp'],
        'Antipsychotic': ['Antipsychotic drug', 'Neuroleptics', 'Neuroleptikum', 'Pipamperon', 'Pipamperonum', 'Pimpamperonum', 'Pipamperonum dihydrol', 'Pipamperonum dihydrochloridum', 'Pipamperonum (HCI)', 'Pipamperonum / Dipiperon', 'Quetiapin / Serequel', 'Levomepromazin', 'Levomeprovazinum / Nozinan', 'Haloperidol/^Haldol', 'Dopamin Antagonist', 'Risperdal (RispÃ©ridone)', 'Truxal', 'Quetiapin', 'Quetiapin / Seroquel', 'Olanzapin', 'Neuroleptika', 'Quetiapinum', 'Haloperidol', 'Seroquel', 'Abilify', 'Pipamperon / Dipiperon', 'Pipamperonum/Dipiperon', 'Haldol'],
        'Phosphate Binder': ['Ca Acetat', 'Ca Azetat', 'Ca - Azetat', 'Ca/Acetat', 'Ca- Acetat', 'Calcium-Acetat', 'calciumacetat', 'Sevelamer/ Renagel', 'Fosrenol', 'Phopsphocaps', 'Phosphocaps', 'Chloridhydroxid', 'Chloridhydroxid/Phosphonorm', 'Calcifriat', 'Phoscaps', 'Ca-Acetat', 'Sevelamer / Renagel', 'Sevelamer/Renagel', 'Lanthan / Fosrenol', 'Sevelamer', 'Phosphat', 'Phosphat / Phoscap', 'Calcium Acetat', 'Calcium Azetat', 'Renagel', 'Calciumacetat 400mg', 'Calcium / Calcium-Acetat', 'Phosphat / Phoscaps', 'Calcium / Ca-Acetat'],
        'Bone Density Agent': ['Bisphosphonat (Alendronat)', 'Alendronat/Fosamax', 'AlendronsÃ¤ure (Fosamax)', 'Alendronate', 'Alendronat', 'Alendromate', 'Alendromat', 'actonel', 'Bonivia', 'Bonviva 1 X / 3 MOIS', 'IbandronsÃ¤ure / Bonviva', 'Pamidronat', 'Pamidronasacidum', 'Pamidronat / Aredia', 'ZoledronsÃ¤ure', 'acide zolodronique (Zeneta)', 'Teriparatid (Forsteo)', 'Prolia', 'Denosumab (Prolia)', 'Bisphosphonat', 'Fosomab (Fosomax)', 'Acidum ibandronicum ut natrii ibandronas hydricus / Bonviva', 'Denosumab'],
        'GLP-1 Receptor Agonist': ['GLP-1-Rezeptor-Agonisten', 'GLP Analogon', 'Injectable antidiabetic', 'Semaglutide', 'Semagluti', 'ozempic', 'Ozempic (semaglutid)', 'Dulaglutid', 'liraglutide (Victoza)', 'Victtoza', 'antidiabetic (victosa ) S / C', 'Semaglutid', 'semaglutid', 'Liraglutid', 'Trulicity', 'Ozempic', 'Liraglutide', 'Semaglutid s.c.', 'liraglutid', 'Semaglutid (Ozempic)', 'Liraglutid (Victoza)', 'Dulaglutide', 'Dulaglutid (Trulicity)', 'Victoza', 'Glucagon-like-peptide 1', 'Dulaglutid s.c.'],
        'Bicarbonate Supplement': ['Hydrogencarbonas', 'Natrii hydrogencarbonas', 'sodiumbicarbonate', 'Natriumhydrogencarbonat', 'sodium bicarbonat', 'Natriumhydrogencarbonat /Nephrotrans', 'Natrii hydrogencarbonas / Nephrotrans', 'nephrotrans', 'Nephrotrans 500mg', 'Nephrotrans', 'Na-hydrogencarbonat / Nephrotrans', 'Natriumbicarbonat 1.2g', 'oral Bicarbonate: Nephrotrane', 'Natriumbicarbonat', 'Natrii hydrocarbonas/Nephrotrans', 'Sodiumbicarbonat (Nephrotrans)', 'Natrii hydrogemcarbonans'],
        'Anxiolytic/Sedative': ['Anxiolit', 'Diazepam (Valium)', 'Diazepam/Valium', 'Oxazepamum.', 'Oxazepamum', 'Flunitrazepamum', 'Bromazepamum', 'Bromazepanum', 'Bromazepamum / Lexotanil', 'Lorazepamum / Temesta', 'Lorazepnaum/Temesta', 'Xanax', 'Xamax', 'Tranxilium', 'Imovane', 'Chloraldurat', 'Auxolit 15 mg', 'Alprazolam', 'Lorazepamum', 'Temesta', 'Zolpidemtartrat', 'Oxazepam', 'Zolpidemi tartras', 'Zolpidem / Stilnox', 'Oxazepam / Seresta', 'Zolpidem', 'Urbanyl', 'Zolpidem/Stilnox'],
        'Antifungal': ['Nystatinum-Zinci oxidum', 'Nystatinum Zinci oxidum', 'Nystatinmu/Multilind', 'Nystatinum/Multillind', 'Nystatinum-Zinci oxidum/Multilind', 'Mycostatine (Nystatine) Alternatively', 'nystatin', 'Amphotericinum B. Excipiens.', 'Antimycosique', 'Sporanox', 'Anidulafungin', 'Mycolog topic', 'Mycostatine (Nystatine)', 'Nystatinum/Multilind', 'Amphotericin', 'Nystatinum / Multilind', 'Mycostatin', 'Amphotericin B', 'Nystatinum/7Multilind'],
        'Uric Acid Reducer': ['Allopurinol', 'Alopurinol', 'Allopurinl', 'Allopupinol', 'Allopurinnol', "all'opur", 'Zylolic', 'Allopurinol / Zyloric', 'Allopurinol (Urikostatikum)', 'Adenuric', 'Aduneric', 'AdÃ©nuric', 'AdÃ©nuric.', 'Adenuric Febuxostat', 'fasturtec iv (gut crisis)', 'Allopurinolum', 'Zyloric', 'allopurinol', 'Febuxostat', 'Zyloric (Allopurinol)', 'ADENURIC', 'Allopur', 'FASTURTEC', 'AdenÃ¼ric', 'Urikostatika'],
        'Laxative': ['Macrogolum', 'Macrogolum  / Transipeg', 'Macrogol, NaCl, Na-sulfat, Natriumhydrogencarbonat /  Transipeg', 'Macrogol, NaCl, Na-sulfat, KCI, Natriumhydrogencarbonat / Transipeg', 'Lactitol', 'Lactitolum monohydricum', 'Metamucil', 'Natriumpicosulphat / Laxoberon', 'Natriumpicosulfat/Laxoberon', 'Natriumpicosulfat / Laxoberon', 'Macrogolum/Transipeg', 'Natriumpicosulfat'],
        'Antiparkinson Agent': ['Levodopa Benserazid (Madopar)', 'Levodopa + Benserazid (Madopar)', 'dopar (Levodopa+Benserazid)', 'Madorpar', 'Levodopa + Carbidopa + Entacapon (Stalevo)', 'Lerodopa', 'Ropinirol', 'Pramipexolum', 'Pramipexal', 'Pramiprexol', 'Adartrel', 'Sifroc', 'Bromocriptine', 'Madopar', 'Levodopum', 'Pramipexol', 'Sifrol', 'Levodopa (Antiparkinson)', 'Levodopa'],
        'Cholesterol Absorption Inhibitor': ['antihyperlipidemia ttt other than statin (ezetrol)', 'ttt hyperlipidemia other than statin (Ezetrol)', 'Cholesterinabsorptionshemmer', 'Ezetimib.', 'Ezetemib', 'Zertimib', 'azetidone', 'EZETROL', 'Ezetimib', 'Ezetrol', 'Ezetimibe', 'Ezetrol (other antihyperlipidemic ttt)', 'Ezetrol (antihyperlipidemia ttt other than statin)'],
        'Analgesic (Non-Opioid)': ['Paracetamolum/Dafalgan', 'Paracetamol / Dafalgan', 'Paracetamolium/Dafalgan', 'Pyrazolunderivat', 'Pyrazolonderivat/ Novalgin', 'Metamizol / Novalgin', 'Paracetamolum', 'Paracetamolum / Dafalgan', 'Mephanol', 'Pyrazolonderivat/Novalgin', 'Pyrazolonderivat / Novalgin', 'Pyrazolonderivat', 'Nopilforte', 'Paracetamolum/Daflagan'],
        'ARB/Diuretic Combo': ['ARB + diuretic (erdabyclore) et diuretic (torem)', 'Candesartanum cilexetilum, Hydrochlorothiazidum / Atacand plus', 'diuretic (micardis plus)', 'micardis plus (plus diuretic)', 'diuretic (co-aprovel)', 'Votum plus (ARB- Duiretic)', 'Sevikar HCT', 'diuretic (voltum plus)', 'diuretic (edarbyclore)', 'ARB + diuretic (erdabyclore)'],
        'Anticholinergic (Bladder)': ['vesicare', 'Vesicare', 'Solifenacin / Vesicare', 'Trospiumchlorid / Spasmo-Urgenin Neo', 'Trospium chlorid / Spasmo-Urgenin', 'Trospiumchlorid/Spasmo Urgenin', 'Detrusitol', 'Toviaz', 'Toviaz 4mg/tgl', 'Trospiumchlorid', 'Spasmourgenin', 'Trospiumchlorid/Spasmourgenin', 'Fesoterodin', 'Trospiumchlorid / Spasmo-Urgenin'],
        'Renin-Inhibitor': ['Renin-Inhibitor', 'Renininhibitor', 'Reninhemmer', 'Reninrezeptor-Blocker(Rasilez)', 'RenÃ¯n antagonist', 'Aliskiren', 'Aliskiren (Renin Hemmer)', 'Rasilez', 'Rasilez (Aliskirenum, inhibiteur de la rÃ©nine)', 'Rasilez (AliskirÃ¨ne)', 'aliskiren (Antihypertonikum)', 'Aliskirenum', 'Aliskiron', 'Renin-Inhibitor (Rasilez)'],
        'Opioid Analgesic': ['Fentanyl', 'Analgeticum/Fentanyl', 'Opium/Ventanyl', 'Tramadolhydrochlorid', 'Tramadolhydrochlorid / Tramal', 'Tramacol', 'methadone', 'Morphin (Sevre-Long)', 'Hydromorphon', 'Palladon', 'Oxycodon hydrochlorid', 'Targiu', 'Buprenorphin / Temgesic', 'Methadon', 'Morphin', 'Temgesic', 'Methadone'],
        'Antiemetic': ['Domperidon', 'Domperidon/Motilium', 'Metoclopramid (HCI)', 'Metocolopramidi hydrochlorid', 'Metoclopramid / Paspertin', 'Metocroplamidi / Paspertin', 'Tropisetron', 'Tropisetron/Navoban', 'Metoclopramidi/Paspertin', 'Domperidon / Motilium', 'Metoclopramidi hydrochlorid', 'Metoclopramid'],
        'Iron Supplement': ['Ferrum', 'Ferrum / Venofer', 'Venofer', 'ferinject', 'Ferinject', 'perfusion Ferinject', 'ferrinject i/v', 'Eisen / Ferinject', 'Ferinjekt/Pentamide', 'Eisencarboxymaltose', 'Maltofer', 'Eisen / Venofer', 'Ferinject 500mg', 'Perfusion Ferinject', 'Ferrum/Venofer'],
        'Antithyroid Agent': ['Neo-mercazole', 'neomercazole', 'nÃ©o-mercazole', 'Neomercazol: Basedow tt', 'Carbimazol / Neo Mercazol', 'Carbimazol / Neo-Mercazole', 'Thyrozol (hyperthyroidie )', 'Neomercazole', 'carbimazolum', 'neo-mercazole', 'Neo-Mercazole', 'Neomercazol'],
        'Calcimimetic': ['Cinacalcet', 'Cinacalvet', 'Cinacalcef', 'Cinacalcet.', 'Cinacalcet)', 'Cinacalcet/Mimpara', 'Mimpara/cinacalcet', 'Mimpara', 'cinacalcet', 'Cinacalcet / Mimpara', 'Cinacalcet (Mimpara)', 'Mimpara (Cinacalcet)', 'Cinacalcat/Mimpara'],
        'Antiparasitic': ['Nivaquine', 'Mephaquin (MÃ©floquine)', 'Pyrimethaminum/Dapson', 'pyrimÃ©thamine', 'Albendazolum', 'Albendazol', 'Hydroxychloroquin (Plaquenil)', 'PlaquÃ©nil', 'Pentamidine', 'Pyrimethamin', 'Pentamidin'],
        'Hormonal Therapy (Oncology)': ['Arimidex', 'Goserelin', 'Goserelin/Zoladex', 'Tamoxifenum', 'Tamixifencitrat', 'Fulvestrant', 'Firmagon', 'AROMASIN', 'taitement for cancer prostate', 'Tamoxifen', 'Letrozol', 'Zoladex'],
        'Electrolyte Supplement': ['KCL DragÃ©es', 'KCL Dragees', 'Kalium Hausmann', 'Laktat/Citrat /Magnegon', 'Mg / Magnesium Diasporal', 'Mg / Magnesiocard', 'Natrium Chlorid', 'Magnesiocard', 'Kalium', 'Magnesium'],
        'Antihistamine': ['Clemastin', 'Clemastin/Tavegyl', 'Clemastine/Tavegyl', 'Antihystaminikum / Tavegyl', 'Citrizinum/Cetrin', 'Ceterizinum/Cetrin', 'Xyzal', 'Hydroxycin HCI', 'Clemastin / Tavegyl', 'Telfast'],
        'Hormonal Agent': ['Estriol', 'Ovestin Ovula', 'Estrogen Substitution', 'Testosteron', 'Testosteron-undecylat', 'contraceptives', 'Anticonceptivum', 'Visanne', 'Testoviron', 'Adrenocorticotropin'],
        'Vitamin Supplement (B12)': ['Vitamin B12', 'Cyanocobalaminum', 'Cianocobalaminum', 'Cyanoccobalaminum/Vitarubin', 'Vitarubin', 'Vitarubine', 'Vit. B12', 'Cyanocobalaminum/Vitarubin', 'Vit. B 12'],
        'Sympathomimetic': ['Sympathomimetika', 'Dobutaminum', 'Dobutaminum / Dobutrex', 'Dobutamin /Dobutrex', 'Adrenalin', 'Ephedrin', 'Midodrin / Gutron', 'Simdax', 'Levosimendan'],
        'Dermatological Agent': ['Neotigason', 'Acitretin / Neotigason', 'acitretin', 'Isotretinoin', 'Isotretinoine', 'Priorin', 'Prionin', 'Acicutan', 'Aldara Creme (Imiquimod)'],
        'Vitamin Supplement (B-Complex)': ['Vit B1,2,6/C/FolsÃ¤ure', 'Vit B1,2,6/C/FolsÃ¤ure / Dialvit', 'Vit.B Komplex / Becotal', 'Vit B1,2,6/C/FolsÃ¤ure/Dialvit'],
        'Antispasmodic': ['Pinaveriumbromid', 'Scopolamini butylbromidi', 'Scopolamini butylbromidum', 'Scopolamini butylbromidum/Buscopan', 'anticholinergics'],
        'Hepatobiliary Agent': ['acidum ursodeoxycholicum', 'Acidum ursodeoxycholicum', 'Ursodeoxycholic acid', 'Acidum ursodesoxycholicum'],
        'CCB': ['Amlodipinum', 'Amlodipin', 'Lercanidipini hydrochloridum', 'Lercanidipini', 'calcium antagonist', 'Adalat', 'Amloclipine'],
        'Anticoagulant': ['Liquemine', 'Calciparine', 'Apixaban', 'Eliquis', 'Eliquin', 'Fondaparinux-Natrium', 'Rivaroxaban', 'Sintrom'],
        'Heart Rate Reducer (If Channel Inhibitor)': ['Ivabradine', 'Procoralan (Ivabradin)', 'Ivabradine (Procoralan)', 'Procoralan'],
        'SGLT2 Inhibitor': ['SGLT2 Inhibitor', 'Jordiance', 'jasdiance', 'Daplagifozine', 'Forxiga', 'Jardiance', 'Empagliflozine'],
        'Antihypertensive (unspecified)': ['Antihypertensiva', 'Antihypertensive drugs', 'Hypertension Drug', 'Antihypertonikum'],
        'PCSK9 inhibitor': ['lipid-lowering agent (pravulent)', 'Alirocumab', 'Alirocumab (praluent)', 'Alirocumab (Praluent)'],
        'PDE5 Inhibitor': ['Tadalafil (Cialis)', 'Tadalafil', 'Tadala', 'Adcurca', 'sildenafil', 'Sildenofil', 'Sildenafil'],
        'Beta-blocker': ['Beta blocker', 'Beta-Blocker i.v.', 'Propranololi hydrochl/Inderal', 'Labetalol', 'Atenololum'],
        'Anti-gout Agent': ['Colchicin', 'Colchicine.', 'Colchin', 'Colchine', 'Colchicine', 'Colchizin', 'Cholchicin'],
        'Angiotensin Receptor-Neprilysin Inhibitor (ARNI)': ['Entresto', 'Sacubitril (Entresto)', 'Salubitril'],
        'Metabolic Disorder Agent': ['Cystagon', 'CYSTAGON', 'Cystagon/ Cystamin', 'Cysteamin', 'Cysteamine'],
        'Immunostimulant': ['Immunostimulants', 'Urovaxom', 'Bronchovaxom', 'immunstimulantic drug E.coli'],
        'Potassium Binder': ['sorbisterit', 'Polystyrolsulfonat', 'Sulfoniertes Kopolymer', 'Kopolymer'],
        'Vitamin Supplement (B9)': ['Acidum folicum', 'folic acid', 'calcium folinat', 'Acide Falique'],
        'ARB': ['Candesartanum cilexetilum', 'Candesartan', 'Olmesartanmedoxomil', 'Losartan-Kalium'],
        'Vitamin Supplement (K)': ['VItamin K/ Konakion', 'Phytomenadion/Konakion', 'Phytomenadion'],
        'Leukotriene Antagonist': ['Leukotriene antagonists (antiasthmatic)', 'Singulair', 'Lukair'],
        'Statin': ['Fluvastatin/Lescol', 'Fluvastatin', 'Fluvastatinum', 'Atorvastatin'],
        'Pancreatic Enzymes': ['Pancreatin', 'Creon', 'Pancreatis pulvis', 'Pankreatin'],
        'ACE-Inhibitor': ['Lisinoprilum', 'Captoprilum', 'Ramiprilum', 'Lisoprinolum'],
        '5-alpha-reductase Inhibitor': ['Finasterid / Proscar', 'Proscar', 'Avodart'],
        'Vitamin Supplement (B6)': ['Vitamin B6', 'Vit B6', 'Pyridoxin'],
        'Antiplatelet': ['Plavix', 'BRILIQUE', 'Efient', 'Clopidogrel'],
        'Vaccine': ['Vaccin H1N1', 'Meniingokokken-Impfung (Menveo)'],
        'Nootropic': ['Piracetam', 'Limptar', 'Tebokan', 'Myrtaven'],
        'Anticholinergic (Ophthalmic)': ['Scopolamin (Augentropfen)'],
        'DPP-4 Inhibitor': ['Linagliptin', 'Trajenta', 'liaglipid'],
        'Antidiabetic (Oral, unspecified)': ['Orale antidiabetica'],
        'Antidiabetic (Combination)': ['Janumet', 'iardiance-Met'],
        'Hemorheologic Agent': ['Pentoxifylin', 'Pentoxifyllin'],
        'Joint Health Supplement': ['Condroxulf', 'Condrosulf'],
        'Colony-Stimulating Factor': ['Neupogen', 'Filgrastim'],
        'Probiotic': ['Saccharomyces boulardii lyophilisiert'],
        'NSAID': ['Diclofenac / Voltaren', 'Ibuprofen/Brufen'],
        'Antiflatulent': ['Simeticon / Flatulex', 'Simeticon'],
        'Antimigraine Agent': ['Zolmitriptan', 'Zolmitripan'],
        'Amino Acid Supplement': ['Methionin / Acimethin'],
        'Cholinesterase Inhibitor': ['Exelon', 'Mestinon'],
        'CNS Stimulant': ['Ritaline (MÃ©thylphÃ©nidate)'],
        'Muscle Relaxant': ['Tizanidinum', 'Tizanidium'],
        'H2 Receptor Blocker': ['Zantic', 'Ranitidinum'],
        'Fibrinogen Concentrate': ['Haemocomplettan'],
        'Vitamin Supplement (B3)': ['Nicotinamide'],
        'Smoking Cessation Aid': ['Nicotin patch'],
        'Herbal/Dietary Supplement': ['Cimifemin'],
        'Mucolytic/Antidote': ['Acetylcystein'],
        'Cholinergic agonist': ['Bethanechol'],
        'Mineral Supplement': ['Zink Verla'],
        'Anti-vertigo Agent': ['Betaserc'],
        'Opioid Antagonist': ['Naloxon'],
        'Vasopressor': ['Noradrenalin'],
        'Sulfonylurea': ['Diamicron'],
        'Anti-anginal': ['Ranolazin'],
        'Antidiarrheal': ['Imodium'],
        'Biguanide': ['Methformin'],
        'Antacid': ['Alucol'],
        'Insulin': ['Lantus'],
        'Other': ['other', 'Hormono-radiotherapy', 'Antirheumatikum', 'Galaktose', 'Geon', 'Tyuesyurt', 'Ulovaxon', 'Acide Fatigue', 'Carmethin', 'Prolin', 'Zytorix', 'Acidium', 'Natriumthiosulfat'],
        'Unknown': ['Unknown'],
        'Not applicable': ['Not applicable'],
    }

    # Drug normalization map built as key = drug -> value = category
    DRUG_NORMALIZATION_MAP = {k: v for v, ks in INVERSE_DRUG_NORMALIZATION_MAP.items() for k in ks}

    # Coarser mapping for drug categories (useful? or to modify?)
    INVERSE_DRUG_CATEGORY_MAP = {
        # --- Core Infection & Immunity-Related Categories ---
        'Immunosuppressant': ['Immunosuppressant', 'Biologic/Immunosuppressant', 'Biologic Agent'],
        'Corticosteroid': ['Corticosteroid'],
        'Chemotherapy/targeted therapy': ['Chemotherapy Agent', 'Targeted Therapy Agent', 'Chemotherapy/Targeted Therapy'],
        'Myeloid growth factor': ['Colony-Stimulating Factor'],
        'Immunoglobulin therapy': ['Immunoglobulin Therapy (IVIG)'],
        'Immunostimulant': ['Immunostimulant'],
        'Antibiotic': ['Antibiotic'],
        'Antifungal': ['Antifungal'],
        'Antiviral': ['Antiviral'],
        'Antiparasitic': ['Antiparasitic'],
        'Vaccine': ['Vaccine'],

        # --- Common Comorbidity & Transplant Management Categories ---
        'Antihypertensive': ['Diuretic', 'ARB/Diuretic Combo', 'ARB', 'ACE-Inhibitor', 'Beta-blocker', 'CCB', 'Alpha-blocker', 'Renin-Inhibitor', 'Antihypertensive (unspecified)', 'Angiotensin Receptor-Neprilysin Inhibitor (ARNI)'],
        'Antidiabetic': ['Biguanide', 'Sulfonylurea', 'DPP-4 Inhibitor', 'SGLT2 Inhibitor', 'GLP-1 Receptor Agonist', 'Antidiabetic (Oral, unspecified)', 'Antidiabetic (Combination)', 'Insulin'],
        'Antithrombotic agent': ['Anticoagulant', 'Antiplatelet'],
        'Analgesic': ['Opioid Analgesic', 'Analgesic (Non-Opioid)', 'NSAID'],
        'Gastrointestinal agent': ['Proton Pump Inhibitor (PPI)', 'H2 Receptor Blocker', 'Laxative', 'Antiemetic', 'Antidiarrheal', 'Antacid', 'Antiflatulent', 'Pancreatic Enzymes', 'Antispasmodic'],
        'Lipid-Lowering agent': ['Statin', 'PCSK9 inhibitor', 'Cholesterol Absorption Inhibitor'],
        'Electrolyte/mineral agent': ['Electrolyte Supplement', 'Mineral Supplement', 'Bicarbonate Supplement', 'Iron Supplement', 'Potassium Binder', 'Phosphate Binder'],
        'Bone/urate metabolism agent': ['Bone Density Agent', 'Calcimimetic', 'Uric Acid Reducer', 'Anti-gout Agent'],
        'Supplement': ['Vitamin Supplement (B6)', 'Vitamin Supplement (B9)', 'Vitamin Supplement (B12)', 'Vitamin Supplement (B3)', 'Vitamin Supplement (B-Complex)', 'Vitamin Supplement (K)', 'Calcium Supplement', 'Vitamin D Supplement', 'Calcium/Vitamin D Supplement', 'Herbal/Dietary Supplement', 'Joint Health Supplement', 'Probiotic', 'Amino Acid Supplement'],
            
        # --- Other Functional Categories ---
        'Cardiovascular agent': ['Vasopressor', 'Vasodilator', 'Heart Rate Reducer (If Channel Inhibitor)', 'PDE5 Inhibitor', 'Anti-anginal', 'Antiarrhythmic'],
        'Neurological agent': ['Anticonvulsant', 'Muscle Relaxant', 'Antiparkinson Agent', 'Cholinesterase Inhibitor', 'Antimigraine Agent', 'Cholinergic agonist'],
        'Psychiatric agent': ['Antidepressant', 'Antipsychotic', 'Anxiolytic/Sedative', 'CNS Stimulant', 'Nootropic', 'Smoking Cessation Aid'],
        'Hormonal therapy': ['Hormonal Therapy (Oncology)', 'Hormonal Agent', 'Thyroid Hormone', 'Antithyroid Agent', '5-alpha-reductase Inhibitor'],
        'Respiratory agent': ['Inhaled Corticosteroid/Bronchodilator', 'Leukotriene Antagonist', 'Antihistamine', 'Mucolytic/Antidote'],
        'Hematologic agent': ['Erythropoiesis-Stimulating Agent (ESA)', 'Fibrinogen Concentrate', 'Hemorheologic Agent'],

        # --- Merged, Procedural & Miscellaneous Categories ---
        'Centrally Acting Agent': ['Centrally Acting Agent'],
        'Anticholinergic': ['Anticholinergic (Bladder)', 'Anticholinergic (Ophthalmic)'],
        'Medical/therapeutic procedure': ['Medical/Therapeutic Procedure'],
        'Other/miscellaneous': ['Other', 'Metabolic Disorder Agent', 'Dermatological Agent', 'Hepatobiliary Agent', 'Sympathomimetic', 'Anti-vertigo Agent', 'Opioid Antagonist'],
        'Unknown': ['Unknown'],
        'Not applicable': ['Not applicable'],
    }
    
    # Drug category coarsing map built as key = category -> value = coarse category
    DRUG_CATEGORY_MAP = {k: v for v, ks in INVERSE_DRUG_CATEGORY_MAP.items() for k in ks}

    # Use coarse mapping to do a drug mapping as key = drug -> value = coarse category
    DRUG_COARSE_NORMALIZATION_MAP = {}
    for drug, cat in DRUG_NORMALIZATION_MAP.items():
        coarse_cat = DRUG_CATEGORY_MAP[cat]
        DRUG_COARSE_NORMALIZATION_MAP[drug] = coarse_cat
