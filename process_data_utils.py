import os
import json
import pandas as pd
from typing import Union
from dataclasses import dataclass, asdict, field


class DataConfig:
    """ ...
    """
    DATA_DIR = "data"
    RAW_DATA_NAME = "SAS_Data_Dictionary_FUP226_raw-01Jan2023_v1.xlsx"
    RAW_DATA_PATH = os.path.join(DATA_DIR, RAW_DATA_NAME)
    PROCESSED_DATA_SUBDIR = os.path.join(DATA_DIR, "processed")
    STAT_ROW_NAMES = ["MIN", "P1", "P25", "MEDIAN", "P75", "P99", "MAX", "MEAN", "MODE"]
    FEAT_INFO_DICTS = {
        # Patient baseline information sheet
        "#4_PAT_BL": {
            "centreid": str,
            "bmi": float,
            # "birthday": pd.Timestamp,
        },
        
        # Patient socio-demographic questionnaire (?)
        "#5_PAT_PSQ": {
            "ethnicity": str,
        }
    }


@dataclass
class EAVRecord:
    """ Data class for any static value
    """
    entity: int  # broad range entity -> keep this?
    attribute: str  # attribute name (medication, infection status, etc.)
    value: Union[int, float, str]  # attribute value
    

@dataclass
class EAVRecordTimed(EAVRecord):
    """ Data class for any dynamic event
    """
    timestamp: float  # days after admission (negative values if before)


@dataclass
class PatientRecord:
    """ Data class holding patient ID, static and dynamic patient data
    """
    patient_id: int
    feature_list: list[EAVRecord] = field(default_factory=list)
    event_list: list[EAVRecordTimed] = field(default_factory=list)
    
    def __post_init__(self):
        """ Always save patient instance data to a json file
        """
        os.makedirs(DataConfig.PROCESSED_DATA_SUBDIR, exist_ok=True)
        json_name = f"patient_{self.patient_id}.json"
        self.json_path = os.path.join(DataConfig.PROCESSED_DATA_SUBDIR, json_name)
        self.save()
        
    def save(self) -> None:
        """ Save this patient to the json file specified by self.json_path
        """
        with open(self.json_path, "w", encoding="utf-8") as json_file:
            json.dump(asdict(self), json_file, indent=2)
            
    @classmethod
    def load(cls, json_path: str) -> "PatientRecord":
        """ Load a patient record from the json file given by json_path
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        return cls(
            patient_id=data["patient_id"],
            feature_list=[EAVRecord(**record) for record in data["feature_list"]],
            event_list=[EAVRecordTimed(**record) for record in data["event_list"]],
        )
    
    def add_element(
        self,
        entity: int,
        attribute: str,
        value: Union[int, float, str],
        timestamp: float|None=None,
    ) -> None:
        """ Add a static feature or a timed event to the patient data
        """
        if timestamp is None:
            self.feature_list.append(EAVRecord(entity, attribute, value))
        else:
            self.event_list.append(EAVRecordTimed(entity, attribute, value, timestamp))
        self.save()


if __name__ == "__main__":
    """ Test data classes
    """
    # Create a test patient data instance and save its data
    saved_patient = PatientRecord(patient_id=123)
    saved_patient.add_element(entity=1, attribute="age", value=42)
    saved_patient.add_element(entity=1, attribute="name", value="Alice")
    saved_patient.add_element(entity=2, attribute="bp-sys", value=120, timestamp=10.1)
    saved_patient.add_element(entity=2, attribute="bp-dia", value=125, timestamp=-5.5)
    
    # Load saved patient data into a patient record instance
    load_path = os.path.join(DataConfig.PROCESSED_DATA_SUBDIR, "patient_123.json")
    loaded_patient = PatientRecord.load(load_path)
    
    # Check if all went as expected
    if saved_patient == loaded_patient and saved_patient is not loaded_patient:
        print("Test patient successfully saved and loaded!")
    import ipdb; ipdb.set_trace()