import os
import json
from typing import Union
from dataclasses import dataclass, asdict


class ProcessDataConfig:
    """ ...
    """
    DATA_DIR = "data"
    RAW_DATA_NAME = "SAS_Data_Dictionary_FUP226_raw-01Jan2023_v1.xlsx"
    RAW_DATA_PATH = os.path.join(DATA_DIR, RAW_DATA_NAME)
    PROCESSED_DATA_SUBDIR = os.path.join(DATA_DIR, "processed")
    STAT_ROW_NAMES = ["MIN", "P1", "P25", "MEDIAN", "P75", "P99", "MAX", "MEAN", "MODE"]


@dataclass
class EAVRecord:
    """ Data class for any static value
    """
    entity_id: int  # broad range entity
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
    static_list: list[EAVRecord]
    dynamic_list: list[EAVRecordTimed]
    
    def __post_init__(self):
        """ Always save patient instance data to a json file
        """
        json_name = f"patient_{self.patient_id}.json"
        os.makedirs(ProcessDataConfig.PROCESSED_DATA_SUBDIR, exist_ok=True)
        self.json_path = os.path.join(ProcessDataConfig.PROCESSED_DATA_SUBDIR, json_name)
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
            static_list=[EAVRecord(**record) for record in data["static_list"]],
            dynamic_list=[EAVRecordTimed(**record) for record in data["dynamic_list"]],
        )


if __name__ == "__main__":
    """ Test data classes
    """
    # Create a test patient data instance and save its data
    saved_patient = PatientRecord(
        patient_id=123,
        static_list=[
            EAVRecord(entity_id=1, attribute="age", value=42),
            EAVRecord(entity_id=1, attribute="name", value="Alice"),
        ],
        dynamic_list=[
            EAVRecordTimed(entity_id=2, attribute="bp-sys", value=120, timestamp=10.1),
            EAVRecordTimed(entity_id=2, attribute="bp-dia", value=125, timestamp=-5.5),
        ],
    )
    
    # Load saved patient data into a patient record instance
    loaded_patient = PatientRecord.load("data/processed/patient_123.json")
    if saved_patient == loaded_patient and saved_patient is not loaded_patient:
        print("Test patient successfully saved and loaded!")
    import ipdb; ipdb.set_trace()
