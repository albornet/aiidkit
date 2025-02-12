import os
import pandas as pd
from typing import Any
from datetime import datetime
from dataclasses import dataclass


class ProcessDataConfig:
    RAW_DATA_DIR = "data"
    RAW_DATA_NAME = "SAS_Data_Dictionary_FUP226_raw-01Jan2023_v1.xlsx"
    RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, RAW_DATA_NAME)
    STAT_ROW_NAMES = ["MIN", "P1", "P25", "MEDIAN", "P75", "P99", "MAX", "MEAN", "MODE"]


@dataclass
class EAVRecord:
    entity_id: int  # Broad range entity
    attribute: str  # Attribute name (medication, infection status, etc.)
    value: Any  # Value of the attribute (could be int, float, str, etc.)


@dataclass
class EAVRecordTimed(EAVRecord):
    timestamp: pd.Timestamp  # or might map it to something more normalized


@dataclass
class PatientRecord:
    patient_id: int
    static_list: list[EAVRecord]
    dynamic_list: list[EAVRecordTimed]
