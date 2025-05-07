import pandas as pd
import numpy as np
from enum import Enum

# what is wrong with this import ?
from .data_utils import *


class Exit(Enum):
    STILL_INCLUDED = 0,
    DIED = 1,
    LOSS_TO_FOLLOW_UP = 2


class Gender(Enum):
    FEMALE = 0,
    MALE = 1,
    OTHER = 2


def get_patient_birthday(patient_ID:int, data: pd.DataFrame) -> pd.Timestamp:
    # TODO Check that it returns a valid date
    birthday = data.loc[data["patid"] == patient_ID]["birthday"]
    return birthday.item()


def get_patient_gender(patient_ID:int, data:pd.DataFrame) -> Gender:
    gender = data.loc[data["patid"] == patient_ID]["sex"]
    if gender.item() == "Female":
        return Gender.FEMALE
    elif gender.item() == "Male":
        return Gender.MALE
    return Gender.OTHER


def get_total_cholesterol(patient_ID:int, data:pd.DataFrame) -> float:
    total_cholesterol = data.loc[data["patid"] == patient_ID]["chol"]
    if len(total_cholesterol) == 0:
        return np.nan
    return total_cholesterol.item()


def get_weight(patient_ID:int, data:pd.DataFrame) -> float:
    weight = data.loc[data["patid"] == patient_ID]["weight"]
    if len(weight) == 0:
        return np.nan
    return weight.item()


def get_height(patient_ID:int, data:pd.DataFrame) -> float:
    height = data.loc[data["patid"] == patient_ID]["height"]
    if len(height) == 0:
        return np.nan
    return height.item()


def get_hba1c(patient_ID:int, data:pd.DataFrame) -> float:
    hba1c = data.loc[data["patid"] == patient_ID]["hba1c"]
    if len(hba1c) == 0:
        return np.nan
    return hba1c.item()


def get_hba1c_date(patient_ID:int, data:pd.DataFrame) -> pd.Timestamp:
    hba1c_date = data.loc[data["patid"] == patient_ID]["hba1cdate"]
    return hba1c_date.item()


def get_exit_status(patient_ID:int, data:pd.DataFrame) -> Exit:
    exit_status = data.loc[data["patid"] == patient_ID]["exit"]

    if exit_status.item() == "Still included":
        return Exit.STILL_INCLUDED
    elif exit_status.item() == "Died":
        return Exit.DIED
    elif exit_status.item() == "Loss to follow-up":
        return Exit.LOSS_TO_FOLLOW_UP
    return Exit.LOSS_TO_FOLLOW_UP


class InfectionDisease(LongitudinalData):
    def __init__(self, value:str, date: pd.Timestamp):
        super().__init__(value, date)


def get_infectious_diseases(patient_ID:int, data:pd.DataFrame) -> List[InfectionDisease]:
    infectious_diseases = get_longitudinal_data(patient_ID=patient_ID,data_name="id", 
                          data=data, subclass=InfectionDisease)
    return infectious_diseases 