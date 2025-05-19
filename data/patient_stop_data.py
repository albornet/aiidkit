import pandas as pd


def get_death_date(patient_ID:int, data:pd.DataFrame) -> pd.Timestamp:
    death_date = data.loc[data["patid"] == patient_ID]["deathdate"]
    if len(death_date) == 0:
        return pd.Timestamp(pd.NaT)
    return death_date.item()


def get_dropout_date(patient_ID:int, data:pd.DataFrame) -> pd.Timestamp:
    dropout_date = data.loc[data["patid"] == patient_ID]["dropoutdate"]
    if len(dropout_date) == 0:
        return pd.Timestamp(pd.NaT)
    return dropout_date.item()


def get_latest_date_known_to_be_alive(patient_ID:int, data:pd.DataFrame) -> pd.Timestamp:
    dropout_date = data.loc[data["patid"] == patient_ID]["lastalivedate"]
    if len(dropout_date) == 0:
        return pd.Timestamp(pd.NaT)
    return dropout_date.item()