import pandas as pd
import constants
from typing import Union
from warnings import warn

csts = constants.ConstantsNamespace()


def select_valid_values(
    data: pd.DataFrame,
    valid_mask: pd.Series,
    patient_ID: int,
    value_key: str,
) -> pd.DataFrame:
    """ Warn about invalid entries and return only valid rows
    """
    invalid = data.loc[~valid_mask & data[value_key].notna(), value_key].unique()
    if invalid.size:
        warn(f"Invalid value(s) {invalid} in column {value_key} for patient {patient_ID} ignored.")
    return data.loc[valid_mask]


def get_date_by_key(
    patient_ID:int,
    data:pd.DataFrame,
    value_key:str,
    valid_range:tuple[pd.Timestamp, pd.Timestamp]=csts.VALID_DATE_RANGE,
    nan_like_values:tuple[Union[str, pd.Timestamp], ...]=csts.NAN_LIKE_DATES,
) -> pd.Timestamp:
    """ Get date(s) by key if in the valid range
    """
    # Filter relevant patient data
    dates = data.loc[data["patid"] == patient_ID, [value_key]]  # .copy()
    
    # Handle any nan-like entries with pd.NaT
    dates[value_key] = dates[value_key].where(~dates[value_key].isin(nan_like_values), other=pd.NaT)
    dates[value_key] = pd.to_datetime(dates[value_key], errors="coerce")

    # Return valid entries only (i.e., dates within the valid_range)
    valid_mask = dates[value_key].between(*valid_range)
    return select_valid_values(dates, valid_mask, patient_ID, value_key)


def get_categorical_feature_by_key(
    patient_ID:int,
    data:pd.DataFrame,
    value_key:str,
    valid_categories:list[str]|None=None,
    nan_like_values:tuple[str, ...]=csts.NAN_LIKE_CATEGORIES,
) -> pd.DataFrame:
    """ Get a categorical feature by key if it is in the valid categories
    """
    # Filter relevant patient data
    feats = data.loc[data["patid"] == patient_ID, [value_key]]  # .copy()

    # Replace nan-like entries with pd.NA only if they are not in the valid list
    feats[value_key] = feats[value_key].where(~feats[value_key].isin(nan_like_values), other=pd.NA)

    # Return valid entries only (i.e., entries that are in valid_categories)
    if valid_categories is None: return feats
    valid_mask = feats[value_key].isin(valid_categories)
    return select_valid_values(feats, valid_mask, patient_ID, value_key)


def get_numerical_feature_by_key(
    patient_ID:int,
    data:pd.DataFrame,
    value_key:str,
    valid_range:tuple[Union[int, float], Union[int, float]]|None=None,
    nan_like_values:tuple[Union[str, float], ...]=csts.NAN_LIKE_NUMBERS,
) -> pd.DataFrame:
    """ Get a numerical feature by key if it is in the valid range
    """
    # Select given patient feature value(s)
    feats = data.loc[data["patid"] == patient_ID, [value_key]]  # .copy()
    
    # Handle nan-like entries
    feats[value_key] = feats[value_key].where(~feats[value_key].isin(nan_like_values), other=pd.NA)
    feats[value_key] = pd.to_numeric(feats[value_key], errors="coerce")
    
    # Return valid entries only (i.e., entries that are inside the valid range)
    if valid_range is None: return feats
    valid_mask = feats[value_key].between(*valid_range)
    return select_valid_values(feats, valid_mask, patient_ID, value_key)


def clean_time_value_pairs(
    data: pd.DataFrame,
    value_type: str,
    valid_time_range: tuple[pd.Timestamp, pd.Timestamp]=csts.VALID_DATE_RANGE,
    valid_number_range: tuple[Union[int, float], Union[int, float]]|None=None,
    valid_categories: list[str]|None=None,
    nan_like_times: tuple[pd.Timestamp, ...]=csts.NAN_LIKE_DATES,
    nan_like_numbers: tuple[Union[str, float], ...]=csts.NAN_LIKE_NUMBERS,
    nan_like_categories: tuple[Union[str, float], ...]=csts.NAN_LIKE_CATEGORIES,
) -> pd.DataFrame:
    """ Clean longitudinal data by removing invalid entries
    """
    # Handle nan-like times and times outside the valid ranges
    data["time"] = data["time"].where(~data["time"].isin(nan_like_times), other=pd.NaT)
    data["time"] = data["time"].where(data["time"].between(*valid_time_range), other=pd.NaT)
    data["time"] = pd.to_datetime(data["time"], errors="coerce")

    # Handle nan-like values and values that are invalid / outside the valid ranges
    if value_type == "numerical":
        if valid_number_range is not None:
            data["value"] = data["value"].where(data["value"].between(*valid_number_range), other=pd.NA)
        data["value"] = data["value"].where(~data["value"].isin(nan_like_numbers), other=pd.NA)
    elif value_type == "categorical":
        if valid_categories is not None:
            data["value"] = data["value"].where(data["value"].isin(valid_categories), other=pd.NA)
        data["value"] = data["value"].where(~data["value"].isin(nan_like_categories), other=pd.NA)
    else:
        raise ValueError(f"Invalid value_type: {value_type}. Expected 'numerical' or 'categorical'.")
    
    # Return entries with valid values only, allowing for an NaT time if corresponding value is valid
    return data.dropna(subset=["value"]).reset_index(drop=True)


def get_time_value_pairs(
    patient_ID:int,
    data:pd.DataFrame,
    time_key:str,
    value_key:str,
    value_type:str="categorical",
    nan_like_times:tuple[pd.Timestamp, ...]=csts.NAN_LIKE_DATES,
    nan_like_numbers:tuple[Union[str, float], ...]=csts.NAN_LIKE_NUMBERS,
    nan_like_categories:tuple[Union[str, float], ...]=csts.NAN_LIKE_CATEGORIES,
    valid_time_range:tuple[pd.Timestamp, pd.Timestamp]=csts.VALID_DATE_RANGE,
    valid_number_range:tuple[Union[int, float], Union[int, float]]|None=None,
    valid_categories:list[str]|None=None,
) -> pd.DataFrame:
    """ Get time-value pairs for a given patient, returning data pairs with
        columns "time" and "value", and "attribute" for the value key
    """
    # Select given patient feature value(s) and associated time(s)
    feats = data.loc[data["patid"] == patient_ID, [value_key, time_key]]  # .copy()
    feats = feats.rename(columns={time_key: "time", value_key: "value"})
    feats = feats.assign(attribute=value_key)

    # Clean the time-value pairs
    feats = clean_time_value_pairs(
        feats, value_type,
        valid_time_range, valid_number_range, valid_categories,
        nan_like_times, nan_like_numbers, nan_like_categories,
    )

    return feats


def get_longitudinal_data(
    patient_ID:int, 
    data:pd.DataFrame,
    time_key:str,
    value_key:str,
    value_type:str="categorical",
    attribute_key:str|None=None,
    nan_like_times:tuple[pd.Timestamp, ...]=csts.NAN_LIKE_DATES,
    nan_like_numbers:tuple[Union[str, float], ...]=csts.NAN_LIKE_NUMBERS,
    nan_like_categories:tuple[Union[str, float], ...]=csts.NAN_LIKE_CATEGORIES,
    valid_time_range:tuple[pd.Timestamp, pd.Timestamp]=csts.VALID_DATE_RANGE,
    valid_number_range:tuple[Union[int, float], Union[int, float]]|None=None,
    valid_categories:list[str]|None=None,
) -> pd.DataFrame:
    """ Get longitudinal data for a given patient ID, data key, and date key
        - Made for raw data with format "{value_key}_{i}" and "{time_key}_{i}"
        - Optionally, an attribute key can be provided to use different attributes
          of the same data key, instead of using the data key as the attribute
    """
    # Select all patient columns matching value_key and corresponding dates
    patient_df = data.loc[data["patid"] == patient_ID]
    value = patient_df.filter(regex=rf"^{value_key}_[0-9]+$")
    time = patient_df.filter(regex=rf"^{time_key}_[0-9]+$")
    if attribute_key is not None:
        attr = patient_df.filter(regex=rf"^{attribute_key}_[0-9]+$")

    # Rename columns to have matching variable names when melting them
    value.columns = [col.replace(f"{value_key}_", "") for col in value.columns]
    time.columns = [col.replace(f"{time_key}_", "") for col in time.columns]
    if attribute_key is not None:
        attr.columns = [col.replace(f"{attribute_key}_", "") for col in attr.columns]

    # Melt value and time to a "long" format and merge into two columns
    value_long = value.melt(value_name="value", var_name="var_id")
    time_long = time.melt(value_name="time", var_name="var_id")
    long_df = pd.concat([value_long, time_long], axis=1)
    if attribute_key is not None:
        attr_long = attr.melt(value_name="attribute", var_name="var_id")
        long_df = pd.concat([long_df, attr_long], axis=1)
    long_df = long_df.drop(columns="var_id")  # dropping the helper column

    # Clean the time-value pairs
    long_df = clean_time_value_pairs(
        long_df, value_type,
        valid_time_range, valid_number_range, valid_categories,
        nan_like_times, nan_like_numbers, nan_like_categories,
    )

    # Add attribute column as the value key if not queried from the attribute key
    if attribute_key is None:
        long_df = long_df.assign(attribute=value_key)
    
    return long_df


# def get_longitudinal_data_old(
#     patient_ID:int, 
#     data:pd.DataFrame,
#     value_key:str,
#     time_key:str,
#     attribute_key:str|None=None,
#     nan_like_values:tuple[Union[str, float], ...]=csts.NAN_LIKE_NUMBERS,
#     nan_like_times:tuple[pd.Timestamp, ...]=csts.NAN_LIKE_DATES,
#     valid_time_range:tuple[Union[int, float], Union[int, float]]=csts.VALID_DATE_RANGE,
# ) -> pd.DataFrame:
#     """ Get longitudinal data for a given patient ID, data key, and date key
#         - Made for raw data with format "{value_key}_{i}" and "{time_key}_{i}"
#         - Optionally, an attribute key can be provided to use different attributes
#           of the same data key, instead of using the data key as the attribute
#     """
#     # Select all patient columns matching value_key and corresponding dates
#     patient_df = data.loc[data["patid"] == patient_ID]
#     value_pattern = rf"^{value_key}_[0-9]+$"
#     time_pattern = rf"^{time_key}_[0-9]+$"
#     value = patient_df.filter(regex=value_pattern)
#     time = patient_df.filter(regex=time_pattern)
#     if attribute_key is not None:
#         attributes_pattern = rf"^{attribute_key}_[0-9]+$"
#         attr = patient_df.filter(regex=attributes_pattern)

#     # Rename columns to have matching variable names when melting them
#     value.columns = [col.replace(f"{value_key}_", "") for col in value.columns]
#     time.columns = [col.replace(f"{time_key}_", "") for col in time.columns]
#     if attribute_key is not None:
#         attr.columns = [col.replace(f"{attribute_key}_", "") for col in attr.columns]

#     # Melt value and time to a "long" format and merge into two columns
#     value_long = value.melt(value_name="value", var_name="var_id")
#     time_long = time.melt(value_name="time", var_name="var_id")
#     long_df = pd.concat([value_long, time_long], axis=1)
#     if attribute_key is not None:
#         attr_long = attr.melt(value_name="attribute", var_name="var_id")
#         long_df = pd.concat([long_df, attr_long], axis=1)

#     # Handle times outside the valid ranges, and rows without informative entries
#     long_df = long_df.drop(columns="var_id")
#     long_df.loc[~long_df["time"].between(*valid_time_range), "time"] = pd.NaT
#     long_df = long_df[~(long_df["value"].isin(nan_like_values) & long_df["time"].isin(nan_like_times))]
#     long_df = long_df.dropna(subset=["value", "time"], how="all")
#     long_df["time"] = pd.to_datetime(long_df["time"], errors="coerce")

#     # Format dataframe
#     long_df = long_df.reset_index(drop=True)
#     if attribute_key is None:
#         long_df = long_df.assign(attribute=value_key)

#     return long_df


class LongitudinalData:
    def __init__(self, value, date: pd.Timestamp):
        self.value = value
        self.date = date


def clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """ Remove unnecessary stat rows below patient info; only useful when using
        the small dict dataset
    """
    STAT_ROW_NAMES = ["MIN", "P1", "P25", "MEDIAN", "P75", "P99", "MAX", "MEAN", "MODE"]
    data = data[~data["_STAT_"].isin(STAT_ROW_NAMES)]  
    data = data.dropna(subset="patid")
    return data
