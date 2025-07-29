import re
from enum import Enum
from typing import Any
from datetime import datetime
import pandas as pd
from typing import Union
from warnings import warn

import src.constants as constants
csts = constants.ConstantsNamespace()


def concatenate_clinical_information(
    list_of_dfs: list[pd.DataFrame],
) -> pd.DataFrame:
    """ Concatenate (row-wise) a list of dataframes containing clinical inforamtion
    """
    to_concat = [df for df in list_of_dfs if not df.empty]
    if len(to_concat) == 0:
        return pd.DataFrame()
    elif len(to_concat) == 1:
        return to_concat[0]
    return pd.concat(to_concat, ignore_index=True)


def get_valid_categories(
    df: pd.DataFrame,
    key: str,
) -> list[str]:
    """ Get all valid categories of a given column
    """
    category_index = df[key].value_counts().index
    valid_categories = category_index[~category_index.isin(csts.NAN_LIKE_CATEGORIES)]
    return valid_categories.tolist()


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
    patient_ID: int,
    data: pd.DataFrame,
    value_key: str,
    valid_range: tuple[pd.Timestamp, pd.Timestamp]=csts.VALID_DATE_RANGE,
    nan_like_values: tuple[Union[str, pd.Timestamp], ...]=csts.NAN_LIKE_DATES,
) -> pd.DataFrame:
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
    patient_ID: int,
    data: pd.DataFrame,
    value_key: str,
    valid_categories: list[str]|None=None,
    nan_like_values: tuple[str, ...]=csts.NAN_LIKE_CATEGORIES,
    context_key: str|None=None,
) -> pd.DataFrame:
    """ Get a categorical feature by key if it is in the valid categories
        TODO: ADD df.melt(var_name="attribute", value_name="value") TO THIS FUNCTION BEFORE RETURN
    """
    # Check what keys need to be retrieved from the patient data
    retrieved_keys = [value_key]
    if context_key is not None:
        retrieved_keys.append(context_key)

    # Filter relevant patient data
    feats = data.loc[data["patid"] == patient_ID, retrieved_keys]  # .copy()

    # Replace nan-like entries with pd.NA only if they are not in the valid list
    feats[value_key] = feats[value_key].where(~feats[value_key].isin(nan_like_values), other=pd.NA)

    # Select valid entries (i.e., entries that are in valid_categories)
    if valid_categories is None:
        return feats.dropna(subset=[value_key])  # return feats
    valid_mask = feats[value_key].isin(valid_categories)
    valid_feats = select_valid_values(feats, valid_mask, patient_ID, value_key)

    return valid_feats


def get_numerical_feature_by_key(
    patient_ID: int,
    data: pd.DataFrame,
    value_key: str,
    valid_range: tuple[Union[int, float], Union[int, float]]|None=None,
    nan_like_values: tuple[Union[str, float], ...]=csts.NAN_LIKE_NUMBERS,
) -> pd.DataFrame:
    """ Get a numerical feature by key if it is in the valid range
        TODO: ADD df.melt(var_name="attribute", value_name="value") TO THIS FUNCTION BEFORE RETURN
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
    
    # Keep entries with valid values only, allowing for NaT time if corresponding value is valid
    data = data.dropna(subset=["value"])
    
    return data


def get_time_value_pairs(
    patient_ID: int,
    data: pd.DataFrame,
    time_key: str,
    value_key: str,
    value_type: str="categorical",
    nan_like_times: tuple[pd.Timestamp, ...]=csts.NAN_LIKE_DATES,
    nan_like_numbers: tuple[Union[str, float], ...]=csts.NAN_LIKE_NUMBERS,
    nan_like_categories: tuple[Union[str, float], ...]=csts.NAN_LIKE_CATEGORIES,
    valid_time_range: tuple[pd.Timestamp, pd.Timestamp]=csts.VALID_DATE_RANGE,
    valid_number_range: tuple[Union[int, float], Union[int, float]]|None=None,
    valid_categories: list[str]|None=None,
    context_key: str|None=None,
) -> pd.DataFrame:
    """ Get time-value pairs for a given patient, returning data pairs with
        columns "time" and "value", and "attribute" for the value key
    """
    # Check what keys need to be retrieved from the patient data
    retrieved_keys = [value_key, time_key]
    if context_key is not None:
        retrieved_keys.append(context_key)

    # Select given patient feature value(s) and associated time(s)
    feats = data.loc[data["patid"] == patient_ID, retrieved_keys]  # .copy()
    feats = feats.rename(columns={time_key: "time", value_key: "value"})
    feats = feats.assign(attribute=value_key)
    
    # Clean the time-value pairs, if required
    feats = clean_time_value_pairs(
        data=feats, value_type=value_type,
        valid_time_range=valid_time_range, valid_number_range=valid_number_range,
        valid_categories=valid_categories, nan_like_times=nan_like_times,
        nan_like_numbers=nan_like_numbers, nan_like_categories=nan_like_categories,
    )

    return feats


def get_longitudinal_data(
    patient_ID: int, 
    data: pd.DataFrame,
    time_key: str,
    value_key: str,
    value_type: str="categorical",
    attribute_key: str|None=None,
    nan_like_times: tuple[pd.Timestamp, ...]=csts.NAN_LIKE_DATES,
    nan_like_numbers: tuple[Union[str, float], ...]=csts.NAN_LIKE_NUMBERS,
    nan_like_categories: tuple[Union[str, float], ...]=csts.NAN_LIKE_CATEGORIES,
    valid_time_range: tuple[pd.Timestamp, pd.Timestamp]=csts.VALID_DATE_RANGE,
    valid_number_range: tuple[Union[int, float], Union[int, float]]|None=None,
    valid_categories: list[str]|None=None,
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


def generate_flat_eavt_df_from_object(
    root_obj: Any,
    time_key: str="time",  # "infection_date"
    filtered_attributes: list[str]=[],  # ["patient_ID"]
    filtered_values: list[Any]=[],  # ["UNKNOWN", "Unknown", None]
) -> pd.DataFrame:
    """ Generate a generic EAVT (entity-attribute-value-time) dataframe from any
        python object with a nested structure

        Args:
            root_obj: object to flatten
            time_attribute_names: key to retrieve time associated to the object
            filtered_attributes: list of attributes to filter out of the EAVT table
        
        Returns:
            dataFrame in the EAVT format
    """
    # Initialize the flattened version of the object
    rows = []

    # Helper function to flatten the object recursively
    def flatten(obj: Any, parent_name: str, parent_attr: str, list_index: int=None):
        
        # Filter out keys that should not appear in the EAVT table as attributes
        if parent_attr.strip("_") in filtered_attributes + [time_key]:
            return

        # Identify event time from the object (fallback on root object if possible)
        event_time = None  # default value
        if hasattr(root_obj, time_key): event_time = getattr(root_obj, time_key)
        if hasattr(obj, time_key): event_time = getattr(obj, time_key)

        # Helper function to fill-in formatted EAVT table entries
        def update_rows(value: str|int|float|bool):
            entity_pattern = r"((?<=[a-z])[A-Z]|(?<=[A-Z])[A-Z](?=[a-z]))"
            entity = re.sub(pattern=entity_pattern, repl=r" \1", string=parent_name)
            attribute = parent_attr.strip("_").replace("_", " ")
            attribute = attribute[0].upper() + attribute[1:]
            
            time = event_time
            if value in filtered_values: return
            if isinstance(value, datetime):
                value = None
                time = value
            
            eavt = {"entity": entity, "attribute": attribute, "value": value, "time": time}
            rows.append(eavt)

        # Handle simple values, which terminate the recursion
        if obj is None or isinstance(obj, (str, int, float, bool)):
            update_rows(obj)
            return
            
        # Handle enums by using each string name
        if isinstance(obj, Enum):
            update_rows(obj.name)
            return

        # Handle lists by iterating and recursing for each item
        if isinstance(obj, list):
            if not obj:
                update_rows(None)
            for i, item in enumerate(obj):
                flatten(item, parent_name, parent_attr, list_index=i)
            return

        # Create the linking row from the parent to this new child entity
        child_name = f"{obj.__class__.__name__}"
        if list_index is not None:  # for objects that are list elemetns
            child_name = f"{child_name} ({list_index})"
        update_rows(child_name)

        # Handle dictionaries by iterating through key-value pairs
        if isinstance(obj, dict):
            for key, val in obj.items():
                flatten(val, child_name, key)
            return

        # Handle custom objects by iterating through their attributes
        if hasattr(obj, "__dict__"):
            for attr, val in vars(obj).items():
                flatten(val, child_name, attr)
            return

        # Fallback for any other type
        update_rows(str(obj))

    # Start the recursion from the root object's attributes
    root_name = f"{root_obj.__class__.__name__}"
    for attr, val in vars(root_obj).items():
        flatten(val, root_name, attr)
        
    return pd.DataFrame(rows)


def generate_wide_eavt_df_from_object(
root_obj: Any,
    time_key: str = "time",  # "infection_date"
    filtered_attributes: list[str] = [],  # ["patient_ID"]
    filtered_values: list[Any] = [],  # ["UNKNOWN", "Unknown", None]
) -> pd.DataFrame:
    """ Generate a generic "wide" EAVT (entity-attribute-value-time) dataframe
        from any python object with a nested structure.

        In this "wide" version, nested object attributes are concatenated to form
        a single descriptive attribute for the root entity, rather than creating
        new entities for nested objects.

        Args:
            root_obj: object to flatten.
            time_key: key to retrieve the time associated with the object.
            filtered_attributes: list of raw attribute names to filter out.
            filtered_values: list of values to filter out of the EAVT table.

        Returns:
            A pandas DataFrame in the wide EAVT format.
    """
    rows = []

    def format_attribute_part(attr: str) -> str:
        """Formats a single part of an attribute path."""
        s = attr.strip("_").replace("_", " ")
        if not s:
            return ""
        return s[0].upper() + s[1:]

    # Helper function to flatten the object recursively
    def flatten(obj: Any, entity_name: str, attribute_path: str):

        # Identify event time, preferring the nested object's time if available
        event_time = getattr(root_obj, time_key, None)
        if hasattr(obj, time_key):
            event_time = getattr(obj, time_key)

        # Helper function to fill-in formatted EAVT table entries
        def update_rows(value: Any, attr_path: str):
            if value in filtered_values:
                return

            time = event_time
            # If the value is a datetime object, use it as the event time
            # and don't record it as a value itself.
            if isinstance(value, datetime):
                time = value
                value = None

            eavt = {"entity": entity_name, "attribute": attr_path, "value": value, "time": time}
            rows.append(eavt)

        # Handle simple values, which terminate the recursion
        if obj is None or isinstance(obj, (str, int, float, bool)):
            update_rows(obj, attribute_path)
            return

        # Handle enums by using their string name
        if isinstance(obj, Enum):
            update_rows(obj.name, attribute_path)
            return

        # Handle lists by iterating and appending the index to the attribute path
        if isinstance(obj, list):
            if not obj:
                update_rows(None, attribute_path)  # Record that the list is empty
                return
            for i, item in enumerate(obj):
                new_attribute_path = f"{attribute_path} ({i})"
                flatten(item, entity_name, new_attribute_path)
            return

        # Handle dictionaries by iterating through key-value pairs
        if isinstance(obj, dict):
            for key, val in obj.items():
                if key.strip("_") in filtered_attributes + [time_key]:
                    continue
                new_attribute_path = f"{attribute_path} {format_attribute_part(key)}".strip()
                flatten(val, entity_name, new_attribute_path)
            return

        # Handle custom objects by iterating through their attributes
        if hasattr(obj, "__dict__"):
            for attr, val in vars(obj).items():
                if attr.strip("_") in filtered_attributes + [time_key]:
                    continue
                new_attribute_path = f"{attribute_path} {format_attribute_part(attr)}".strip()
                flatten(val, entity_name, new_attribute_path)
            return

        # Fallback for any other type
        update_rows(str(obj), attribute_path)

    # Prepare the root entity name with proper spacing (e.g., "BacterialInfection" -> "Bacterial Infection")
    entity_pattern = r"((?<=[a-z])[A-Z]|(?<=[A-Z])[A-Z](?=[a-z]))"
    root_entity_name = re.sub(pattern=entity_pattern, repl=r" \1", string=root_obj.__class__.__name__)

    # Start the recursion from the root object's attributes
    for attr, val in vars(root_obj).items():
        if attr.strip("_") in filtered_attributes + [time_key]:
            continue

        initial_attribute_path = format_attribute_part(attr)
        flatten(val, root_entity_name, initial_attribute_path)

    return pd.DataFrame(rows)