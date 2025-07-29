# src/data/process/infection_labels.py

import pandas as pd
from datasets import DatasetDict
from functools import partial

INFECTION_TYPES = ["bacterial", "viral", "fungal"]
LABEL_CLASSES = ["healthy"] + INFECTION_TYPES


def get_all_infection_events(
    df: pd.DataFrame,
) -> dict[str, list]:
    """
    Extracts all clinically relevant infection events from a patient
    """
    clinically_relevant_inf_events = df.loc[
        df["entity"].str.contains("infection", case=False)
        & (df["attribute"] == "Clinically relevant")
        & (df["value"] == "True")
    ]
    return {
        "infection_time": clinically_relevant_inf_events["time"].tolist(),
        "infection_type": clinically_relevant_inf_events["entity"].tolist(),
    }

def _get_categorical_target(future_infections: list[str]) -> list[int]:
    """
    Compute the multi-hot encoded label for future infections
    """
    label = [0] * len(INFECTION_TYPES)
    for i, inf_type in enumerate(INFECTION_TYPES):
        if any(inf_type in s.lower() for s in future_infections):
            label[i] = 1

    return label


def _get_one_hot_target(
    future_times: list[int],
    future_types: list[str],
) -> int:
    """
    Compute the categorical label for the first future infection
    """
    if not future_times:
        return LABEL_CLASSES.index("healthy")

    first_infection_type = min(zip(future_times, future_types))[1].lower()
    for i, inf_type in enumerate(INFECTION_TYPES):
        if inf_type in first_infection_type:
            return LABEL_CLASSES.index(inf_type)
    
    return LABEL_CLASSES.index("healthy")  # fallback


def create_prediction_windows(
    patient_sample: dict,
    prediction_horizon: int,
    task_grain: int,
) -> dict[str, list]:
    """
    Generate multiple training samples (partial sequences with labels) from a single
    patient's full sequence.
    """
    # Unpack patient data
    all_times = patient_sample["time"]
    infection_events = patient_sample["infection_events"]
    max_time = max(all_times) if all_times else 0

    # This will hold the new, chopped-up samples
    data_columns = list(patient_sample.keys()) + ["categorical_target", "one_hot_target"]
    new_samples = {key: [] for key in data_columns}
    
    # Iterate through the patient's timeline, creating a sample every `task_grain` days
    for cutoff_day in range(0, max_time, task_grain):
        # Define the prediction window
        prediction_start_day = cutoff_day  # +1? not sure, since transplant day is zero
        prediction_end_day = cutoff_day + prediction_horizon

        # Find all infections that occur within this future window
        future_times, future_types = [], []
        for time, type in zip(
            infection_events["infection_time"],
            infection_events["infection_type"],
        ):
            if prediction_start_day <= time <= prediction_end_day:
                future_times.append(time)
                future_types.append(type)
        
        # Compute the labels for this specific window
        categorical_target = _get_categorical_target(future_types)
        one_hot_target = _get_one_hot_target(future_times, future_types)

        # Create the partial input sequence (events up to cutoff_day)
        time_mask = [t <= cutoff_day for t in all_times]
        for key, values in patient_sample.items():
            # The 'infection_events' dict is not a sequence, handle it separately
            if key != "infection_events":
                new_samples[key].append([v for v, m in zip(values, time_mask) if m])
        
        # Keep original full list for reference
        new_samples["infection_events"].append(patient_sample["infection_events"])
        new_samples["categorical_target"].append(categorical_target)
        new_samples["one_hot_target"].append(one_hot_target)

    return new_samples


def generate_prediction_dataset(
    dataset: DatasetDict,
    prediction_horizon: int=30, # units: days
    task_grain: int=7,  # units: days
) -> DatasetDict:
    """
    Transform a dataset of full patient sequences into a dataset of partial sequences,
    each with corresponding infection prediction labels

    Args:
        dataset (DatasetDict): input huggingface dataset
        prediction_horizon (int): number of future days (N) to predict infections for
        task_grain (int): step size in days to create new partial sequences

    Returns:
        DatasetDict: The transformed dataset ready for model training
    """
    task_creation_fn = partial(
        create_prediction_windows,
        prediction_horizon=prediction_horizon,
        task_grain=task_grain,
    )

    # Important to use batched=True since one patient creates several sequences!
    prediction_dataset = dataset.map(
        task_creation_fn,
        batched=True,
        remove_columns=dataset["train"].column_names, # Remove old columns
        desc=f"Generating {prediction_horizon}-day prediction windows",
    )
    
    return prediction_dataset