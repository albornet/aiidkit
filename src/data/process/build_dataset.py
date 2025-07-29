import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.contrib.concurrent import process_map
from collections import defaultdict
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, concatenate_datasets
from src.data.process.infection_labels import get_all_infection_events
import src.constants as constants
csts = constants.ConstantsNamespace()


def get_formatted_patient_dataset(
    huggingface_dataset_path: str,
    base_vocab: dict[str, int]|None=None,
    load_huggingface_dataset: bool=False,
) -> tuple[DatasetDict, dict[str, int], dict[str, np.ndarray]]:
    """ Create a patient dataset, then format it for language model processing
    
        Returns:
            dataset (DatasetDict): formatted huggingface patient dataset
            type_vocab (dict[str, int]): mapping of type tokens to type token ids
            bin_edges_by_type (dict[str, np.ndarray]): quantile bins for each type
    """
    # Build formatted dataset from pre-processed patient csv filess
    if not load_huggingface_dataset:
        build_huggingface_patient_dataset(
            input_data_dir=csts.PREPROCESSED_DIR_PATH,
            output_data_dir=csts.HUGGINGFACE_DIR_PATH,
        )
    hf_dataset = DatasetDict.load_from_disk(huggingface_dataset_path)
    dataset, intervals_by_type = bin_values_by_type(hf_dataset)

    # Collect all possible expressions for entities, attributes, and binned values
    vocab_sets = {"entity": set(), "attribute": set(), "value_binned": set()}
    for sample in concatenate_datasets([dataset["train"], dataset["validation"]]):
        for key, vocab_set in vocab_sets.items():
            vocab_set.update(sample[key])
    # vocab_sets["joined"] = set().union(*vocab_sets.values())

    # Build vocabularies by extending the base_vocab
    if base_vocab is None: base_vocab = {"[PAD]": 0, "[UNK]": 1}
    vocabs = {key: dict(base_vocab) for key in vocab_sets.keys()}
    for key, vocab_set in vocab_sets.items():
        for idx, term in enumerate(sorted(list(vocab_set))):
            vocabs[key][term] = idx + len(base_vocab)

    # Map the dataset using the vocabularies, and handle unknown tokens
    unk_token_id = base_vocab["[UNK]"]
    dataset = dataset.map(
        lambda sample: {
            "entity": [vocabs["entity"].get(e, unk_token_id) for e in sample["entity"]],
            "attribute": [vocabs["attribute"].get(a, unk_token_id) for a in sample["attribute"]],
            "value_binned": [vocabs["value_binned"].get(v, unk_token_id) for v in sample["value_binned"]],
        },
        desc="Mapping tokens to IDs",
    )
    
    # Format data sequences as torch tensors
    dataset.set_format(
        type="torch",
        columns=["entity", "attribute", "value_binned", "time", "infection_events"],
    )

    # (TEST) For validation data, keep only the data samples until transplantation day (i.e. time <= 0)
    dataset["validation"] = dataset["validation"].map(
        lambda sample: filter_by_time(sample, time_range=(None, 60)), 
        desc="Filtering validation set for time <= 60",
    )
    
    return dataset, intervals_by_type, vocabs


def build_huggingface_patient_dataset(
    input_data_dir: str,
    output_data_dir: str=None,
) -> DatasetDict:
    """ Build a huggingface dataset from individual patient csv files
    """ 
    # Create list of patient dictionaries read from patient csv files
    patient_csv_paths = [str(p) for p in Path(input_data_dir).rglob("patient_*.csv")]
    data = process_map(
        process_patient_csv, patient_csv_paths, max_workers=8, chunksize=100,
        desc="Reading patient files",
    )

    # Create train, validation, and test splits by patient (70% // 15% // 15%)
    train_data, valtest_data = train_test_split(data, test_size=0.3, random_state=1)
    val_data, test_data = train_test_split(valtest_data, test_size=0.5, random_state=1)

    # Create huggingface dataset dictionary that includes all splits
    dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
        "test": Dataset.from_list(test_data)
    })

    # Save dataset to disk
    dataset.save_to_disk(output_data_dir)
    print("Huggingface dataset created and saved to disk")


def filter_by_time(
    sample: dict[str, list],
    time_range: tuple[int|None, int|None]=(None, 0),
    time_column: str="time",
    columns_to_filter: list[str]=["entity", "attribute", "value_binned", "time"],
) -> dict[str, list]:
    """ Keeps data points in a sequence where time is in a specific range
        - Note: the range is inclusive, i.e. [start, end]
        - Note: time_range is a tuple of (start, end), where None means no limit
    """
    # Create a boolean mask based on the inclusive range.
    start, end = time_range
    mask = [
        (start is None or t >= start) and (end is None or t <= end)
        for t in sample[time_column]
    ]
    
    # Apply this mask to all relevant columns
    for key in columns_to_filter:
        if key in sample:
            sample[key] = [v for v, m in zip(sample[key], mask) if m]
            
    return sample


def process_patient_csv(patient_csv_path: str):
    """ Function to read a patient csv file to a dictionary of input and labels
    """
    # Load patient data as a pandas dataframe
    patient_df = pd.read_csv(patient_csv_path)
    
    # Fill-in time for static values using the first transplantation date
    patient_df["time"] = pd.to_datetime(patient_df["time"])
    tpx_event_mask = patient_df["attribute"] == "Transplantation event"
    first_tpx_time = patient_df.loc[tpx_event_mask, "time"].iloc[0]
    patient_df["time"] = patient_df["time"].fillna(first_tpx_time)

    # Normalize time as the difference of days with the transplantation date
    patient_df["time"] = (patient_df["time"] - first_tpx_time).dt.days
    patient_df = patient_df.sort_values("time").reset_index(drop=True)  # .astype(str)

    # Handle mixed-type and NaN values (types are added later)
    patient_df = patient_df.dropna(subset=["value", "attribute"])
    patient_df["value"] = patient_df["value"].astype(str)

    # Join input features and infection labels into a sample dictionary
    sample = patient_df[["entity", "attribute", "value", "time"]].to_dict(orient="list")
    sample.update({"infection_events": get_all_infection_events(patient_df)})
    
    return sample


def bin_values_by_type(
    dataset: DatasetDict,
    bin_labels: list[str]={
        0: "Lowest", 1: "Lower", 2: "Low", 3: "Middle",
        4: "High", 5: "Higher", 6: "Highest",
    },
) -> DatasetDict:
    """ Post-processing a huggingface dataset dictionary to bin values by
        quantiles computed over each feature type
    """
    # Group training and validation longitudinal sample values by attribute
    train_val_data = concatenate_datasets([dataset["train"], dataset["validation"]])
    values_by_attr = defaultdict(list)
    attribute_types = defaultdict(lambda: "numerical")
    feature_types = {}
    for sample in train_val_data:

        # Fill-in groups by attribute type (either category or value)
        for entity, attribute, value in zip(sample["entity"], sample["attribute"], sample["value"]):
            feature_types[attribute] = entity
            try:
                values_by_attr[attribute].append(float(value))
            except ValueError:  # i.e., if str value is not a number
                values_by_attr[attribute].append(value)
                attribute_types[attribute] = "categorical"

    # Compute bin intervals for continuous data
    attribute_intervals: dict[str, pd.IntervalIndex] = {}
    for attribute, values in values_by_attr.items():
        if attribute_types[attribute] == "numerical" and len(set(values)) > 10:
            try:
                binned = pd.qcut(x=values,  q=len(bin_labels))
            except ValueError: # might fail for very skewed data, like with many 0.0
                binned = pd.cut(x=values, bins=len(bin_labels))
            attribute_intervals[attribute] = binned.categories
        else:  # correcting the type of numerical values with low numerosity
            attribute_types[attribute] = "categorical"

    # Define numerical value binning, given bin intervals
    def bin_sample_values(values, attributes):
        values_binned = []
        for value, attribute in zip(values, attributes):
            if attribute_types[attribute] == "numerical":
                try: bin_idx = attribute_intervals[attribute].get_loc(float(value))
                except KeyError: bin_idx = len(attribute_intervals[attribute]) - 1
                values_binned.append(bin_labels[bin_idx])
                
            elif attribute_types[attribute] == "categorical":
                try: category = str(int(float(value)))
                except ValueError: category = value
                values_binned.append(category)
            
        return {"value_binned": values_binned}

    # Apply the binning to all dataset samples (including test set)
    bin_fn = lambda s: bin_sample_values(s["value"], s["attribute"])
    binned_dataset = dataset.map(bin_fn, desc="Binning values")

    return binned_dataset, attribute_intervals
