import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, Features, Value, Sequence, concatenate_datasets
from datetime import datetime
from dateutil.parser import parse
from functools import partial
from collections import defaultdict
from typing import Union
import src.data.constants as constants
csts = constants.ConstantsNamespace()


def main():
    """ Testing function
    """
    build_huggingface_patient_dataset(
        input_data_dir=csts.PREPROCESSED_DIR_PATH,
        output_data_dir=csts.HUGGINGFACE_DIR_PATH,
    )

    fmt_dataset, type_vocab, bin_edges_by_type = get_formatted_patient_dataset(
        huggingface_dataset_path=csts.HUGGINGFACE_DIR_PATH,
        num_value_bins=5,
        num_added_tokens=0,
    )

    import ipdb; ipdb.set_trace()


def build_huggingface_patient_dataset(
    input_data_dir: str,
    output_data_dir: str=None,
) -> DatasetDict:
    """ Build a huggingface dataset from individual patient csv files
    
    Returns:
        Dataset: processed huggingface dataset
    """
    # Function to read a patient csv file to a dictionary
    def process_csv(file_path):
        df = pd.read_csv(file_path)
        
        # Fill-in NaN times (for static values) using a default time
        first_tpx_time = df.loc[df["attribute"] == "tpxdate", "time"].iloc[0]
        df["time"] = df["time"].fillna(first_tpx_time)
        
        # Handle mixed-type and NaN values (types are added later)
        df = df.dropna(subset=["value", "attribute"])
        df["value"] = df["value"].astype(str)

        return df[["attribute", "value", "time"]].to_dict(orient="list")
        
    # Create list of patient dictionaries read from patient csv files
    data = []
    for folder, _, file_names in os.walk(input_data_dir):  # recursive walk (why not)
        for file_name in file_names:
            file_root, file_ext = os.path.splitext(file_name)
            patient_id = file_root.split("patient_")[-1]
            is_valid_file_name = (patient_id.isdigit() and file_ext == ".csv")
            if not is_valid_file_name: continue
            file_path = os.path.join(folder, file_name)
            data.append(process_csv(file_path))
    
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
    

def get_formatted_patient_dataset(
    huggingface_dataset_path: str,
    num_value_bins: int,
    num_added_tokens: int,
) -> tuple[DatasetDict, dict[str, int], dict[str, np.ndarray]]:
    """ Create a patient dataset, then format it for language model processing
    
        Returns:
            dataset (DatasetDict): formatted huggingface patient dataset
            type_vocab (dict[str, int]): mapping of type tokens to type token ids
            bin_edges_by_type (dict[str, np.ndarray]): quantile bins for each type
    """
    # Load unformatted dataset
    dataset = DatasetDict.load_from_disk(huggingface_dataset_path)
    
    # Identify vocabulary for "attribute" feature set (note: not using test set)
    all_types = set([
        t for split, data in dataset.items() if split in ["train", "validation"]
        for sample in data for t in sample["attribute"]
    ])
    type_vocab = {k: v + num_added_tokens for v, k in enumerate(all_types)}
    
    # Format dataset to huggingface format, but with input_embed as input
    dataset, bin_edges_by_type = bin_values_by_type(dataset, num_value_bins, num_added_tokens)
    dataset = dataset.map(partial(encode_fn, type_vocab=type_vocab))
    dataset.set_format(type="torch", columns=["time", "value", "attribute"])

    return dataset, type_vocab, bin_edges_by_type


def bin_values_by_type(
    dataset: DatasetDict,
    num_value_bins: int,
    num_added_tokens: int,
) -> DatasetDict:
    """ Post-processing a huggingface dataset dictionary to bin values by
        quantiles computed over each feature type
    """
    # Use only training and validation to compute quantile bins
    train_val_data = concatenate_datasets([dataset["train"], dataset["validation"]])
    
    # Group values by type with all dataset samples
    values_by_attribute = defaultdict(list)
    attribute_types = defaultdict(lambda: "numerical")
    for sample in train_val_data:

        # Fill-in groups by attribute type (either category or value)
        for value, attribute in zip(sample["value"], sample["attribute"]):
            try:
                values_by_attribute[attribute].append(float(value))
            except ValueError:  # i.e., if str value is not a number
                values_by_attribute[attribute].append(value)
                attribute_types[attribute] = "categorical"

    # Compute N-tile bin edges for each numerical attribute
    # Here, "100" means "100%", and "-1" is used because "right=True" is used
    # when digitizing, i.e., very large values will be assigned the last bin
    binned_space = np.linspace(0, 100, num_value_bins - 1)
    bin_edges_by_type = {}
    categories_by_type = {}
    for attribute, values in values_by_attribute.items():
        if attribute_types[attribute] == "numerical":
            bin_edges_by_type[attribute] = np.percentile(values, binned_space)
        elif attribute_types[attribute] == "categorical":
            categories_by_type[attribute] = set(values)

    from pprint import pprint
    pprint(categories_by_type)
    import ipdb; ipdb.set_trace()

    # Bin the values for each sample, i.e., map each float-value to an int
    def bin_sample_values(times, values, types):
        """ Find value bin index using the precomputed bin edges for a given type
        """
        binned_values = []
        for value, attrs_ in zip(values, types):
            if attrs_ in categorical_attributes:
                binned_values.append(category_to_id[attrs_])
            else:
                bin_edges = bin_edges_by_type[attrs_]  # defined outside function space
                bin_index = np.digitize(value, bin_edges, right=True)  # intervals include right bin_edges
                binned_values.append(bin_index + num_added_tokens)
        
        return {"time": times, "value": binned_values, "attribute": types}

    # Apply the binning to all dataset samples (including test set)
    bin_fn = lambda s: bin_sample_values(s["time"], s["value"], s["attribute"])
    binned_dataset = dataset.map(bin_fn)
    
    # Update data type for the "values" field (from float to int)
    original_features = binned_dataset["train"].features
    new_feature = Features({"value": Sequence(Value("int64"))})
    updated_features = Features({**original_features, **new_feature})
    binned_dataset = binned_dataset.cast(updated_features)
    
    return binned_dataset, bin_edges_by_type



# def bin_values_by_type_old(
#     dataset: DatasetDict,
#     num_value_bins: int,
#     num_added_tokens: int,
# ) -> DatasetDict:
#     """ Post-processing a huggingface dataset dictionary to bin values by
#         quantiles computed over each feature type
#     """
#     # Use only training and validation to compute quantile bins
#     train_val_data = concatenate_datasets([dataset["train"], dataset["validation"]])
    
#     # Group values by type with all dataset samples
#     values_by_type = defaultdict(list)
#     for sample in train_val_data:
#         values = sample["value"]  # list of floats
#         attrs = sample["attribute"]  # list of categories (str)
#         for value, attrs_ in zip(values, attrs):
#             values_by_type[attrs_].append(value)
    
#     # Compute N-tile bin edges for each type. In this conext, "100" means "100%",
#     # and "-1" is used because "right=True" is used when digitizeing, which means
#     # very large values will be assigned the last bin
#     binned_space = np.linspace(0, 100, num_value_bins - 1)
#     bin_edges_by_type = {}
#     for attrs_, values in values_by_type.items():
#         import ipdb; ipdb.set_trace()
#         bin_edges_by_type[attrs_] = np.percentile(values, binned_space)
    
#     # Bin the values for each sample, i.e., map each float-value to an int
#     def bin_sample_values(times, values, types):
#         """ Find value bin index using the precomputed bin edges for a given type
#         """
#         binned_values = []
#         for value, attrs_ in zip(values, types):
#             bin_edges = bin_edges_by_type[attrs_]  # defined outside function space
#             bin_index = np.digitize(value, bin_edges, right=True)  # intervals include right bin_edges
#             binned_values.append(bin_index + num_added_tokens)
        
#         return {"time": times, "value": binned_values, "attribute": types}
    
#     # Apply the binning to all dataset samples (including test set)
#     bin_fn = lambda s: bin_sample_values(s["time"], s["value"], s["attribute"])
#     binned_dataset = dataset.map(bin_fn)
    
#     # Update data type for the "values" field (from float to int)
#     original_features = binned_dataset["train"].features
#     new_feature = Features({"value": Sequence(Value("int64"))})
#     updated_features = Features({**original_features, **new_feature})
#     binned_dataset = binned_dataset.cast(updated_features)
    
#     return binned_dataset, bin_edges_by_type


def normalize_time(
    str_time: str,
    epoch_date: datetime=datetime(1945, 1, 1, 0, 0, 0),  # actual minimum date in the data is 27-06-1946
) -> float:
    """ Normalize a datetime object to a float curor between min_date and max_date
        Values outside the date range are still valid (but outside 0 and 1)
        TODO: USE PATIENT'S TPX DATE TO HAVE MORE PRECISION AROUND THAT TIME
    """
    time_since_epoch = parse(str_time) - epoch_date
    days_since_epoch = time_since_epoch.total_seconds() / (24 * 3600)
    return days_since_epoch


def encode_fn(
    sample: dict[str, list[Union[float, str]]],
    type_vocab: dict[str, int],
) -> dict[str, torch.Tensor]:
    """ Tokenize and tensorize a formated patient triplet list
        TODO: USE PATIENT'S TPX DATE TO HAVE MORE PRECISION AROUND THAT TIME
    """
    # Tensorize and add feature dimension to times
    float_times = [normalize_time(str_time) for str_time in sample["time"]]
    sample["time"] = torch.tensor(float_times, dtype=torch.float32)
    sample["time"] = sample["time"].unsqueeze(-1)

    # Tensorize values
    sample["value"] = torch.tensor(sample["value"], dtype=torch.int64)
    
    # Tensorize and encode attributes, using 1 for unknown feature type tokens
    sample["attribute"] = [type_vocab.get(type_str, 1) for type_str in sample["attribute"]]
    sample["attribute"] = torch.tensor(sample["attribute"], dtype=torch.int64)
    
    return sample


if __name__ == "__main__":
    main()