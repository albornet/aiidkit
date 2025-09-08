import os
import re
import yaml
import torch
import random
import argparse
from functools import partial
from safetensors.torch import load_file
from datasets import Dataset, DatasetDict
from transformers.trainer_callback import EarlyStoppingCallback
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.models import Pooling
from sentence_transformers.trainer import SentenceTransformerTrainingArguments

from scripts.pretrain_token_model import load_model_for_patient_token_embedding
from src.data.process.patient_dataset import load_hf_data_and_metadata
from src.model.evaluate_models import CustomEmbeddingEvaluator
from src.model.patient_sequence_embedder import (
    PatientTokenEmbeddingModule,
    PatientDataCollatorForSequenceEmbedding,
    PatientTrainer,
)
import src.constants as constants
csts = constants.ConstantsNamespace()


def main(args):
    """
    Finetune a patient token embedding model into a patient sequence embedding model
    """
    # Load the original model training and new finetuning configuration dictionaries
    with open(args.pretrain_config_path, "r") as f: pretrain_cfg = yaml.safe_load(f)
    with open(args.finetune_config_path, "r") as f: finetune_cfg = yaml.safe_load(f)
    
    # Load dataset and data collator
    dataset, vocabs = get_sequence_embedding_dataset(
        args.use_supervised_labels, int(args.prediction_horizon),
    )
    valid_label_columns = finetune_cfg["SUPERVISED_LABEL_NAMES"] if args.use_supervised_labels else None
    data_collator = PatientDataCollatorForSequenceEmbedding(valid_label_columns=valid_label_columns)

    # Load original token embedding model with pre-trained weights
    token_embedder = load_model_for_patient_token_embedding(
        model_cfg=pretrain_cfg["MODEL_ARGUMENTS"],
        vocabs=vocabs,
        for_sentence_embedding=True,  # so that hidden states are not sent to cpu
    )
    pretrained_model_dir = pretrain_cfg["TRAINING_ARGUMENTS"]["output_dir"]
    pretrained_model_path = get_latest_checkpoint(pretrained_model_dir)
    model_weights = load_file(pretrained_model_path)
    token_embedder.load_state_dict(model_weights)

    # Build the SentenceTransformer model from the original model
    token_embedding_module = PatientTokenEmbeddingModule(token_embedder)
    pooling_module = Pooling(token_embedding_module.token_embedder.hidden_size, pooling_mode="mean")
    model = SentenceTransformer(modules=[token_embedding_module, pooling_module])

    # Define trainer for the sequence embedding task
    trainer = load_trainer_for_sequence_embedding(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        finetune_cfg=finetune_cfg,
        data_collator=data_collator,
        debug_flag=args.debug,
        use_supervised_labels=args.use_supervised_labels,
    )

    # Fine-tune model for sequence embedding
    trainer.train()


def get_sequence_embedding_dataset(
    use_supervised_labels: bool=False,
    prediction_horizon: int=30,
) -> tuple[DatasetDict, dict[str, dict[str, int]]]:
    """
    Create a dataset of contrastive pairs for sequence embedding
    """
    raise NotImplementedError("TODO: CHECK THAT IT WORKS WITH NEW HF DATA LOADING FUNCTION")
    # Load either a dataset of partial patient sequences with infection labels
    # using true labels to determine positive and negative samples
    if use_supervised_labels:
        horizon_dir = f"{prediction_horizon}_days_horizon"
        dataset_dir = os.path.join(csts.INFECTION_DIR_PATH, horizon_dir)
        dataset, _, vocabs = load_hf_data_and_metadata(dataset_dir, csts.METADATA_DIR_PATH)
        return dataset, vocabs
    
    # Or the full patient sequences, but with positive and anchor samples pre-built
    else:
        dataset_dir = csts.HUGGINGFACE_DIR_PATH
        dataset, _, vocabs = load_hf_data_and_metadata(dataset_dir, csts.METADATA_DIR_PATH)

        # Select correct contrastive pair creation mode
        sequence_columns = [
            "entity", "entity_id", "attribute", "attribute_id",
            "value", "value_binned", "value_id", "time", "days_since_tpx",
        ]
        create_contrastive_pair_fn = partial(
            create_contrastive_pair_without_labels,
            sequence_columns=sequence_columns,
            use_supervised_labels=use_supervised_labels,
        )

        # Create contrastive pairs, only for the training set
        dataset["train"] = dataset["train"].map(
            function=create_contrastive_pair_fn,
            remove_columns=sequence_columns,
            desc="Creating a dataset of contrastive pairs",
            load_from_cache_file=False,
        )
        dataset["validation"] = dataset["validation"].remove_columns(["value"])

        # Dataset has these columns: anchor, positive, each of which includes all of
        # the sequence columns; moreover, the sequence-level columns are given with
        # anchor and positive columns (labels, patient_csv_path, etc.)
        return dataset, vocabs


def create_contrastive_pair_without_labels(
    sample: dict,
    sequence_columns: list[str],
    use_supervised_labels: bool=False,
) -> dict[str, dict[str, torch.Tensor]]:
    """
    Transform a sequence into an anchor/positive pair for contrastive learning
    """
    # Determine a random split point for the anchor sequence
    seq_len = len(sample["days_since_tpx"])
    if use_supervised_labels:
        partial_len = seq_len  # the labelled dataset already has partial sequences
    else:
        partial_len = random.randint(1, seq_len - 1)

    # Create the anchor and positive samples
    anchor = {key: sample[key][:partial_len] for key in sequence_columns}
    positive = {key: sample[key] for key in sequence_columns}

    return {"anchor": anchor, "positive": positive}


def get_latest_checkpoint(results_dir: str) -> str:
    """
    Get the latest checkpoint directory from a general directory
    """
    # Identify checkpoint directory with the highest number
    ckpt_dirs = (
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("checkpoint-")
    )
    max_ckpt_dir = max(ckpt_dirs, key=lambda d: int(re.search(r"\d+", d).group()))
    max_ckpt_dir_path = os.path.join(results_dir, max_ckpt_dir)

    # Ensure the directory contains a single model file
    model_paths = [a for a in os.listdir(max_ckpt_dir_path) if ".safetensors" in a]
    if not model_paths:
        raise FileNotFoundError(f"No model files found in {max_ckpt_dir_path}")
    if len(model_paths) > 1:
        raise ValueError(f"Multiple model files found in {max_ckpt_dir_path}: {model_paths}")

    return os.path.join(max_ckpt_dir_path, model_paths[0])


def load_trainer_for_sequence_embedding(
    model: SentenceTransformer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    finetune_cfg: dict[str, str],
    data_collator: PatientDataCollatorForSequenceEmbedding,
    debug_flag: bool=False,
    use_supervised_labels: bool=False,
) -> PatientTrainer:
    """
    Initialize a sentence-transformer trainer object for sequence embedding
    """
    # In debug mode, evaluation comes earlier for quick assessment
    cfg_train_args: dict = finetune_cfg["TRAINING_ARGUMENTS"]
    if debug_flag:
        cfg_train_args.update({"eval_strategy": "steps", "eval_steps": 10})

    # Define training arguments
    label_names = finetune_cfg["SUPERVISED_LABEL_NAMES"] if use_supervised_labels else None
    training_args = SentenceTransformerTrainingArguments(
        label_names=label_names,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="loss",  # computed from validation set
        greater_is_better=False,
        report_to="wandb" if not debug_flag else "none",
        push_to_hub=False,
        **cfg_train_args,
    )

    # Define a custom evaluator for a sequence embedding model
    cfg_eval_args: dict = finetune_cfg["EVALUATION_ARGUMENTS"]
    metric_computer = CustomEmbeddingEvaluator(
        eval_dataset=eval_dataset,
        embedding_mode="sequence",
        eval_label_key=cfg_eval_args["eval_label_key"] if use_supervised_labels else None,  # MIGHT CHECK IN MODEL_ARGUMENTS FOR THIS KEY
        optuna_trials=cfg_eval_args["optuna_trials"],
        eval_batch_size=cfg_train_args["per_device_eval_batch_size"],
        eval_data_collator=data_collator,
    )

    # Loss function
    if use_supervised_labels:
        train_loss = losses.BatchHardSoftMarginTripletLoss(model=model)
    else:
        train_loss = losses.MultipleNegativesRankingLoss(model=model)
    
    # Fine-tune the model for sequence embedding
    es_patience = cfg_eval_args["early_stopping_patience"]
    trainer = PatientTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        supervised_mode=use_supervised_labels,
        data_collator=data_collator,
        evaluator=[metric_computer],  # more like evaluators
        callbacks=[EarlyStoppingCallback(early_stopping_patience=es_patience)],
    )

    return trainer


if __name__ == "__main__":

    # Common command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("-sl", "--use_supervised_labels", action="store_false", help="Use true labels to evaluate models")
    parser.add_argument("-ph", "--prediction_horizon", default=30, type=int, choices=csts.PREDICTION_HORIZONS, help="Horizon of the infection prediction task for model evaluation")
    parser.add_argument("-ct", "--cutoff_days_train", nargs="+", type=int, default=None, choices=csts.CUTOFF_DAYS, help="Cutoff days for the training dataset, take them all if None")
    parser.add_argument("-cv", "--cutoff_days_valid", nargs="+", type=int, default=None, choices=csts.CUTOFF_DAYS, help="Cutoff days for the evaluation/testing dataset, take them all if None")

    # Arguments that are specific to the current script
    parser.add_argument("-pc", "--pretrain_config_path", default="configs/patient_token_embedder.yaml", help="Path to the pretraining (masked LM) configuration file")
    parser.add_argument("-fc", "--finetune_config_path", default="configs/patient_sequence_embedder.yaml", help="Path to the finetuning (sentence embedding) configuration file")

    # Run the finetuning script
    args = parser.parse_args()
    main(args)