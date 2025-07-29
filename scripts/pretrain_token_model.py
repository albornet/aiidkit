import yaml
import torch
from numpy import ndarray
from datasets import DatasetDict
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback

from src.model.patient_token_embedder import PatientTokenEmbeddingModel, PatientDataCollatorForLanguageModelling
from src.model.evaluate_models import preprocess_logits_for_metrics, CustomEmbeddingEvaluator
from src.data.process.build_dataset import get_formatted_patient_dataset

import src.constants as constants
csts = constants.ConstantsNamespace()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
parser.add_argument("-p", "--pretrain_config_path", default="configs/patient_token_embedder.yaml", help="Path to the pretraining (masked LM) configuration file")
args = parser.parse_args()
DEBUG_FLAG = args.debug


def main():
    """ Train a patient embedding model with MLM (or causal LM)
    """
    # Load model configuration
    with open(args.pretrain_config_path, "r") as f: training_cfg = yaml.safe_load(f)
    
    # Set up trainer with a model and a dataset
    dataset, vocabs = load_formatted_dataset(training_cfg)
    model, data_collator = load_model_and_data_collator(training_cfg, vocabs)
    trainer = load_trainer_for_token_embedding(model, dataset, training_cfg, data_collator)

    # Train the model
    trainer.train()
    print("Training complete")


def load_formatted_dataset(
    training_cfg: dict[str, str],
) -> tuple[DatasetDict, dict[str, int], dict[str, ndarray]]:
    """ Load a dataset formatted for huggingface's trainer object
    """
    # Dataset with formatted times and values
    dataset, _, vocabs = get_formatted_patient_dataset(
        huggingface_dataset_path=csts.HUGGINGFACE_DIR_PATH,
        base_vocab=training_cfg["DEFAULT_BASE_VOCAB"],
        load_huggingface_dataset=training_cfg["LOAD_HUGGINGFACE_DATASET"],
    )

    return dataset, vocabs


def load_model_and_data_collator(
    training_cfg: dict[str, str],
    vocabs: dict[str, dict[str, int]],
) -> PatientTokenEmbeddingModel:
    """ Initialize a patient embedding model and associated data collator
    """
    # Huggingface will only understand the actual torch dtype object
    if "torch_dtype" in training_cfg["ORIGINAL_MODEL_PARAMS"]:
        torch_dtype = getattr(
            torch,
            training_cfg["ORIGINAL_MODEL_PARAMS"]["torch_dtype"],
        )
        training_cfg["ORIGINAL_MODEL_PARAMS"]["torch_dtype"] = torch_dtype

    # Actual patient embeddng model loaded here
    model = PatientTokenEmbeddingModel(
        vocabs=vocabs,
        original_model_id=training_cfg["ORIGINAL_MODEL_ID"],
        language_model_type=training_cfg["ORIGINAL_MODEL_TASK"],
        llm_kwargs=training_cfg["ORIGINAL_MODEL_PARAMS"],
    )

    # Data collator, which depends on the model (define how it is trained)
    mlm = (training_cfg["ORIGINAL_MODEL_TASK"] == "masked")
    data_collator = PatientDataCollatorForLanguageModelling(
        mlm=mlm,
        num_mlm_labels=len(vocabs["value_binned"]),
        num_tokens_max=model.num_tokens_max,
    )

    return model, data_collator


def load_trainer_for_token_embedding(
    model: PatientTokenEmbeddingModel,
    dataset: DatasetDict,
    training_cfg: dict[str, str],
    data_collator: PatientDataCollatorForLanguageModelling,
) -> Trainer:
    """ Initialize a huggingface trainer object
    """
    # In debug mode, evaluation comes earlier for quick assessment
    cfg_training_args: dict = training_cfg["TRAINING_ARGUMENTS"]
    if DEBUG_FLAG:
        cfg_training_args.update({"eval_strategy": "steps", "eval_steps": 10})
    
    # Load training arguments
    training_arguments = TrainingArguments(
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="silhouette_score",  # computed from validation set
        greater_is_better=True,
        report_to="wandb" if not DEBUG_FLAG else "none",
        **cfg_training_args,
    )

    # Define a custom evaluator for a token embedding model
    cfg_eval_args: dict = training_cfg["EVALUATION_ARGUMENTS"]
    metric_computer = CustomEmbeddingEvaluator(
        eval_dataset=dataset["validation"],
        embedding_mode="token",
        optuna_trials=cfg_eval_args["optuna_trials"],
        infection_days_low=cfg_eval_args["infection_days_low"],
        infection_days_high=cfg_eval_args["infection_days_high"],
    )

    # Define trainer object
    causal = (training_cfg["ORIGINAL_MODEL_TASK"] == "causal")
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=metric_computer,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if causal else None,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5),
        ],
    )

    return trainer


if __name__ == "__main__":
    main()