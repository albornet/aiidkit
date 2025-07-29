import os
import re
import yaml
import random
from safetensors.torch import load_file
from datasets import Dataset
from transformers.trainer_callback import EarlyStoppingCallback
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.models import Pooling
from sentence_transformers.trainer import SentenceTransformerTrainingArguments
from scripts.pretrain_token_model import load_formatted_dataset, load_model_and_data_collator

from src.model.evaluate_models import CustomEmbeddingEvaluator
from src.model.patient_sequence_embedder import (
    PatientTokenEmbeddingModule,
    PatientDataCollatorForSequenceEmbedding,
    PatientTrainer,
)

import src.constants as constants
csts = constants.ConstantsNamespace()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
parser.add_argument("-p", "--pretrain_config_path", default="configs/patient_token_embedder.yaml", help="Path to the pretraining (masked LM) configuration file")
parser.add_argument("-f", "--finetune_config_path", default="configs/patient_sequence_embedder.yaml", help="Path to the finetuning (sentence embedding) configuration file")
args = parser.parse_args()
DEBUG_FLAG = args.debug


def main():
    """ Finetune a patient token embedding model into a patient sequence embedding model
    """
    # Load the original model training and new finetuning configuration dictionaries
    with open(args.pretrain_config_path, "r") as f: pretrain_cfg = yaml.safe_load(f)
    with open(args.finetune_config_path, "r") as f: finetune_cfg = yaml.safe_load(f)

    # Prepare data (training and evaluation data are built differently)
    dataset, vocabs = load_formatted_dataset(pretrain_cfg)
    dataset["train"] = dataset["train"].map(
        function=create_contrastive_pair,
        remove_columns=["entity", "attribute", "value", "value_binned", "time"],
        desc="Creating a dataset of contrastive pairs",
    )
    dataset["validation"] = dataset["validation"].remove_columns(["value"])
    data_collator = PatientDataCollatorForSequenceEmbedding()

    # Load original token embedding model with pre-trained weights
    token_embedder, _ = load_model_and_data_collator(pretrain_cfg, vocabs)
    pretrained_model_dir = pretrain_cfg["TRAINING_ARGUMENTS"]["output_dir"]
    pretrained_model_path = get_latest_checkpoint(pretrained_model_dir)
    model_weights = load_file(pretrained_model_path)
    token_embedder.load_state_dict(model_weights)
    
    # Build the SentenceTransformer model from the original model
    token_embedding_module = PatientTokenEmbeddingModule(token_embedder)
    pooling_module = Pooling(token_embedding_module.token_embedder.hidden_size, pooling_mode="mean")
    model = SentenceTransformer(modules=[token_embedding_module, pooling_module])

    # Define trainer and fine-tune model for sequence embedding
    trainer = load_trainer_for_sequence_embedding(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        finetune_cfg=finetune_cfg,
        data_collator=data_collator,
    )
    trainer.train()


def create_contrastive_pair(sample: dict) -> dict:
    """ Transforms a sequence into an anchor/positive pair for contrastive learning
    """
    # Determine a random split point for the anchor sequence
    seq_len = len(sample["time"])
    partial_len = random.randint(1, seq_len - 1)

    # Create the anchor and positive samples
    sequence_keys = ["entity", "attribute", "value_binned", "time"]
    anchor = {key: sample[key][:partial_len] for key in sequence_keys}
    positive = {key: sample[key] for key in sequence_keys}
    
    return {"anchor": anchor, "positive": positive}


def get_latest_checkpoint(results_dir: str) -> str:
    """ Get the latest checkpoint directory from a general directory
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
) -> PatientTrainer:
    """ Initialize a sentence-transformer trainer object for sequence embedding
    """
    # In debug mode, evaluation comes earlier for quick assessment
    cfg_train_args: dict = finetune_cfg["TRAINING_ARGUMENTS"]
    if DEBUG_FLAG:
        cfg_train_args.update({"eval_strategy": "steps", "eval_steps": 10})
    
    # Define training arguments
    training_args = SentenceTransformerTrainingArguments(
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="silhouette_score",  # computed from validation set
        greater_is_better=True,
        report_to="wandb" if not DEBUG_FLAG else "none",
        **cfg_train_args,
    )

    # Define a custom evaluator for a sequence embedding model
    cfg_eval_args: dict = finetune_cfg["EVALUATION_ARGUMENTS"]
    metric_computer = CustomEmbeddingEvaluator(
        eval_dataset=eval_dataset,
        embedding_mode="sequence",
        optuna_trials=cfg_eval_args["optuna_trials"],
        infection_days_low=cfg_eval_args["infection_days_low"],
        infection_days_high=cfg_eval_args["infection_days_high"],
        eval_batch_size=cfg_train_args["per_device_eval_batch_size"],
        eval_data_collator=data_collator,
    )

    # Fine-tune the model for sequence embedding
    train_loss = losses.MultipleNegativesRankingLoss(model)
    trainer = PatientTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        data_collator=data_collator,
        evaluator=[metric_computer],  # more like evaluators
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    return trainer


if __name__ == "__main__":
    main()