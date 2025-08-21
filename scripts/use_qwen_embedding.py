import os
import yaml
import argparse
import numpy as np
import torch
from functools import partial
from datasets import DatasetDict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
)
from vllm import LLM
from transformers import BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.trainer import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.training_args import BatchSamplers

from src.data.process.patient_dataset import load_hf_data_and_metadata
from src.model.evaluate_models import CustomEmbeddingEvaluator
import src.constants as constants
csts = constants.ConstantsNamespace()


def main(args):
    """
    Main function to run the training pipeline
    """
    # Small warning message (script entirely relies on supervised labels)
    if not args.use_supervised_labels:
        print(
            "Argument `use_supervised_labels` was set to False, but will be "
            "ignored, as the current script always uses supervised labels."
        )

    # Get configuration for training the model
    with open(args.config_path, "r") as config_file:
        cfg = yaml.safe_load(config_file)

    # Get the correct dataset
    dataset, _, _ = load_hf_data_and_metadata(
        data_dir=csts.INFECTION_DIR_PATH,
        prediction_horizon=args.prediction_horizon,
        metadata_dir=csts.METADATA_DIR_PATH,
        cutoff_days_train=args.cutoff_days_train,
        cutoff_days_valid=args.cutoff_days_valid,
    )

    # Shorten the dataset if required
    if args.debug:
        dataset["train"] = dataset["train"].select(range(1000))
        dataset["validation"] = dataset["validation"].select(range(250))
        dataset["test"] = dataset["test"].select(range(10))

    # Set the input and label keys for the classification task
    format_key = "markdown" if cfg["data"]["use_markdown"] else "text"
    input_key = f"patient_card_{format_key}"
    label_key = cfg["train"]["true_label_key"]

    # Get the correct model for finetuning / linear probing
    model = load_correct_model(
        model_cfg=cfg["model"],
        train_cfg=cfg["train"],
    )
    
    # Train the model / classifier
    train_model_or_linear_prober(
        model=model,
        dataset=dataset,
        input_key=input_key,
        label_key=label_key,
        train_cfg=cfg["train"],
        eval_cfg=cfg["evaluation"],
    )


def load_correct_model(
    model_cfg: dict,
    train_cfg: dict,
) -> SentenceTransformer|LLM:
    """
    Get the model correctly set for finetuning with or without LoRA / QLoRA
    """
    # If the whole sentence-embedding model is trained on the classification task
    if args.finetune:

        # QLoRA quantization config needs to be handled before model loading
        quantization_config = None
        if train_cfg["quantization"].get("use_4bit", False):
            quantization_config = BitsAndBytesConfig(**train_cfg["quantization"]["bnb_config"])
            model_cfg["model_kwargs"].update({"quantization_config": quantization_config})

        # Initialize embedding model
        model = SentenceTransformer(
            model_name_or_path=model_cfg["model_name"],
            model_kwargs=model_cfg["model_kwargs"],
        )

        # Use smaller sequences (taken up to the end) in debug mode
        if args.debug:
            model.max_seq_length = model_cfg["max_seq_length"]
            model.tokenizer.truncation_side = model_cfg["truncation_side"]

        # Prevent any issue with gradient checkpointing in QLoRA
        if quantization_config is not None:
            model.model.config.use_cache = False

    # If only a linear prober is trained on top of the model's output
    else:
        model = LLM(
            model=model_cfg["model_name"],
            task="embed",
            gpu_memory_utilization=0.90,
            max_num_seqs=1,
            swap_space=4.0,
            cpu_offload_gb=4.0,
        )
    
    return model


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )


def finetune_model(
    model: SentenceTransformer,
    train_dataset: DatasetDict,
    eval_dataset: DatasetDict,
    train_cfg: dict,
    eval_cfg: dict,
):
    """
    Finetune a sentence embedding model for classification using a Trainer.
    """
    # Define training arguments
    output_subdir = f"finetuned_{args.prediction_horizon}_days_horizon"
    output_path = os.path.join(train_cfg["output_dir"], output_subdir)
    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_path,
        batch_sampler=BatchSamplers.GROUP_BY_LABEL,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="loss",  # change to correct one once good
        greater_is_better=False,
        report_to="wandb" if not args.debug else "none",
        push_to_hub=False,
        **train_cfg["training_arguments"],
    )

    # Handle LoRA / QLoRA
    if train_cfg.get("use_lora", False):

        # Handle quantized LoRA
        if train_cfg.get("quantization", {}).get("use_4bit", False):
             model = prepare_model_for_kbit_training(model)

        # In the latest PEFT, `add_adapter` is preferred (older: get_peft_model)
        lora_config = LoraConfig(**train_cfg["lora"])
        model.add_adapter(lora_config, adapter_name="lora_adapter")
        print_trainable_parameters(model)

    # Initialize our custom evaluator
    custom_evaluator = CustomEmbeddingEvaluator(
        eval_dataset=eval_dataset,
        embedding_mode="sequence",
        eval_batch_size=training_args.per_device_eval_batch_size,
        optuna_trials=args.optuna_trials if args.use_optuna else 0,
        **eval_cfg,
    )

    # Initialize the trainer
    train_loss = losses.BatchAllTripletLoss(model=model)
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=train_loss,
        evaluator=[custom_evaluator],  # more like evaluators
    )

    # Train the model
    trainer.train()
    trainer.save_model(output_path)
    print(f"Finetuned model saved to {output_path}")


def train_model_or_linear_prober(
    model: SentenceTransformer|LLM,
    dataset: DatasetDict,
    input_key: str,
    label_key: str,
    train_cfg: dict,
    eval_cfg: dict,
):
    """
    Train either the whole model with finetuning or a linear classifier on top of
    the model's sentence embeddings
    """
    # Train the whole model using fine-tuning
    if args.finetune:

        # Ensure input and label keys are correct
        if label_key not in dataset["train"].column_names:
            raise ValueError(f"Label key '{label_key}' not found in the dataset.")
        dataset = dataset.rename_columns({input_key: "text", label_key: "label"})
        dataset = dataset.remove_columns([
            c for c in dataset["train"].column_names if c not in ["text", "label"]
        ])

        # Finetune the model
        finetune_model(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            train_cfg=train_cfg,
            eval_cfg=eval_cfg,
        )

    # Using linear probing with a scikit classifier (no weight update)
    else:

        # Extract sentence embeddings for all splits
        extract_fn = partial(compute_patient_card_embeddings, model=model, input_key=input_key)
        with torch.no_grad():
            dataset = dataset.map(
                function=extract_fn,
                batch_size=512,  # train_cfg["eval_batch_size"],
                batched=True,
                desc="Extracting embeddings from patient cards",
            )

        # Train a linear prober from the model's sentence embeddings
        train_classifier(
            dataset=dataset,
            feature_key="patient_card_embedding",
            label_key=label_key,
        )


def compute_patient_card_embeddings(
    batch: dict,
    model: LLM,  # in "embed" mode
    input_key: str,
):
    """
    Compute sentence embeddings from a batch of patient cards
    """
    # patient_card_sentences = batch[input_key]
    # patient_card_embedding = model.encode(patient_card_sentences, show_progress_bar=False)
    outputs = model.embed(batch[input_key])
    patient_card_embedding = [output.outputs.embedding for output in outputs]

    return {"patient_card_embedding": patient_card_embedding}


def train_classifier(
    dataset: DatasetDict,
    feature_key: str,
    label_key: str,
):
    """
    Train a simple logistic regression classifier on patient embeddings
    """
    dataset.set_format(type="numpy", columns=[feature_key, label_key])
    X_train = np.array(dataset["train"][feature_key])
    y_train = np.array(dataset["train"][label_key])
    X_valid = np.array(dataset["validation"][feature_key])
    y_valid = np.array(dataset["validation"][label_key])

    classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_valid)
    y_score = classifier.predict_proba(X_valid)[:, 1]

    # Use CustomEvaluator to compute metrics
    n_classes = len(np.unique(y_valid))
    if n_classes == 2:
        y_pred = CustomEmbeddingEvaluator._get_preds_from_best_threshold(y_true=y_valid, y_score=y_score)
        auprc_score = average_precision_score(y_valid, y_score)
        auroc_score = roc_auc_score(y_valid, y_score)
        print(f"ROC AUC Score: {auroc_score:.4f}")
        print(f"AUPRC Score: {auprc_score:.4f}")

    ece = CustomEmbeddingEvaluator._expected_calibration_error(y_true=y_valid, y_score=y_score)
    print(f"Expected Calibration Error: {ece:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_valid, y_pred))


if __name__ == "__main__":

    # Common command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("-sl", "--use_supervised_labels", action="store_false", help="Use true labels to evaluate models")
    parser.add_argument("-ph", "--prediction_horizon", default=30, type=int, choices=csts.PREDICTION_HORIZONS, help="Horizon of the infection prediction task for model evaluation")
    parser.add_argument("-ct", "--cutoff_days_train", nargs="+", type=int, default=None, choices=csts.CUTOFF_DAYS, help="Cutoff days for the training dataset, take them all if None")
    parser.add_argument("-cv", "--cutoff_days_valid", nargs="+", type=int, default=None, choices=csts.CUTOFF_DAYS, help="Cutoff days for the evaluation/testing dataset, take them all if None")

    # Arguments that are specific to the current script
    parser.add_argument("-qc", "--qwen_config_path", default="configs/qwen_embedder.yaml", help="Path to the config file")
    parser.add_argument("-ft", "--finetune", action="store_true", help="Finetune the model")

    # Train the model
    args = parser.parse_args()
    main(args)
