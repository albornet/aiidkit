import os
import yaml
from transformers import Trainer, TrainingArguments

from src.model.patient_token_embedder import (
    PatientTokenEmbeddingModel,
    PatientDataCollatorForLanguageModelling,
)
from src.model.evaluate_models import  compute_silhouette_score
from src.data.process.process_utils import (
    build_huggingface_patient_dataset,
    get_formatted_patient_dataset,
)

import src.constants as constants
csts = constants.ConstantsNamespace()
model_cfg_path = os.path.join(csts.MODEL_CONFIG_DIR_PATH, "patient_token_embedder.yaml")
with open(model_cfg_path, "r") as f:
    model_cfg = yaml.safe_load(f)


def main():
    """ Train a patient embedding model with MLM (or causal LM?)
    """
    # Load formatted dataset
    build_huggingface_patient_dataset(
        input_data_dir=csts.PREPROCESSED_DIR_PATH,
        output_data_dir=csts.HUGGINGFACE_DIR_PATH,
    )
    dataset, _, vocabs = get_formatted_patient_dataset(
        huggingface_dataset_path=csts.HUGGINGFACE_DIR_PATH,
        base_vocab=model_cfg["DEFAULT_BASE_VOCAB"],
    )

    # Initialize the full patient embedding model
    model = PatientTokenEmbeddingModel(
        original_model_id=model_cfg["ORIGINAL_MODEL_ID"],
        language_model_type=model_cfg["ORIGINAL_MODEL_TASK"],
        vocabs=vocabs,
        llm_kwargs=model_cfg["ORIGINAL_MODEL_PARAMS"],
    )

    # Training arguments
    training_arguments = TrainingArguments(
        output_dir=csts.RESULT_DIR_PATH,
        logging_dir=os.path.join(csts.RESULT_DIR_PATH, "logs"),
        remove_unused_columns=False,
        **model_cfg["DEFAULT_TRAINING_ARGUMENTS"],
        load_best_model_at_end=True,
        metric_for_best_model="silhouette_score",
        greater_is_better=True,
        report_to="wandb",
    )

    # Initialize data collator, which implements MLM or CausalLM
    data_collator = PatientDataCollatorForLanguageModelling(
        mlm=(model_cfg["ORIGINAL_MODEL_TASK"] == "masked"),
        num_mlm_labels=len(vocabs["value_binned"]),
        num_tokens_max=model.num_tokens_max,
    )

    # Define trainer object
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_silhouette_score,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # Train the model
    trainer.train()
    print("Training complete")


if __name__ == "__main__":
    main()