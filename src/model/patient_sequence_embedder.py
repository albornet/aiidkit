import os
import torch
import torch.nn as nn
from safetensors.torch import save_file
from dataclasses import dataclass
from sentence_transformers.trainer import SentenceTransformerTrainer
from src.model.patient_token_embedder import (
    PatientTokenEmbeddingModel,
    PatientDataCollatorForLanguageModelling,
)


class PatientTokenEmbeddingModule(nn.Module):
    """ Simple wrapper for the core patient token embedding model, formatting its output
        for the pooling layer of the sentence tranformer model
    """
    def __init__(self, token_embedder: PatientTokenEmbeddingModel):
        super().__init__()
        self.token_embedder = token_embedder
        self.pad_id = 0

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """ Add required "token_embeddings" and "attention_mask" keys for the pooling layer
        """
        # Separate the attention mask from the actual features
        attention_mask = features.pop("attention_mask")

        # Pass the batch through your core model
        outputs = self.token_embedder(
            input_dict=features,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Update the dictionary with the outputs the pooling layer expects
        features.update({
            "token_embeddings": outputs.hidden_states[-1],
            "attention_mask": attention_mask,
        })

        return features
    
    def save(self, output_path: str):
        model_path = os.path.join(output_path, "model.safetensors")
        save_file(self.token_embedder.state_dict(), model_path)


class PatientTrainer(SentenceTransformerTrainer):
    """ Custom sentence transformer trainer that overrides the default loss computation
    """
    def __init__(self, *args, **kwargs):
        """ Custom __init__ to handle a model without a tokenizer
        """
        model = kwargs.get("model")
        if model is not None and not hasattr(model, "tokenizer"):
            model.tokenizer = None
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """ Loss function (MultipleNegativesRankingLoss) does not require labels
        """
        # Training mode: inputs is [anchor, positive]
        if isinstance(inputs, list):
            loss = self.loss(inputs, labels=None)
            return (loss, {}) if return_outputs else loss

        # Evaluation mode: input is a single batch dictionary
        else:
            outputs = model(inputs)
            return (torch.tensor(0.0, device=model.device), outputs)


@dataclass
class PatientDataCollatorForSequenceEmbedding(PatientDataCollatorForLanguageModelling):
    """ Data collator for sequence embedding, preprocessing batches like the LM collator
    """
    # Needed for compatibility with sentence-transformer
    valid_label_columns: list[str]|None=None

    # Overriding parent arguments that are not relevant to the sequence embedding task
    mlm: bool=False
    num_mlm_labels: int|None=None
    mlm_probability: float=0.0

    def _process_and_prepare(self, feature_dicts: list[dict]) -> dict[str, torch.Tensor]:
        """ Prepare a batch with attention mask as a key
        """
        padded_batch, attention_mask = self._preprocess_batch(feature_dicts)
        padded_batch["attention_mask"] = attention_mask

        return padded_batch

    def __call__(
        self,
        features: list[dict[str, dict]]
    ) -> list[dict[str, torch.Tensor]]|dict[str, torch.Tensor]:
        """ Batching function used by the sentence-transformer trainer pipeline
        """
        # Identify if the data collator is being used during training or evaluation
        feature_keys = features[0].keys()
        training_mode = ("anchor" in feature_keys and "positive" in feature_keys)
        
        # Training mode (no labels needed)
        if training_mode:
            anchor_dicts = [feat["anchor"] for feat in features]
            positive_dicts = [feat["positive"] for feat in features]
            padded_anchors = self._process_and_prepare(anchor_dicts)
            padded_positives = self._process_and_prepare(positive_dicts)
            
            return [padded_anchors, padded_positives]
        
        # Evaluation mode (keeping track of labels)
        else:
            samples, external_labels = self._separate_labels(features)
            padded_batch = self._process_and_prepare(samples)
            padded_batch.update(external_labels)

            return padded_batch
        