import random
import torch
import torch.nn as nn
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoConfig,
)
from transformers.modeling_outputs import (
    MaskedLMOutput,
    CausalLMOutput,
)
from transformers.data.data_collator import DataCollatorMixin
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass, field
from typing import Union, Optional
from src.model.model_utils import PatientEmbeddingLayer


class PatientTokenEmbeddingModel(nn.Module):
    def __init__(
        self,
        original_model_id: str,
        language_model_type: str,
        vocabs: dict[str, dict[str, int]],
        tie_embedding_to_decoder: bool=False,
        llm_kwargs: dict[str, int]=None,
    ):
        super().__init__()
        assert language_model_type in ["masked", "causal"],\
            "language_model_type must be masked or causal"
        
        # Load parametesr of the given model and modify them as required
        config = AutoConfig.from_pretrained(original_model_id)
        if llm_kwargs is not None:
            for value, key in llm_kwargs.items():
                setattr(config, value, key)

        # Initialize model from an original pre-trained LLM and return hidden states
        if language_model_type == "masked":
            self.llm = AutoModelForMaskedLM.from_config(config)
        elif language_model_type == "causal":
            self.llm = AutoModelForCausalLM.from_config(config)
        self.hidden_size = self.llm.config.hidden_size
        self.num_tokens_max = self.llm.config.max_position_embeddings
        
        # Create embedding layer and replace the LLM one to prevent incompatibility
        self.embedding_layer = PatientEmbeddingLayer(
            embedding_dim=self.hidden_size,
            vocabs=vocabs,
        )
        
        # Modify the LLM head (classifier) to match the number of value tokens
        num_value_tokens = len(vocabs["value_binned"])
        self.llm.config.vocab_size = num_value_tokens
        new_decoder_layer = nn.Linear(
            in_features=self.hidden_size, 
            out_features=num_value_tokens,
            bias=True,
        )
        self.llm.set_output_embeddings(new_decoder_layer)
        
        # Make all weights contiguous and untie any tied weight
        self.apply(self._reset_weights_fn)  # "apply" is recursive
        
        # Tie output decoder linear weights to input value embedding weights
        if tie_embedding_to_decoder:
            output_embeddings = self.llm.get_output_embeddings()
            input_embeddings = self.embedding_layer.value_embedding
            output_embeddings.weight = input_embeddings.weight
        
    def _reset_weights_fn(self, module: nn.Module) -> None:
        """ Make any weight or bias parameter contiguous and untie any shared
            weights in a module by cloning the contiguous parameters
        """
        try:
            if hasattr(module, "weight") and module.weight is not None:
                module.weight = nn.Parameter(module.weight.contiguous().clone())
            if hasattr(module, "bias") and module.bias is not None:
                module.bias = nn.Parameter(module.bias.contiguous().clone())
        except RuntimeError:
            if not hasattr(self, "already_printed_warnings"):
                self.already_printed_warnings = set()
            warning = "Module %s was not reset!" % module._get_name()
            if warning not in self.already_printed_warnings:
                print(warning)
            self.already_printed_warnings.add(warning)
            
    def forward(
        self,
        input_dict: dict[str: list[str|int]],
        attention_mask: Optional[torch.Tensor]=None,
        head_mask: Optional[torch.Tensor]=None,
        labels: Optional[torch.LongTensor]=None,
        output_attentions: Optional[bool]=None,
        output_hidden_states: Optional[bool]=None,
        return_dict: Optional[bool]=None,
        **kwargs,
    ) -> Union[MaskedLMOutput, CausalLMOutput, tuple[torch.Tensor, ...]]:
        """ Masked LM forward function adapted for patient embeddings
        
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss
            Indices should be in [-100, 0, ..., config.vocab_size]
            Tokens with indices set to -100 are ignored (masked), the loss is
            only computed for the tokens with labels in [0, ..., config.vocab_size].
        """
        # Generate patient embeddings
        patient_embeddings = self.embedding_layer(**input_dict)

        # Forward to the LLM model using inputs_embeds
        outputs = self.llm(
            input_ids=None,  # inputs_embeds is used instead
            inputs_embeds=patient_embeddings,
            labels=labels,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Need to reshape and to save some memory
        if outputs.hidden_states is not None:
            last_hidden_state = outputs.hidden_states[-1]
            if last_hidden_state.ndim == 3:  # normal case
                outputs.hidden_states = (last_hidden_state,)
            else:  # exceptions for some models when using flash-attn-2
                batch_size, seq_len = attention_mask.shape
                hidden_size = last_hidden_state.shape[-1]
                unflattened = torch.zeros(
                    batch_size, seq_len, hidden_size,
                    device=last_hidden_state.device,
                    dtype=last_hidden_state.dtype
                )
                unflattened[attention_mask.bool()] = last_hidden_state
                outputs.hidden_states = (unflattened,)
        
        return outputs
        

@dataclass
class PatientDataCollatorForLanguageModelling(DataCollatorMixin):
    """ Data collator used for the PatientEmbedding-based language model
        Modified from transformers.data.data_collator.py
    """
    mlm: bool=True
    pad_id: int=0
    mask_id: int=1
    bos_id: int=2
    eos_id: int=3
    unk_id: int=4
    num_tokens_max: int=512
    num_mlm_labels: Optional[int]=None
    mlm_probability: float=0.15
    return_tensors: str="pt"
    features_to_pad: list[str] = field(
        default_factory=lambda: ["entity", "attribute", "value_binned", "time"]
    )
    task_label_names: Optional[list[str]] = field(
        default_factory=lambda: ["infection_labels", "infection_events"],
        metadata={"help": "The list of keys to treat as labels."}
    )

    def _separate_labels(self, samples: list[dict]) -> tuple[list[dict], dict]:
        """ Separate non-feature labels (e.g., for classification) from the samples
        """
        # Identify which of the specified label names are present in the batch
        present_keys = samples[0].keys()
        keys_to_extract = [key for key in self.task_label_names if key in present_keys]
        if not self.task_label_names or not samples: return samples, {}
        if not keys_to_extract: return samples, {}

        # Extract the labels by popping them out of each sample
        external_labels = {
            key: [s.pop(key) for s in samples] for key in keys_to_extract
        }

        return samples, external_labels

    def _truncate_sequences(self, samples: list[dict]) -> list[dict]:
        """ Truncate samples longer than the maximum allowed sequence length
        """
        effective_max_len = self.num_tokens_max - 2  # for bos and eos tokens
        for i, sample in enumerate(samples):
            seq_len = next(iter(sample.values())).shape[0]
            if seq_len > effective_max_len:
                start_idx = random.randint(0, seq_len - effective_max_len)
                end_idx = start_idx + effective_max_len
                samples[i] = {key: val[start_idx:end_idx] for key, val in sample.items()}
        
        return samples

    def _add_bos_eos_ids(self, sequence: torch.Tensor, data_key: str) -> torch.Tensor:
        """ Add bos and eos token ids or first and last time to a sequence
            given the data it contains
        """
        if data_key == "time":
            to_add = [sequence[0].unsqueeze(0), sequence[-1].unsqueeze(0)]
        else:
            to_add = [torch.tensor([self.bos_id]), torch.tensor([self.eos_id])]
        
        return torch.cat([to_add[0], sequence, to_add[-1]], dim=0)
        
    def _add_special_tokens(self, samples: list[dict]) -> list[dict]:
        """ For now, only add BOS and EOS tokens to each sequence in all samples
        """
        for i, sample in enumerate(samples):
            samples[i] = {k: self._add_bos_eos_ids(v, k) for k, v in sample.items()}
        
        return samples

    def _pad_batch(self, samples: list[dict]) -> tuple[dict, torch.Tensor]:
        """ Pad all sequences in the batch to the same length
        """
        padded_batch = {}
        for key in self.features_to_pad:
            sequences = [s[key] for s in samples]
            padding_value = 0.0 if key == 'time' else float(self.pad_id)
            padded_batch[key] = pad_sequence(sequences, batch_first=True, padding_value=padding_value)
        
        attention_mask = (padded_batch["entity"] != self.pad_id).long()
        
        return padded_batch, attention_mask
    
    def _preprocess_batch(self, samples: list[dict]) -> tuple[dict, torch.Tensor]:
        """ Pipeline to truncate, encapsulate, pad samples, and compute attention masks
        """
        samples = self._truncate_sequences(samples)
        samples = self._add_special_tokens(samples)
        padded_batch, attention_mask = self._pad_batch(samples)
        
        return padded_batch, attention_mask

    def torch_call(self, samples: list[dict]) -> dict:
        """ Collate patient embedding samples and create labels for LM training
        """
        # Preprocess the input samples
        samples, external_labels = self._separate_labels(samples)
        padded_batch, attention_mask = self._preprocess_batch(samples)

        # Prepare labels for the language modeling task
        values_to_predict = padded_batch["value_binned"]
        if self.mlm:
            masked_values, labels = self._masked_modelling(values_to_predict)
            padded_batch["value_binned"] = masked_values
        else:
            labels = self._causal_modelling(values_to_predict)

        # Assemble the final batch
        batch = {
            "input_dict": padded_batch,
            "labels": labels,
            "attention_mask": attention_mask,
        }
        batch.update(external_labels)

        return batch
    
    def _masked_modelling(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ Prepare masked tokens inputs and labels for masked language modeling
            Modified from transformers.data.data_collator.py
        """
        # Prepare labels and mask array
        labels = inputs.clone()  # labels are the unmasked version of inputs
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        
        # Sample tokens in each sequence for mlm training, using mlm_probability
        probability_matrix.masked_fill_(labels == self.pad_id, value=0.0)  # pad tokens never masked
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # special code to only compute loss on masked tokens
        
        # 80% of the time, replace masked input tokens with mask token
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_id
        
        # 10% of the time, replace masked input tokens with random value token id
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.num_mlm_labels, labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        
        return inputs, labels
    
    def _causal_modelling(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """ Prepare labels for causal language modeling by shifting tokens to the right
            Modified from transformers.data.data_collator.py
        """
        # Create labels from inputs (no need of manual shift: done in trainer)
        labels = inputs.clone()
                
        # Ensure original padding tokens are also ignored in the labels
        if self.pad_id is not None:
            labels[inputs == self.pad_id] = -100
            
        return labels
    