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
from dataclasses import dataclass
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
        
        # outputs["infection_labels"] = kwargs.get("infection_labels")
        return outputs
        

@dataclass
class PatientDataCollatorForLanguageModelling(DataCollatorMixin):
    """ Data collator used for the PatientEmbedding-based language model
        Modified from transformers.data.data_collator.py
    
    Args:
        mlm (bool): whether or not to use masked language modeling
        mlm_probability(float): probability with which tokens are mask randomly
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
    
    def torch_call(
        self,
        samples: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """ Collate patient embedding samples and create mlm labels for training
        """
        # Take labels out for batching
        label_names = [n for n in samples[0].keys() if "label" in n]
        external_labels = {n: [s.pop(n) for s in samples] for n in label_names}
        
        # Control each sample for sequence length
        effective_num_token_max = self.num_tokens_max - 2   # for bos and eos tokens
        for batch_idx, _ in enumerate(samples):
            seq_len = next(iter(samples[batch_idx].values())).shape[0]
            if seq_len > effective_num_token_max:
                
                # Randomly select a starting index for slicing
                start_idx = random.randint(0, seq_len - effective_num_token_max)
                end_idx = start_idx + effective_num_token_max
                
                # Slice all tensors in the sample by keys (with the same slicing)
                for data_key in samples[batch_idx]:
                    random_slice = samples[batch_idx][data_key][start_idx:end_idx]
                    samples[batch_idx][data_key] = random_slice
        
        # Add token ids for beginning and end of sequence
        for batch_idx, _ in enumerate(samples):
            for data_key in samples[batch_idx]:
                to_enclose = samples[batch_idx][data_key]
                enclosed = self.add_bos_eos_ids(to_enclose, data_key)
                samples[batch_idx][data_key] = enclosed

        # Pad all sequences needed for the embedding layer
        padded_sequences = {}
        features_to_pad = ["entity", "attribute", "value_binned", "time"]
        for key in features_to_pad:
            sequences = [s[key] for s in samples]
            padding_value = 0.0 if key == 'time' else float(self.pad_id)
            padded_sequences[key] = pad_sequence(
                sequences, batch_first=True, padding_value=padding_value
            )
        attention_mask = (padded_sequences["entity"] != self.pad_id).long()

        # The language modeling task is on the 'value_binned' feature
        values_to_predict = padded_sequences["value_binned"]

        # Mask input and keep masked value labels for masked language modelling
        if self.mlm:
            assert self.num_mlm_labels is not None, "Define the number of labels for mlm"
            masked_values, labels = self.masked_modelling(values_to_predict)
            padded_sequences["value_binned"] = masked_values
        
        # Compute labels from unmasked sequences for causal language modelling
        else:
            labels = self.causal_modelling(values_to_predict)

        # Assemble the final batch as the model's forward() method expects
        batch = {
            "input_dict": padded_sequences,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        # Put external labels back in, after all batch processing happened
        for label_name in external_labels:
            batch[label_name] = external_labels[label_name]

        return batch
        
    def add_bos_eos_ids(
        self,
        sequence: torch.Tensor,
        data_key: str,
    ) -> torch.Tensor:
        """ Add bos and eos token ids or first and last time to a sequence
            given the data it contains
        """
        if data_key == "time":
            to_add = [sequence[0].unsqueeze(0), sequence[-1].unsqueeze(0)]
        else:
            to_add = [torch.tensor([self.bos_id]), torch.tensor([self.eos_id])]
        
        return torch.cat([to_add[0], sequence, to_add[-1]], dim=0)
    
    def masked_modelling(
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
    
    def causal_modelling(
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
    