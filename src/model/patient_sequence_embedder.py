import torch
import torch.nn as nn
import sentence_transformers

...

from src.model.model_utils import PatientEmbeddingLayer
import src.constants as constants
csts = constants.ConstantsNamespace()


class PatientSequenceEmbeddingModel(nn.Module):
    def __init__(
        self,
        original_model_id: str,
    ):
        super().__init__()
        ...
