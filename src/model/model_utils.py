import math
import torch
import torch.nn as nn


class PatientEmbeddingLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        vocabs: dict[str, dict[str, int]],
    ):
        """ Layer to embed patient data:
            - Entities, attributes, and values are embedded via learned vocabs
            - Times are embedded using rotary embeddings
            - Final embedding is the sum of all component embeddings

            Args:
                embedding_dim: output dimension of the embeddings
                vocabs: vocabs for text data
        """
        super().__init__()
        self.vocabs = vocabs
        self.embedding_dim = embedding_dim

        # Create embedding layers for each vocabulary
        self.token_embedding_dict = nn.ModuleDict({
            key: nn.Embedding(len(vocab), embedding_dim)
            for key, vocab in vocabs.items()
        })

        # Rotary embedding for time
        self.time_embedding = TimeEmbedding(embedding_dim)

    def forward(self, **kwargs) -> torch.Tensor:
        """ Forward pass for the patient embedding layer

            Args:
                kwargs: A dictionary of input tensors
                - Expects keys "entity", "attribute", "value_binned", "time"
                - Each value is a tensor of shape (batch_size, seq_len)

            Returns:
                Combined patient embedding tensor
        """
        # Check input arguments
        if not all(key in kwargs for key in self.token_embedding_dict):
            raise KeyError(f"Required inputs: {list(self.token_embedding_dict.keys())}")
        if "time" not in kwargs:
            raise KeyError("Required input: time")
        
        # Sum the embeddings from all vocabulary-based features
        token_embeddings = sum(
            self.token_embedding_dict[key](sequence)
            for key, sequence in kwargs.items()
            if key in self.token_embedding_dict
        )

        # Apply time embeddings using the time input
        time_embeddings = self.time_embedding(kwargs["time"])

        return token_embeddings + time_embeddings


class TimeEmbedding(nn.Module):
    """ Creates a sinusoidal time embedding for a tensor of integers.
    """
    def __init__(
        self,
        embedding_dim: int,
        time_scale: int=10000,
    ):
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("Embedding dimension must be even")
        self.half_dim = embedding_dim // 2
        self.ratio = math.log(time_scale) / self.half_dim

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        """ Args: times: tensor of any shape containing integer time differences
            Returns: tensor of shape (*days_tensor.shape, embedding_dim)
        """
        freq_indices = torch.arange(self.half_dim, device=times.device)
        times_scaled = times.unsqueeze(-1) * torch.exp(-self.ratio * freq_indices)
        embeddings = torch.cat([torch.sin(times_scaled), torch.cos(times_scaled)], dim=-1)

        return embeddings
