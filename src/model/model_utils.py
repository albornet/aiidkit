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


def test_time_embedding_visualization():
    """ Tests and visualizes the TimeEmbedding layer
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.manifold import TSNE
    print("Running TimeEmbedding visualization test...")

    # Test parameters
    embedding_dim = 512
    time_range_start = -10000
    time_range_end = 10000
    step = 25  # use a step to avoid too many points, which slows down t-SNE

    # Generate time embeddings
    time_embedding_layer = TimeEmbedding(embedding_dim)
    time_inputs = torch.arange(time_range_start, time_range_end + 1, step).float()
    with torch.no_grad():  # no need to track gradients
        embeddings = time_embedding_layer(time_inputs)
    
    # Reduce dimensionality with t-SNE
    n_components = 3
    tsne = TSNE(n_components=n_components, perplexity=30, max_iter=5000)
    reduced_embeddings = tsne.fit_transform(embeddings.numpy())
    
    # Plot the results
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(projection="3d" if n_components > 2 else "2d")
    scatter = ax.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        reduced_embeddings[:, 2] if n_components > 2 else None,
        s=10,
        c=time_inputs.numpy(),
        cmap="viridis",
        alpha=0.8,
    )

    # Save polished figure
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Time Value", fontsize=12, weight="bold")
    ax.set_title(
        f"t-SNE Visualization of Time Embeddings (Dim={embedding_dim})", 
        fontsize=16, 
        weight="bold"
    )
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    if n_components > 2:
        ax.set_zlabel("t-SNE Dimension 3", fontsize=12)
    plt.savefig("time_embedding_tsne.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    test_time_embedding_visualization()