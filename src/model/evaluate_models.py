import os
import numpy as np
import umap
import hdbscan
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from transformers.trainer_utils import EvalPrediction
import src.constants as constants
csts = constants.ConstantsNamespace()


def plot_and_save_embeddings(
    embeddings_2d: np.ndarray,
    cluster_labels: np.ndarray,
    output_path: str,
):
    """ Save a 2D scatter plot of embeddings, colored by cluster cluster_labels

    Args:
        embeddings_2d (np.ndarray): A 2D array of embeddings.
        cluster_labels (np.ndarray): An array of cluster cluster_labels corresponding to the embeddings.
        output_path (str): The file path where the plot will be saved.
    """
    # Get cluster labels, excluding noise (-1)
    unique_labels = set(cluster_labels)
    cluster_keys = sorted([l for l in unique_labels if l != -1])
    num_clusters = len(cluster_keys)

    # Get the color map
    cmap = plt.cm.get_cmap("Spectral", num_clusters)
    color_map = {key: cmap(i) for i, key in enumerate(cluster_keys)}
    color_map[-1] = (0, 0, 0, 1) # Black for noise
    
    # Plot the reduced embeddings and color clusters
    plt.figure(figsize=(6, 5))
    for k in unique_labels:
        col = color_map[k]
        class_member_mask = (cluster_labels == k)
        xy = embeddings_2d[class_member_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=6 if k != -1 else 3,  # make noise points smaller
            alpha=0.8 if k != -1 else 0.3,  # make noise points more transparent
        )

    num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    plt.title(f"UMAP Projection of Patient Embeddings (HDBSCAN Clusters: {num_clusters})")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(True)
    
    # Save the figure
    plt.savefig(output_path)
    plt.close()  # close the plot to free up memory


def compute_silhouette_score(eval_preds: EvalPrediction):
    """ Compute the silhouette score and plots embeddings from a language model
    """
    # Extract the last layer's hidden states
    # - task_labels shape: (eval_data_size, padded_seq_len)
    # - output_logits shape: (eval_data_size, padded_seq_len, vocab_size)
    # - last_hidden_state shape: (eval_data_size, padded_seq_len, hidden_dim)
    predictions, task_labels = eval_preds
    output_logits, hidden_states = predictions
    last_hidden_state = hidden_states[-1]

    # Compute non-padded token embedding average for each sample
    no_pad_mask = (task_labels != -100).astype(np.float32)
    masked_embeddings = last_hidden_state * np.expand_dims(no_pad_mask, axis=-1)
    sum_embeddings = np.sum(masked_embeddings, axis=1)
    num_tokens = np.sum(no_pad_mask, axis=1, keepdims=True)
    avg_embeddings = sum_embeddings / num_tokens

    # Reduce embedding dimensionality (for clustering and plotting)
    reducer_10d = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.0, random_state=1234)
    reducer_2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=1234)
    reduced_embeddings_10d = reducer_10d.fit_transform(avg_embeddings)
    reduced_embeddings_2d = reducer_2d.fit_transform(avg_embeddings)

    # Cluster reduced embeddings with HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, gen_min_span_tree=True)
    cluster_labels = clusterer.fit_predict(reduced_embeddings_10d)

    # Compute silhouette score
    unique_clusters = np.unique(cluster_labels)
    num_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
    if num_clusters > 1:
        score = silhouette_score(reduced_embeddings_10d, cluster_labels)
    else:
        print(
            f"HDBSCAN found only {num_clusters} cluster(s). "
            "The silhouette score requires at least 2 clusters. Returning 0.0."
        )
        score = 0.0

    # Visualization of 2D-reduced embeddings, with cluster labels
    plot_dir = os.path.join(csts.RESULT_DIR_PATH, "evaluate_plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_and_save_embeddings(
        embeddings_2d=reduced_embeddings_2d,
        cluster_labels=cluster_labels,
        output_path=os.path.join(plot_dir, "embedding_plot.png"),
    )
    
    return {"silhouette_score": score}


# def preprocess_logits_for_metrics(logits, labels):
#     """ Align logits and labels for metric computation
#     """
#     if isinstance(logits, tuple):
#         # The first element of the tuple is the prediction scores.
#         logits = logits[0]
#     # For causal LM, the last logit is not needed for prediction
#     # and the first label is not predicted.
#     return logits[:, :-1, :].argmax(dim=-1), labels[:, 1:]