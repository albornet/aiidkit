import wandb
import numpy as np
import optuna
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)
import umap
import hdbscan
import plotly.graph_objects as go
from plotly.colors import qualitative
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score
from transformers.trainer_utils import EvalPrediction
from datasets import Dataset

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
import src.constants as constants
csts = constants.ConstantsNamespace()


def preprocess_logits_for_metrics(logits, labels):
    """ Align logits and labels for metric computation (useful for causal language models)
    """
    # The first element of the tuple is the prediction scores
    if isinstance(logits, tuple): logits = logits[0]

    # For causal LM, last logit not needed for prediction and first label not predicted
    return logits[:, :-1, :].argmax(dim=-1), labels[:, 1:]


def objective(trial: optuna.Trial, embeddings: np.ndarray) -> float:
    """ Objective function for clustering UMAP-reduced embeddings with HDBSCAN
    """
    params = {
        # UMAP parameters
        "n_components": trial.suggest_int("n_components", 2, 20),
        "n_neighbors": trial.suggest_int("n_neighbors", 5, 50),
        "min_dist": trial.suggest_float("min_dist", 0.0, 0.5),
        # HDBSCAN parameters
        "min_cluster_size": trial.suggest_int("min_cluster_size", 5, 60),
    }
    params["min_samples"] = trial.suggest_int("min_samples", 1, params["min_cluster_size"])

    # Pass all params; the next function will sort them out
    reduced_embeddings, cluster_labels = reduce_and_cluster(embeddings, **params)
    
    # Calculate and return the score
    num_clusters = len(np.unique(cluster_labels)) - (1 if -1 in np.unique(cluster_labels) else 0)
    return silhouette_score(reduced_embeddings, cluster_labels) if num_clusters > 1 else -1.0


def reduce_and_cluster(
    embeddings: np.ndarray,
    compute_clusters: bool=True,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray|None]:
    """ Reduce a set of embeddings with UMAP and cluster them with HDBSCAN
    """
    # Filter kwargs to create dictionaries for each function
    UMAP_KEYS = {"n_components", "n_neighbors", "min_dist"}
    HDBSCAN_KEYS = {"min_cluster_size", "min_samples"}
    umap_args = {k: v for k, v in kwargs.items() if k in UMAP_KEYS}
    hdbscan_args = {k: v for k, v in kwargs.items() if k in HDBSCAN_KEYS}

    # Compute dimensionality-reduced embeddings
    reduced_embeddings = umap.UMAP(**umap_args, random_state=1234).fit_transform(embeddings)
    if not compute_clusters:
        return reduced_embeddings, None
    
    # Compute clusters from dimensionality-reduced embeddings
    cluster_labels = hdbscan.HDBSCAN(**hdbscan_args).fit_predict(reduced_embeddings)
    return reduced_embeddings, cluster_labels


def plot_clustered_embeddings(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
) -> go.Figure:
    """ Interactive 2D scatter plot of embeddings colored by cluster labels
    """
    # Create a color mapping for each cluster in a Plotly-compatible RGB string format
    unique_labels = set(labels)
    cluster_keys = sorted([l for l in unique_labels if l != -1])
    palette = qualitative.Plotly
    color_map = {-1: "#000000"}  # black
    for i, key in enumerate(cluster_keys):
        color_map[key] = palette[i % len(palette)]

    # Plot samples with corresponding label colors
    fig = go.Figure()
    for k in sorted(list(unique_labels), key=lambda x: (x != -1, x)):  # noise last
        class_member_mask = (labels == k)
        xy = embeddings_2d[class_member_mask]
        fig.add_trace(go.Scatter(
            x=xy[:, 0], y=xy[:, 1], mode="markers", name=str(k),
            marker=dict(
                color=color_map[k], line=dict(color="black", width=1),
                size=3 if k == -1 else 6, opacity=0.3 if k == -1 else 0.8,
            )
        ))

    # Polish figure
    fig.update_layout(
        xaxis_title="UMAP-1", yaxis_title="UMAP-2", legend_title="Clusters",
        yaxis_scaleanchor="x", width=1080, height=1080,
        margin=dict(l=40, r=40, b=40, t=80),
    )

    return fig


class CustomMetricComputer:
    """ Class used by huggingface's trainer to compute embedding metrics
    """
    def __init__(self, eval_dataset: Dataset):
        self.infection_labels = eval_dataset["infection_labels"]
        
    def __call__(self, eval_preds: EvalPrediction) -> dict[str, float]:
        """ Evaluate embeddings with different cluster-related metrics
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

        # Identify the best set of hyperparameters for silhouette score computation
        study = optuna.create_study(direction="maximize", sampler=TPESampler())
        objective_fn = lambda trial: objective(trial, avg_embeddings)
        study.optimize(objective_fn, n_trials=25, show_progress_bar=True)
        best_params = study.best_params
        
        # Compute silhouette score using the best set of hyper-parameters
        reduced_embeddings, cluster_labels = reduce_and_cluster(avg_embeddings, **best_params)
        score = silhouette_score(reduced_embeddings, cluster_labels)

        # Create a 2-dimensional visualization
        vis_params = best_params.copy()
        vis_params["n_components"] = 2
        reduced_embeddings_2d, _ = reduce_and_cluster(avg_embeddings, compute_clusters=False, **vis_params)
        cluster_plot = plot_clustered_embeddings(reduced_embeddings_2d, cluster_labels)
        
        # Compute infection-label-related metrics
        truth_labels = self.get_first_infection_type(days_low=0, days_high=300)
        ami_score = adjusted_mutual_info_score(truth_labels, cluster_labels)
        truth_plot = plot_clustered_embeddings(reduced_embeddings_2d, truth_labels)
        
        # Log plots directly to wandb
        if wandb.run:
             wandb.log({"cluster_plot": cluster_plot, "cluster_plot_truth": truth_plot})
        
        # Return only the numerical, serializable metrics
        return {"silhouette_score": score, "ami_score": ami_score}
    
    def get_first_infection_type(
        self,
        days_low: int=0,
        days_high: int=None,
    ) -> list[str]:
        """ Get the type of the first infection, given a period of time constrained
            by days_low and days_high, which refer to the transplantation date
        """
        truth_labels = []
        for labels in self.infection_labels:
            label = "Healthy"
            if len(labels["infection_time"]) > 0:
                infection_time = labels["infection_time"][0]
                lower_ok = True if days_low is None else infection_time > days_low
                upper_ok = True if days_high is None else infection_time <= days_high

                if lower_ok and upper_ok:
                    label = labels["infection_type"][0]

            truth_labels.append(label.split(" Infection")[0])

        return np.array(truth_labels, dtype=str)
