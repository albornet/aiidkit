import gc
import yaml
import argparse
import torch
import optuna
import wandb
from pathlib import Path
from .pretrain_token_model import run_training
from src import constants as csts
csts = csts.ConstantsNamespace()

# TAKE HOME MESSAGES FROM OPTUNA RUNS:
# - Uncertainty weighting doesn't seem to change much (but not sure)
# - Using pretrained embeddings for input layer is beneficial


def main(args):
    """
    Identify the best set of parameters for a given training script
    """
    # Print the configuration and arguments
    print(f"Running Optuna tuning with the following fixed hyper-parameters:")
    print(f"  - Use supervised labels: {args.use_supervised_labels} (TODO: CHECK THIS)")
    print(f"  - Prediction horizon: {args.prediction_horizon}")
    print(f"  - Cutoff days (training): {args.cutoff_days_train}")
    print(f"  - Cutoff days (evaluation): {args.cutoff_days_valid}")
    print(f"  - Use pretrained embeddings for input layer: {args.use_pretrained_embeddings_for_input_layer}")

    # Load the base configuration from the YAML file
    with open(args.pretrain_config_path, "r") as f:
        config = yaml.safe_load(f)

    # Create optuna storage path and directory
    run_id = get_run_id()
    db_dir = Path(config["TRAINING_ARGUMENTS"]["output_dir"]) / run_id
    db_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{db_dir / 'optuna_results.db'}"

    # The direction must match what is set in the training script
    study = optuna.create_study(
        direction="maximize",
        study_name="pretrain_token_model_optuna",
        sampler=optuna.samplers.TPESampler(),
        storage=storage,
    )

    # Define the objective function
    study.optimize(
        func=objective_fn,
        n_trials=args.n_trials,
    )


def objective_fn(
    trial: optuna.Trial,
) -> float:
    """
    Optuna objective function for the graining script, will modify configuration,
    train the model, and return the best metric, based on the validation dataset
    """
    # Load the base configuration from the YAML file
    with open(args.pretrain_config_path, "r") as f:
        config = yaml.safe_load(f)

    # Prior to sampling, set configuration elements changing with command line arguments
    config["MODEL_ARGUMENTS"]["use_pretrained_embeddings_for_input_layer"] = \
        args.use_pretrained_embeddings_for_input_layer

    # Sample hyper-parameters for training and model configuration dictionaries
    config["TRAINING_ARGUMENTS"] = sample_train_hyperparameters(trial, config["TRAINING_ARGUMENTS"])
    config["MODEL_ARGUMENTS"] = sample_model_hyperparameters(trial, config["MODEL_ARGUMENTS"])

    # Modify the output_dir for this specific trial to avoid overwriting results
    trial_id = f"trial_{trial.number}"
    run_id = get_run_id()
    new_output_dir = Path(config["TRAINING_ARGUMENTS"]["output_dir"]) / run_id / trial_id
    config["TRAINING_ARGUMENTS"]["output_dir"] = str(new_output_dir)
    config["TRAINING_ARGUMENTS"]["run_name"] = trial_id

    # Gain computation time by skipping clustering analysis during evaluation
    config["EVALUATION_ARGUMENTS"]["do_clustering_analysis"] = False

    # Run training and collect best metric
    try:
        best_run_metric = run_training(config, args)

    # Return low metric value in case of invalid hyper-parameters or GPU memory issues
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}. Returning 0.0 as metric.")
        return 0.0  # <- only works for metrics to maximize and > 0.0!

    # Reset wandb logging and free up RAM to have a fresh run for the next trial
    finally:

        # If wandb is being used in this trial, finish the run for the next one
        if wandb.run is not None:
            wandb.finish()

        # Free up cpu and gpu RAM
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return best_run_metric


def get_run_id() -> str:
    """
    Generate a unique run id based on key command line arguments
    """
    pe = int(args.use_pretrained_embeddings_for_input_layer)
    sl = int(args.use_supervised_labels)
    ph = int(args.prediction_horizon)
    ct = "-".join(map(str, args.cutoff_days_train)) if args.cutoff_days_train else "all"
    cv = "-".join(map(str, args.cutoff_days_valid)) if args.cutoff_days_valid else "all"

    return f"pe{pe}_sl{sl}_ph{ph}_ct{ct}_cv{cv}"


def sample_train_hyperparameters(
    trial: optuna.Trial,
    train_cfg: dict[str, str],
) -> dict[str, str]:
    """
    Select hyper-parameters for training argument config dictionary
    """
    train_cfg["learning_rate"] = \
        trial.suggest_float("learning_rate", 5e-6, 5e-4, log=True)
    train_cfg["weight_decay"] = \
        trial.suggest_float("weight_decay", 0.01, 0.3, log=True)
    train_cfg["warmup_steps"] = \
        trial.suggest_int("warmup_steps", 500, 2500, step=250)

    return train_cfg


def sample_model_hyperparameters(
    trial: optuna.Trial,
    model_cfg: dict[str, int|float|str],
) -> dict[str, int|float|str]:
    """
    Select hyper-parameters for model argument config dictionary

    Note: some hyper-parameter combinations might be invalid, but in that
    case, run_training will raise an error and metric will be low
    """
    # First, select hidden size, which is restricted is the sentence model is used
    if model_cfg["use_pretrained_embeddings_for_input_layer"]:
        model_cfg["original_model_params"]["hidden_size"] = 768
        trial.set_user_attr("hidden_size", 768)
    else:
        model_cfg["original_model_params"]["hidden_size"] = \
            trial.suggest_categorical("hidden_size", [128, 192, 256, 384, 512, 768])

    # Then, select other model parameters
    model_cfg["use_positional_encoding_for_input_layer"] = \
        bool(trial.suggest_categorical("use_positional_encoding_for_input_layer", [0, 1]))
    model_cfg["original_model_params"]["num_attention_heads"] = \
        trial.suggest_categorical("num_attention_heads", [4, 6, 8, 12, 16])
    model_cfg["original_model_params"]["num_hidden_layers"] = \
        trial.suggest_categorical("num_hidden_layers", [4, 6, 8, 12, 16, 22])
    hidden_to_ff_factor = trial.suggest_categorical("hidden_to_ff_factor", [1.5, 2.0, 3.0, 4.0])
    intermediate_size = int(model_cfg["original_model_params"]["hidden_size"] * hidden_to_ff_factor)
    model_cfg["original_model_params"]["intermediate_size"] = intermediate_size

    # Define the hyperparameter search space for training the supervised task
    model_cfg["use_uncertainty_weighting"] = \
        bool(trial.suggest_categorical("use_uncertainty_weighting", [0, 1]))
    if not model_cfg["use_uncertainty_weighting"]:
        model_cfg["supervised_task_weight"] = \
            trial.suggest_float("supervised_task_weight", 0.0, 1.0, step=0.1)

    return model_cfg


if __name__ == "__main__":

    # Common command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("-sl", "--use_supervised_labels", action="store_false", help="Use true labels to evaluate models")
    parser.add_argument("-ph", "--prediction_horizon", default=30, type=int, choices=csts.PREDICTION_HORIZONS, help="Horizon of the infection prediction task for model evaluation")
    parser.add_argument("-ct", "--cutoff_days_train", nargs="+", type=int, default=None, choices=csts.CUTOFF_DAYS, help="Cutoff days for the training dataset, take them all if None")
    parser.add_argument("-cv", "--cutoff_days_valid", nargs="+", type=int, default=None, choices=csts.CUTOFF_DAYS, help="Cutoff days for the evaluation/testing dataset, take them all if None")

    # Arguments that are specific to the current script
    parser.add_argument("-pc", "--pretrain_config_path", default="configs/patient_token_embedder.yaml", help="Path to the pretraining (masked LM) configuration file")
    parser.add_argument("-pe", "--use_pretrained_embeddings_for_input_layer", action="store_true", help="Use pretrained sentence embeddings for the input embedding layer")
    parser.add_argument("-nt", "--n_trials", type=int, default=100, help="Number of Optuna trials to run")

    # Run optuna tuning script
    args = parser.parse_args()
    main(args)
