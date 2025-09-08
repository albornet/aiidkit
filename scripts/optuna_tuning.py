import gc
import yaml
import argparse
import torch
import optuna
import wandb
from pprint import pprint
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
    or run a final training session with the best parameters from a completed study.
    """
    # Print the configuration and arguments
    print(f"Running with the following fixed hyper-parameters:")
    print(f"  - Use supervised labels: {args.use_supervised_labels} (TODO: CHECK THIS)")
    print(f"  - Prediction horizon: {args.prediction_horizon}")
    print(f"  - Cutoff days (training): {args.cutoff_days_train}")
    print(f"  - Cutoff days (evaluation): {args.cutoff_days_valid}")
    print(f"  - Use pretrained embeddings for input layer: {args.use_pretrained_embeddings_for_input_layer}")

    # Load the base configuration from the YAML file
    with open(args.pretrain_config_path, "r") as f:
        config = yaml.safe_load(f)

    # Set config elements that change with command line arguments
    config["MODEL_ARGUMENTS"]["use_pretrained_embeddings_for_input_layer"] = \
        args.use_pretrained_embeddings_for_input_layer

    # Decide which mode to run based on command-line arguments
    if args.run_best_trial:
        print("\n>>> Running in BEST PARAMS mode: Training with best hyper-parameters from study <<<")
        run_training_with_fixed_params(args, config)
    elif args.run_default:
        print("\n>>> Running in DEFAULT PARAMS mode: Training with default parameters from config <<<")
        run_training_with_fixed_params(args, config)
    else:
        print("\n>>> Running in OPTIMIZATION mode: Searching for best hyper-parameters <<<")
        run_hyperparameter_optimization(args, config)


def run_training_with_fixed_params(args, config):
    """
    Runs a single, full training job.

    If `args.run_best_trial` is set, it first loads an Optuna study and applies
    the best found hyper-parameters to the config.

    If `args.run_default` is set, it uses the parameters loaded from the
    base config file.
    """
    run_id = get_run_id()

    if args.run_best_trial:
        db_path = Path(config["TRAINING_ARGUMENTS"]["output_dir"]) / run_id / 'optuna_results.db'
        storage = f"sqlite:///{db_path}"

        # Check if the database file exists
        if not db_path.is_file():
            raise FileNotFoundError(
                f"Optuna database not found at '{db_path}'. "
                f"Ensure you have run the optimization first with the exact same arguments."
            )

        print(f"Loading study from: {storage}")
        study = optuna.load_study(
            study_name="pretrain_token_model_optuna",
            storage=storage,
        )

        best_trial = study.best_trial
        print(f"Found best trial (#{best_trial.number}) with value: {best_trial.value:.4f}")
        print("Applying best hyper-parameters:")
        pprint(best_trial.params)

        # Apply the best hyper-parameters to the config
        config["TRAINING_ARGUMENTS"] = apply_best_train_params(best_trial.params, config["TRAINING_ARGUMENTS"])
        config["MODEL_ARGUMENTS"] = apply_best_model_params(best_trial.params, config["MODEL_ARGUMENTS"])

        final_run_id = f"best_{run_id}"

    elif args.run_default:
        # No parameter changes needed for this mode
        print("Using default hyper-parameters from the configuration file.")
        final_run_id = f"default_{run_id}"

    # --- Common setup for a single, full training run ---

    # Modify the output_dir for this specific run
    new_output_dir = Path(config["TRAINING_ARGUMENTS"]["output_dir"]) / run_id / final_run_id
    config["TRAINING_ARGUMENTS"]["output_dir"] = str(new_output_dir)
    config["TRAINING_ARGUMENTS"]["run_name"] = final_run_id

    # Enable full features for the final run (model saving, reporting, etc.)
    config["EVALUATION_ARGUMENTS"]["do_clustering_analysis"] = False
    config["TRAINING_ARGUMENTS"]["save_strategy"] = "steps"
    config["TRAINING_ARGUMENTS"]["load_best_model_at_end"] = True
    config["TRAINING_ARGUMENTS"]["report_to"] = "wandb"

    print("\nStarting training run with the following configuration:")
    pprint(config)

    # Run the final training
    run_training(config, args)
    print(f"Training run '{final_run_id}' completed.")


def run_hyperparameter_optimization(args, config):
    """
    Performs hyper-parameter optimization using Optuna.
    """
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
        load_if_exists=True, # Allows resuming a study
    )

    # Lambda used to pass the static args and config to the objective function
    objective = lambda trial: objective_fn(trial, args, config)
    study.optimize(
        func=objective,
        n_trials=args.n_trials,
    )
    print("Hyper-parameter optimization completed.")


def objective_fn(
    trial: optuna.Trial,
    args: argparse.Namespace,
    base_config: dict
) -> float:
    """
    Optuna objective function for the graining script, will modify configuration,
    train the model, and return the best metric, based on the validation dataset
    """
    # Create a deep copy of the config for this trial to avoid side effects
    import copy
    config = copy.deepcopy(base_config)

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

    # Disable saving / logging for hyper-parameter tuning (to gain space and speed)
    config["TRAINING_ARGUMENTS"]["save_strategy"] = "no"
    config["TRAINING_ARGUMENTS"]["load_best_model_at_end"] = False
    config["TRAINING_ARGUMENTS"]["report_to"] = "none"

    # Print trial arguments
    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"  - Trial {trial.number} arguments:")
    for k, v in trial.params.items():
        print(f"    - {k}: {v}")

    # Run training and collect best metric
    best_run_metric = 0.0 # Default value
    try:
        best_run_metric = run_training(config, args)

    # Return low metric value in case of invalid hyper-parameters or GPU memory issues
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}. Returning 0.0 as metric.")
        # Optuna prunes trials that raise exceptions. Returning a value is better.
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


def apply_best_train_params(
    best_params: dict,
    train_cfg: dict[str, str],
) -> dict[str, str]:
    """
    Applies the best training hyper-parameters from an Optuna trial to the config.
    """
    train_cfg["learning_rate"] = best_params["learning_rate"]
    train_cfg["weight_decay"] = best_params["weight_decay"]
    train_cfg["warmup_steps"] = best_params["warmup_steps"]
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
        # We don't need `set_user_attr` here as it's not a suggested param
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


def apply_best_model_params(
    best_params: dict,
    model_cfg: dict[str, int|float|str],
) -> dict[str, int|float|str]:
    """
    Applies the best model hyper-parameters from an Optuna trial to the config.
    """
    # Set hidden size based on pretrained embeddings or the best trial's value
    if model_cfg["use_pretrained_embeddings_for_input_layer"]:
        model_cfg["original_model_params"]["hidden_size"] = 768
    else:
        model_cfg["original_model_params"]["hidden_size"] = best_params["hidden_size"]

    # Apply other model parameters
    model_cfg["use_positional_encoding_for_input_layer"] = \
        bool(best_params["use_positional_encoding_for_input_layer"])
    model_cfg["original_model_params"]["num_attention_heads"] = best_params["num_attention_heads"]
    model_cfg["original_model_params"]["num_hidden_layers"] = best_params["num_hidden_layers"]

    # Re-calculate intermediate size based on the best trial's parameters
    hidden_size = model_cfg["original_model_params"]["hidden_size"]
    hidden_to_ff_factor = best_params["hidden_to_ff_factor"]
    model_cfg["original_model_params"]["intermediate_size"] = int(hidden_size * hidden_to_ff_factor)

    # Apply supervised task parameters
    model_cfg["use_uncertainty_weighting"] = bool(best_params["use_uncertainty_weighting"])

    # Check if supervised_task_weight was a parameter in the best trial
    if "supervised_task_weight" in best_params:
        model_cfg["supervised_task_weight"] = best_params["supervised_task_weight"]

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
    parser.add_argument("-nt", "--n_trials", type=int, default=50, help="Number of Optuna trials to run")

    # Mutually exclusive group for run modes
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("-rb", "--run_best_trial", action="store_true", help="Run a single training with the best parameters from the Optuna study.")
    mode_group.add_argument("-rd", "--run_default", action="store_true", help="Run a single training with default parameters from the config file.")

    # Run optuna tuning script
    args = parser.parse_args()
    main(args)