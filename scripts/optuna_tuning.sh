#!/bin/bash

# ==============================================================================
# This script runs a Python script for specific combinations of arguments.
# It is configured to run predefined pairs of train/valid cutoff days.
# ==============================================================================

# --- Configuration ---

PYTHON_SCRIPT="optuna_tuning.py"
PRETRAINED_EMBEDDINGS_OPTIONS=(
    "--use_pretrained_embeddings_for_input_layer" 
    ""
)

PREDICTION_HORIZONS=(
    "7" 
    "30" 
    # "90" 
    # "180" 
    # "365"
)


# Define the specific pairs of cutoff days for train and valid sets.
# FORMAT: "<train_cutoff_days>;<valid_cutoff_days>"
# - Use a space to separate multiple values (e.g., "0 7 30").
# - To use the script's None, leave the part for train or valid empty.
#   Example: ";30" runs with None train cutoff and valid cutoff of 30.
#   None means full sequences are taken
CUTOFF_PAIRS=(
    "0;0"
    ";0"
    "30;30"
    ";30"
    "365;365"
    ";365"
    "1000;1000"
    ";1000"
)

# Add any other arguments that should be constant across all runs.
OTHER_ARGS=()


# --- Execution ---

# Check if the python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found."
    echo "Please update the PYTHON_SCRIPT variable in this script."
    exit 1
fi

echo "Starting grid search over specified hyperparameter combinations..."

# Loop through the primary combinations
for pe_option in "${PRETRAINED_EMBEDDINGS_OPTIONS[@]}"; do
  for ph_option in "${PREDICTION_HORIZONS[@]}"; do
    # Loop through the specific pairs of cutoff days
    for pair in "${CUTOFF_PAIRS[@]}"; do
    
      # Split the pair into train and valid options based on the semicolon
      ct_option="${pair%;*}"
      cv_option="${pair#*;}"

      # Start building the command in an array for robustness
      CMD=(python "$PYTHON_SCRIPT")

      # Add the --use_pretrained_embeddings_for_input_layer flag if the option is not empty
      if [[ -n "$pe_option" ]]; then
        CMD+=("$pe_option")
      fi

      # Add --prediction_horizon
      CMD+=("--prediction_horizon" "$ph_option")

      # Add --cutoff_days_train if the option is not empty
      if [[ -n "$ct_option" ]]; then
        CMD+=("--cutoff_days_train" $ct_option)
      fi

      # Add --cutoff_days_valid if the option is not empty
      if [[ -n "$cv_option" ]]; then
        CMD+=("--cutoff_days_valid" $cv_option)
      fi
      
      # Add any other constant arguments
      if [ ${#OTHER_ARGS[@]} -gt 0 ]; then
          CMD+=("${OTHER_ARGS[@]}")
      fi

      # Print the command that is about to be executed
      echo "################################################################################"
      echo "#"
      echo "# RUNNING NEW COMBINATION"
      echo "# Command: ${CMD[@]}"
      echo "#"
      echo "################################################################################"

      # Execute the command
      "${CMD[@]}"

      # Check the exit code of the last command
      if [ $? -ne 0 ]; then
          echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
          echo "WARNING: The last command failed with a non-zero exit code."
          echo "Command was: ${CMD[@]}"
          echo "Continuing with the next combination..."
          echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
      fi

    done
  done
done

echo "================================================================================"
echo "All combinations have been executed."
echo "================================================================================"