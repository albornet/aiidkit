# Patient sequence modeling project

This project processes raw electronic health records (EHR) of kidney transplant recipients to train transformer-based models. The objective is to learn meaningful embeddings from patient sequences that can be used for downstream predictive tasks, such as forecasting post-transplant infections, and for clinical interpretability.

Note: This pipeline requires access to the AIIDKIT dataset. You must place the raw data file in the data/raw directory before running the code.

## 📋 Table of contents

  - [Project workflow](#-project-workflow)
  - [Setup](#️-setup)
  - [Usage](#-usage)
      - [Step 1: Create datasets](#step-1-create-datasets)
      - [Step 2: Pre-train the token embedding model](#step-2-pre-train-the-token-embedding-model)
      - [Step 3: Run hyperparameter optimization](#step-3-run-hyperparameter-optimization)
  - [Scripts overview](#-scripts-overview)
  - [Configuration](#-configuration)

## 🚀 Project workflow

The project follows a clear, step-by-step workflow:

1.  **Data Preprocessing**: Raw patient data from a pickle file is processed into individual patient record CSVs. These are then aggregated into a structured HugginFace datasets object.
2.  **Model Pre-training**: A custom `PatientTokenEmbeddingModel` (a transformer-based architecture) is pre-trained on the patient sequences using a Masked Language Modeling (MLM) objective. This step learns to understand the structure and patterns in the patient data.
3.  **Hyperparameter Tuning**: Optuna is used to systematically search for the best set of hyperparameters for the model architecture and training process, maximizing performance on a validation set. The `optuna_tuning.sh` script automates running experiments across different data slices and prediction horizons.

## ⚙️ Setup

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/albornet/aiidkit
    cd aiidkit
    ```

2.  **Install dependencies:**
    For example, you can use `uv` for efficient package and virtual environment management.

    ```bash
    # Download and install uv
    wget -qO- https://astral.sh/uv/install.sh | sh
    # Create and activate a local virtual environment
    uv venv && source .venv/bin/activate
    # Install dependencies
    uv pip install -r requirements.txt
    ```

3.  **Place the data:**
    Ensure your raw patient data file is located at the path specified in `src/constants.py` (e.g., `data/raw/stcs_data.pkl`).

## 💻 Usage

Follow these steps in order to replicate the full pipeline.

### Step 1: Create datasets

This script processes the raw data into a format suitable for model training.

```bash
python -m scripts.create_patient_datasets
```

The script will:

1.  Read the raw data pickle file.
2.  Generate individual patient CSVs in the `data/preprocessed` directory.
3.  Assemble these CSVs into a Hugging Face `DatasetDict` and save it to `data/huggingface`.
4.  Create specialized datasets for the infection prediction task.

**Optional flags:**

  * `--debug` or `-d`: Runs the script on a small subset of patients (1000) for quick testing.
  * `--explore` or `-e`: Enters an interactive debugging session to explore the raw dataframes.
  * `--postprocess` or `-p`: Skips the creation of individual CSVs and only runs the final Hugging Face dataset creation and task-specific processing.

### Step 2: Pre-train the token embedding model

This script runs a single training job for the `PatientTokenEmbeddingModel` based on a given configuration file. While this can be run standalone, it is typically called by the `optuna_tuning.py` script.

```bash
python -m scripts.pretrain_token_model \
    --pretrain_config_path configs/patient_token_embedder.yaml \
    --prediction_horizon 30
```

This script is highly configurable via command-line arguments to specify the prediction task, data cutoffs, and more.

### Step 3: Run hyperparameter optimization

This is the main entry point for running experiments. The `optuna_tuning.sh` script orchestrates multiple runs of the `optuna_tuning.py` script to test various experimental conditions.

**Before running, configure your experiment in `scripts/optuna_tuning.sh`**:

  * `PREDICTION_HORIZONS`: Set the prediction windows to test (e.g., 30, 60, 90 days).
  * `CUTOFF_PAIRS`: Define which patient sequence lengths to use for training and validation.
  * `PRETRAINED_EMBEDDINGS_OPTIONS`: Toggle the use of pre-trained sentence embeddings for the model's input layer.

The script can be run in three modes:

1.  **Optimization Mode (Default)**: Runs an Optuna hyperparameter search to find the best model and training parameters.

    ```bash
    ./scripts/optuna_tuning.sh
    ```

2.  **Final Run Mode**: After an optimization study is complete, this mode loads the best parameters from the Optuna database and runs a final, full training job, saving the best model.

    ```bash
    ./scripts/optuna_tuning.sh --run-best
    ```

3.  **Default Run Mode**: Runs a single training job using the default parameters specified in the YAML configuration file, without performing a search.

    ```bash
    ./scripts/optuna_tuning.sh --run-default
    ```

## 📜 Scripts overview

  * `scripts/create_patient_datasets.py`: The entry point for all data preprocessing. Converts raw STCS data into clean, sequential patient records ready for model ingestion.
  * `scripts/pretrain_token_model.py`: Core logic for training the patient sequence model. It handles loading data, initializing the model, and running the Hugging Face `Trainer`.
  * `scripts/optuna_tuning.py`: A wrapper around `pretrain_token_model.py` that uses Optuna to perform hyperparameter searches. It defines the search space for parameters like learning rate, layer count, attention heads, etc.
  * `scripts/optuna_tuning.sh`: A shell script to automate running experiments across different configurations (e.g., multiple prediction horizons, data cutoffs). This is the primary script for launching experiments.

## 🔧 Configuration

The model architecture, training arguments, and evaluation settings are primarily controlled by a YAML configuration file.

  * `configs/patient_token_embedder.yaml`: This file contains default settings for:
      * **`MODEL_ARGUMENTS`**: Defines the transformer architecture (e.g., `hidden_size`, `num_hidden_layers`, `num_attention_heads`).
      * **`TRAINING_ARGUMENTS`**: Sets parameters for the Hugging Face `Trainer` (e.g., `output_dir`, `learning_rate`, `per_device_train_batch_size`, `report_to`).
      * **`EVALUATION_ARGUMENTS`**: Configures the evaluation loop (e.g., `metric_for_best_model`, `early_stopping_patience`).

The values in this file serve as the baseline for hyperparameter optimization runs.