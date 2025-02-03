from dataclasses import asdict
import os
import glob
import mlflow
from typing import Optional

def config_to_dict(config_obj):
    """
    Converts a configuration object to a dictionary.
    First attempts to use asdict (for dataclasses); if that fails, uses vars().
    """
    try:
        return asdict(config_obj)
    except Exception:
        return vars(config_obj)

def log_hyperparameters(train_config, model_config, mlflow_logger):
    """
    Logs hyperparameters from both train_config and model_config into MLflow.
    
    Args:
        train_config: Your training configuration (e.g., an instance of TrainConfig).
        model_config: Your model configuration (e.g., an instance of ClinicalTransformerConfig).
        mlflow_logger: An instance of MLFlowLogger (or similar) that exposes a log_hyperparams() method.
    """
    train_params = config_to_dict(train_config)
    model_params = config_to_dict(model_config)
    
    # Combine the dictionaries (if there are overlapping keys, model_config values will override train_config values).
    all_params = {**train_params, **model_params}
    
    mlflow_logger.log_hyperparams(all_params)

def log_best_model(best_model, run_id, checkpoint_path):
    """
    Logs the best model and its checkpoint to MLflow.

    Args:
        best_model (pl.LightningModule): The best PyTorch Lightning model.
        run_id (str): The MLflow run ID.
        checkpoint_path (str): Path to the best checkpoint file.
    """
    mlflow.pytorch.log_model(
        pytorch_model=best_model,
        artifact_path="best_model",
        run_id=run_id
    )
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        mlflow.log_artifact(
            local_path=checkpoint_path,
            artifact_path="best_model_ckpt",
            run_id=run_id
        )
    else:
        print("No best checkpoint found to log.")

def download_best_checkpoint(run_id: str, artifact_path: str) -> Optional[str]:
    """
    Downloads the best checkpoint artifact for the given MLflow run.
    
    Args:
        run_id (str): The MLflow run ID from which to download the checkpoint.
        artifact_path (str): The artifact path where the checkpoint is stored (e.g., "best_model_ckpt").
        
    Returns:
        Optional[str]: The local file path to the checkpoint if found; otherwise, None.
    """
    try:
        # Download all artifacts under the given artifact_path for the specified run.
        local_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path)
        # Search for checkpoint files (*.ckpt) in the downloaded directory.
        ckpt_files = glob.glob(os.path.join(local_dir, "*.ckpt"))
        if ckpt_files:
            # Return the first checkpoint found (or adjust this logic if multiple checkpoints exist)
            return ckpt_files[0]
        else:
            print(f"No checkpoint files found in {local_dir}.")
            return None
    except Exception as e:
        print("Error downloading checkpoint:", e)
        return None
