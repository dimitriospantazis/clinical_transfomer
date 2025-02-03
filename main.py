import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
import mlflow
import os
import json
import random

from src.model.clinical_transformer import ClinicalTransformerConfig
from src.lightning.clinical_transformer_module import ClinicalTransformerModule
from src.lightning.clinical_transformer_data_module import ClinicalDataset
from config.train_config import TrainConfig
from src.utils.feature_schema import load_feature_schema
from src.model.tokenizer import Tokenizer, JointTokenizer
from src.model.collator import Collator
from torch.utils.data import DataLoader
from src.utils.train_utils import log_hyperparameters, log_best_model, download_best_checkpoint
import mlflow.pytorch


# 1. Train configuration.
train_config = TrainConfig(
    experiment_name="clinical_transformer",  # MLFlow experiment name
    run_name="default_run6",                  # MLFlow run name
    resume_run_id="0a56b03a46fd4b30b661338e5ff4ea10",  # Set to a run_id to resume training, or None
    max_epochs=200,                           # Maximum number of epochs
    patience=10,                              # Patience for early stopping
    learning_rate=1e-3,                      # Learning rate
    mlm_probability=0.25,                    # Probability of masking tokens
    mlm_target="both-linked",                # Masking strategy
    mlm_loss_weights=(1.0, 1.0, 1.0),        # Loss weights for (loss_n, loss_v, loss_v_num)
    train_ratio=0.8,                         # Train-validation split ratio
    batch_size=64,                          # Batch size
    num_workers=1,                           # Number of workers for DataLoader
    check_val_every_n_epoch=1,               # Check validation every n epochs
    gradient_clip_val=1.0,                   # Gradient clipping value

    # Scheduler parameters for CosineAnnealingWarmRestarts
    scheduler="cosine_annealing_warm_restarts",
    t_0=10,                                  # Number of epochs for the first restart
    t_mult=1,                                # Factor by which the number of epochs between restarts is multiplied
    eta_min=1e-6                             # Minimum learning rate
)


# 2. Model configuration.
model_config = ClinicalTransformerConfig(
    vocab_size_n=3522,            # len(tokenizer_n), Size of the vocabulary for feature names
    vocab_size_v=146,             # len(tokenizer_v), Size of the vocabulary for feature values
    embed_dim_n=32,               # Embedding size for name tokens
    embed_dim_v=32,               # Embedding size for value tokens
    hidden_dim_num=16,            # Hidden dimension for numerical feature processing
    hidden_size=128,              # Transformer hidden dimension
    num_layers=6,                 # Number of transformer encoder layers
    num_heads=4,                  # Number of attention heads
    dropout=0.1                   # Dropout rate
)


# 3. Load feature schema and create tokenizers.
_, vocab_n, vocab_v = load_feature_schema(os.path.join('data', 'genie', 'feature_schema.json'))
tokenizer_n = Tokenizer(vocab_n)
tokenizer_v = Tokenizer(vocab_v)
joint_tokenizer = JointTokenizer(tokenizer_n, tokenizer_v)


# 4. Load and preprocess data.
file = os.path.join('data', 'genie', 'patient_sequences.json')
with open(file, 'r') as f:
    patient_sequences = json.load(f)
patient_sequences = list(patient_sequences.values())
# Remove patients with up to 3 features (only have 'SEX', 'AGE_AT_SEQ', 'CANCER_TYPE')
patient_sequences = [patient for patient in patient_sequences if len(patient) > 3]

tokenized_data = joint_tokenizer(patient_sequences)
random.seed(42)
shuffled_data = tokenized_data.copy()
random.shuffle(shuffled_data)
split = int(len(shuffled_data) * train_config.train_ratio)
train_data = shuffled_data[:split]
val_data = shuffled_data[split:]


# 5. Create datasets and collators.
train_dataset = ClinicalDataset(train_data)
val_dataset = ClinicalDataset(val_data)

collator_train = Collator(
    joint_tokenizer,
    mlm=True,
    mlm_probability=train_config.mlm_probability,
    mlm_target=train_config.mlm_target,
    shuffle_tokens=False #------------------------------------------------------
)
collator_val = Collator(
    joint_tokenizer,
    mlm=True,
    mlm_probability=train_config.mlm_probability,
    mlm_target=train_config.mlm_target,
    shuffle_tokens=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=train_config.batch_size,
    collate_fn=collator_train,
    num_workers=train_config.num_workers,
    persistent_workers=True,
    pin_memory=True,
    shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=train_config.batch_size,
    collate_fn=collator_val,
    num_workers=train_config.num_workers,
    persistent_workers=True,
    pin_memory=True,
    shuffle=False
)


# 6. Set up MLFlow logger.
mlflow_logger = MLFlowLogger(
    experiment_name=train_config.experiment_name, 
    run_name=train_config.run_name,
    tracking_uri="mlruns"  # Adjust the tracking URI as needed
)
print(f"MLflow Run ID: {mlflow_logger.run_id}")


# 7. Create Lightning module.
model_module = ClinicalTransformerModule(
    model_config=model_config, 
    train_config=train_config
)


# 8. Set up callbacks.
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
checkpoint_callback = ModelCheckpoint(
    dirpath = os.path.join("model_checkpoints"),
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    filename="best_model_{epoch:02d}-{val_loss:.2f}"
)
early_stop_callback = EarlyStopping(monitor="val_loss", patience=train_config.patience, mode="min")
lr_monitor = LearningRateMonitor(logging_interval="epoch")
callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]


# 9. Create Trainer.
trainer = pl.Trainer(
    max_epochs=train_config.max_epochs,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    logger=mlflow_logger,
    gradient_clip_val=train_config.gradient_clip_val,
    check_val_every_n_epoch=train_config.check_val_every_n_epoch,
    callbacks=callbacks,
)


# 10. Log hyperparameters.
log_hyperparameters(train_config, model_config, mlflow_logger)


# 11. Optionally, check if we need to resume training.
ckpt_path = None
if train_config.resume_run_id is not None:
    print(f"Resuming training from MLFlow run id: {train_config.resume_run_id}")
    ckpt_path = download_best_checkpoint(train_config.resume_run_id, "best_model_ckpt")
    if ckpt_path:
        print(f"Found checkpoint: {ckpt_path}")
        model_module = ClinicalTransformerModule.load_from_checkpoint(ckpt_path, model_config=model_config, train_config=train_config)
    else:
        print("No checkpoint found; training from scratch.")


if __name__ == "__main__":

    torch.set_float32_matmul_precision('high')  # Options: 'medium', 'high', 'highest'

    # 12. Train the model.
    trainer.fit(model_module, train_loader, val_loader, ckpt_path=ckpt_path)

    # 13. Retrieve the best checkpoint path
    best_ckpt_path = None
    for callback in callbacks:
        if isinstance(callback, pl.callbacks.ModelCheckpoint):
            best_ckpt_path = callback.best_model_path
            break
    print(f"\nBest checkpoint saved at: {best_ckpt_path}")
    print(f"MLflow Run ID: {mlflow_logger.run_id}")


    if best_ckpt_path:
        # Load the best model from the checkpoint
        best_model = ClinicalTransformerModule.load_from_checkpoint(best_ckpt_path, model_config=model_config, train_config=train_config)

        # Log the best model and checkpoint to MLflow
        log_best_model(best_model, mlflow_logger.run_id, best_ckpt_path)
    else:
        print("No best checkpoint found to log.")

