from dataclasses import dataclass
from typing import Sequence

@dataclass
class TrainConfig:
    experiment_name: str = "ClinicalTransformerExperiment"
    run_name: str = "run1"
    resume_run_id: str = "e6d46beda3934e81aa8afcac8db83233"  # Set to a run_id to resume training, or None
    max_epochs: int = 200
    patience: int = 200
    learning_rate: float = 1e-4
    mlm_probability: float = 0.25
    mlm_target: str = "both-linked"
    mlm_loss_weights: Sequence[float] = (1.0, 1.0, 1.0)  # weights for (loss_n, loss_v, loss_v_num)
    train_ratio: float = 0.8
    batch_size: int = 128
    num_workers: int = 1
    check_val_every_n_epoch: int = 1
    gradient_clip_val: float = 1.0

    # Scheduler parameters for CosineAnnealingWarmRestarts
    scheduler: str = "cosine_annealing_warm_restarts"  
    t_0: int = 10       # Number of epochs for the first restart
    t_mult: int = 1     # Factor by which the number of epochs between restarts is multiplied
    eta_min: float = 1e-6  # Minimum learning rate

