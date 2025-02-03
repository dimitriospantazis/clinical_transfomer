# src/lightning/clinical_transformer_module.py

import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from src.model.clinical_transformer import ClinicalTransformer, ClinicalTransformerConfig
from dataclasses import asdict


class ClinicalTransformerModule(pl.LightningModule):
    def __init__(
        self,
        model_config: ClinicalTransformerConfig,
        train_config,  # instance of TrainConfig
    ):
        """
        LightningModule for the ClinicalTransformer.

        Args:
            model_config (ClinicalTransformerConfig): Configuration for the ClinicalTransformer.
            training_config: Configuration for training parameters (learning_rate and mlm_loss_weights).
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model_config", "train_config"])
        self.hparams.update(asdict(train_config))
        self.model = ClinicalTransformer(model_config)
        self.learning_rate = train_config.learning_rate
        # mlm_loss_weights should be a sequence of 3 values for (loss_n, loss_v, loss_v_num)
        self.mlm_loss_weights = train_config.mlm_loss_weights

    def forward(self, batch):
        """Forward pass through the model."""
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch)
        mlm_loss = outputs.get("mlm_loss")

        # Return the MLM loss for now (no survival loss yet)
        if mlm_loss is not None:
            loss_n, loss_v, loss_v_num = mlm_loss
            # Build a list of weighted losses, but only include values that are not NaN.
            weighted_losses = [
                w * l for w, l in zip(self.mlm_loss_weights, [loss_n, loss_v, loss_v_num])
                if not torch.isnan(l).item()
            ]
            if weighted_losses:
                loss = sum(weighted_losses) / len(weighted_losses)
            else:
                loss = torch.tensor(0.0, device=self.device)
        else:
            loss = torch.tensor(0.0, device=self.device)
        
        # Log individual and overall losses
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss_n", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_loss_v", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_loss_v_num", loss, on_step=False, on_epoch=True, prog_bar=False)

        # Log the learning rate
        opt = self.optimizers()
        lr = opt.param_groups[0]['lr']
        self.log("lr", lr, on_step=False, on_epoch=True, prog_bar=True, logger=True) 

        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch)
        mlm_loss = outputs.get("mlm_loss")
        if mlm_loss is not None:
            loss_n, loss_v, loss_v_num = mlm_loss
            # Build a list of weighted losses for those components that are not NaN.
            weighted_losses = [
                w * l for w, l in zip(self.mlm_loss_weights, [loss_n, loss_v, loss_v_num])
                if not torch.isnan(l).item()
            ]
            if weighted_losses:
                loss = sum(weighted_losses) / len(weighted_losses)
            else:
                # loss = torch.tensor(float('nan'), device=self.device)
                loss = torch.tensor(float(1000000), device=self.device)
        else:
            # loss = torch.tensor(float('nan'), device=self.device)
            loss = torch.tensor(float(1000000), device=self.device)

        # Log individual and overall losses
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss_n", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_loss_v", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_loss_v_num", loss, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        if self.hparams.scheduler == "cosine_annealing_warm_restarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.hparams.t_0,
                T_mult=self.hparams.t_mult,
                eta_min=self.hparams.eta_min,
            )
            return [optimizer], [scheduler]
        else:
            return optimizer
    
    def on_train_epoch_end(self):
        # Retrieve metrics from the callback_metrics dictionary
        train_loss = self.trainer.callback_metrics.get("train_loss")
        val_loss = self.trainer.callback_metrics.get("val_loss")

        # Format the losses safely.
        train_loss_str = f"{train_loss:.4f}" if train_loss is not None else "N/A"
        val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "N/A"
        
        # Print them in the console.
        print(
            f"\nEpoch {self.current_epoch} (train) "
            f"| train_loss: {train_loss_str} "
            f"| val_loss: {val_loss_str}\n"
        )
