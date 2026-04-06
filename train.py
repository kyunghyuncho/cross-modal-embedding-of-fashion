import os
import json
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from models import FashionDataModule, CrossModalRetrievalModel
import argparse

class JSONLoggerCallback(Callback):
    def __init__(self, log_file="lightning_logs/live_metrics.json"):
        self.log_file = log_file
        self.metrics = {"train_loss": [], "val_recall@5": []}
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self._write()

    def _write(self):
        with open(self.log_file, "w") as f:
            json.dump(self.metrics, f)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None:
            self.metrics["train_loss"].append({"step": trainer.global_step, "value": loss.item()})
            self._write()

    def on_validation_epoch_end(self, trainer, pl_module):
        recall = trainer.callback_metrics.get("val_recall@5")
        if recall is not None:
            self.metrics["val_recall@5"].append({"step": trainer.global_step, "value": recall.item()})
            self._write()

class StopTrainingCallback(Callback):
    def __init__(self, flag_file="lightning_logs/stop.flag"):
        self.flag_file = flag_file
        if os.path.exists(self.flag_file):
            os.remove(self.flag_file)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if os.path.exists(self.flag_file):
            print("Stop flag detected. Halting training.")
            trainer.should_stop = True
            os.remove(self.flag_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    pl.seed_everything(42)

    accelerator = "auto"
    if torch.backends.mps.is_available():
        accelerator = "mps"
    elif torch.cuda.is_available():
        accelerator = "gpu"

    dev_mode = os.environ.get("DEV_MODE", "0") == "1"
    batch_size = 32 if dev_mode else args.batch_size

    data_module = FashionDataModule(data_dir="data", batch_size=batch_size)
    
    model = CrossModalRetrievalModel(
        d_image=384,
        d_text=768,
        d_joint=512,
        lr=args.lr,
        weight_decay=1e-4
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="lightning_logs/checkpoints",
        filename="fashion-{epoch:02d}-{val_recall@5:.3f}",
        monitor="val_recall@5",
        mode="max",
        save_top_k=1,
    )

    early_stop_callback = EarlyStopping(
        monitor="val_recall@5",
        min_delta=0.00,
        patience=5,
        verbose=True,
        mode="max"
    )

    os.makedirs("lightning_logs", exist_ok=True)
    json_logger = JSONLoggerCallback("lightning_logs/live_metrics.json")
    stop_check = StopTrainingCallback("lightning_logs/stop.flag")

    with open("lightning_logs/running.flag", "w") as f:
        f.write("running")

    trainer = pl.Trainer(
        max_epochs=args.epochs if not dev_mode else 5,
        accelerator=accelerator,
        callbacks=[checkpoint_callback, early_stop_callback, json_logger, stop_check],
        log_every_n_steps=10 if not dev_mode else 1,
    )

    try:
        print("Starting training...")
        trainer.fit(model, datamodule=data_module)
        print("Training complete.")
    finally:
        if os.path.exists("lightning_logs/running.flag"):
            os.remove("lightning_logs/running.flag")

if __name__ == "__main__":
    main()
