import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from models import FashionDataModule, CrossModalRetrievalModel

def main():
    pl.seed_everything(42)

    # Hardware detection
    accelerator = "auto"
    if torch.backends.mps.is_available():
        accelerator = "mps"
    elif torch.cuda.is_available():
        accelerator = "gpu"

    # In dev mode, our dataset size is very small (1000 total, 800 train), limit batch size.
    dev_mode = os.environ.get("DEV_MODE", "0") == "1"
    batch_size = 32 if dev_mode else 256

    data_module = FashionDataModule(data_dir="data", batch_size=batch_size)
    
    model = CrossModalRetrievalModel(
        d_image=384,
        d_text=768,
        d_joint=512,
        lr=1e-3,
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

    trainer = pl.Trainer(
        max_epochs=20 if not dev_mode else 5,
        accelerator=accelerator,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10 if not dev_mode else 1,
    )

    print("Starting training...")
    trainer.fit(model, datamodule=data_module)
    print("Training complete.")

if __name__ == "__main__":
    main()
