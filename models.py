import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np

class FashionDataset(Dataset):
    def __init__(self, pt_path):
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        self.image_embeddings = data["image_embeddings"]
        self.text_embeddings = data["text_embeddings"]
        self.texts = data["texts"]
        self.length = len(self.image_embeddings)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.image_embeddings[idx], self.text_embeddings[idx]

class FashionDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="data", batch_size=256):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        import os
        self.train_ds = FashionDataset(os.path.join(self.data_dir, "train_embeddings.pt"))
        if stage in ('fit', 'validate'):
            self.val_ds = FashionDataset(os.path.join(self.data_dir, "val_embeddings.pt"))
        if stage == 'test':
            self.test_ds = FashionDataset(os.path.join(self.data_dir, "test_embeddings.pt"))

    def train_dataloader(self):
        # drop_last=True is needed for our symmetric loss in case of small batches at the end
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=4, pin_memory=True)


class CrossModalRetrievalModel(pl.LightningModule):
    def __init__(self, d_image=384, d_text=768, d_joint=512, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.W_image = nn.Linear(d_image, d_joint)
        self.W_text = nn.Linear(d_text, d_joint)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self.validation_step_outputs = []

    def forward_image(self, x_image):
        return F.normalize(self.W_image(x_image), p=2, dim=1)
        
    def forward_text(self, x_text):
        return F.normalize(self.W_text(x_text), p=2, dim=1)

    def forward(self, x_image, x_text):
        z_image = self.forward_image(x_image)
        z_text = self.forward_text(x_text)
        return z_image, z_text

    def training_step(self, batch, batch_idx):
        x_image, x_text = batch
        z_image, z_text = self(x_image, x_text)
        
        logit_scale = self.logit_scale.exp()
        
        logits_per_image = logit_scale * z_image @ z_text.t()
        logits_per_text = logits_per_image.t()

        batch_size = x_image.shape[0]
        labels = torch.arange(batch_size, device=self.device)

        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_image, x_text = batch
        z_image, z_text = self(x_image, x_text)
        self.validation_step_outputs.append((z_image, z_text))

    def on_validation_epoch_end(self):
        z_images = torch.cat([x[0] for x in self.validation_step_outputs])
        z_texts = torch.cat([x[1] for x in self.validation_step_outputs])
        self.validation_step_outputs.clear()

        similarity = z_images @ z_texts.t()
        labels = torch.arange(similarity.shape[0], device=self.device)
        
        recall_at_1 = self._compute_recall(similarity, labels, k=1)
        recall_at_5 = self._compute_recall(similarity, labels, k=5)
        recall_at_10 = self._compute_recall(similarity, labels, k=10)

        self.log("val_recall@1", recall_at_1, prog_bar=True, sync_dist=True)
        self.log("val_recall@5", recall_at_5, prog_bar=True, sync_dist=True)
        self.log("val_recall@10", recall_at_10, prog_bar=True, sync_dist=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * similarity
        logits_per_text = logits_per_image.t()
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        val_loss = (loss_i + loss_t) / 2
        
        self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)

    def _compute_recall(self, similarity, labels, k):
        _, topk_indices = similarity.topk(k, dim=1, largest=True, sorted=True)
        correct = topk_indices.eq(labels.view(-1, 1).expand_as(topk_indices))
        return correct.sum().float() / labels.shape[0]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        return optimizer
