import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28*28, 64), nn.ReLU(), nn.Linear(64, 3))
    def forward(self, x):
        return self.l1(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28*28))
    def forward(self, x):
        return self.l1(x)

# Define a LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx): # Needed
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x, x_hat)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self): # Needed
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss, prog_bar=True)
        return test_loss

# Define the training dataset
train_set = MNIST(root="MNIST", download=True, train=True, transform=transforms.ToTensor())
test_set = MNIST(root="MNIST", download=True, train=False, transform=transforms.ToTensor())

train_set_size = int(len(train_set)*0.8)
val_set_size = int(len(train_set)) - train_set_size

seed = torch.Generator().manual_seed(42)
train_set, val_set = data.random_split(train_set, [train_set_size, val_set_size], generator=seed)

train_set = DataLoader(train_set)
val_set = DataLoader(val_set)
test_set = DataLoader(test_set)


# Train the model
autoencoder = LitAutoEncoder(Encoder(), Decoder())

trainer = pl.Trainer(max_epochs=1)
trainer.fit(model=autoencoder, train_dataloaders=train_set, val_dataloaders=val_set)
trainer.test(model=autoencoder, test_dataloaders=test_set)


