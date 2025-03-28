import os, torch, torch.nn as nn, torch.utils.data as data, torchvision as tv
import lightning as L

encoder = nn.Sequential(nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 3))
decoder = nn.Sequential(nn.Linear(3, 128), nn.ReLU(), nn.Linear(128, 28 * 28))

class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

dataset = tv.datasets.MNIST(".", download=True, transform=tv.transforms.ToTensor())

# Lightning will automatically use all available GPUs!
trainer = L.Trainer()
trainer.fit(LitAutoEncoder(encoder, decoder), data.DataLoader(dataset, batch_size=64))