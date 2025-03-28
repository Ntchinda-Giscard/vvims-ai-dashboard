import lightning as L
from tqdm import tqdm
import torch; import torchvision as tv


# Fabric will automatically use all available GPUs!
fabric = L.Fabric()
fabric.launch()

with fabric.rank_zero_first(local=True):
    dataset = tv.datasets.CIFAR10("data", download=True,
                                  train=True,
                                  transform=tv.transforms.ToTensor())

model = tv.models.resnet18()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
model, optimizer = fabric.setup(model, optimizer)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)
dataloader = fabric.setup_dataloaders(dataloader)

model.train()
num_epochs = 10
for epoch in range(num_epochs):
    for batch in tqdm(dataloader):
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        fabric.backward(loss)
        optimizer.step()