import torch; import torchvision as tv

dataset = tv.datasets.CIFAR10("data", download=True, train=True, transform=tv.transforms.ToTensor())

model = tv.models.resnet18()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=8)

model.train()
num_epochs = 10
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        print(loss.data)