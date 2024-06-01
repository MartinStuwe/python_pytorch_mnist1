import torch
import pandas as pd
import pyarrow
from PIL import Image, ImageOps
import io
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
import glob
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torchvision.datasets as datasets

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.4)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=9216, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        output = F.log_softmax(x, dim=1) # log(\frac{e^x_1}{\sum_j(e^x_j)})
        return output

def train(model, device, train_loader, optimizer, criterion, scaler, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        with autocast():
            output = model(data)
            loss = criterion(output, target)

        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 10 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}, Accuracy: {100. * correct / total}')

    print(f'End of Epoch {epoch}, Loss: {running_loss / len(train_loader)}, Accuracy: {100. * correct / total}')


def evaluate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    val_loss /= len(val_loader)
    accuracy = 100. * correct / total
    print(f"Validation Loss: {val_loss}, Accuracy: {accuracy}\n")
    return val_loss, accuracy

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    print(f'Test Loss: {test_loss}, Accuracy: {accuracy}')
    return test_loss, accuracy


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.1307], [0.3081])
])

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)


train_size = int(0.8 * len(mnist_trainset))
val_size = len(mnist_trainset) - train_size
train_dataset, val_dataset = random_split(mnist_trainset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(mnist_testset, batch_size=64, shuffle=64)

loader = DataLoader(mnist_trainset, batch_size=len(mnist_trainset), shuffle=False)
data=next(iter(loader))[0]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
scaler = GradScaler()

num_epochs = 10
for epoch in range(1, num_epochs + 1):
    train(cnn, device, train_loader, optimizer, criterion, scaler, epoch)
    val_loss, val_accuracy = evaluate(cnn, device, val_loader, criterion)

test_loss, test_accuracy = test(cnn, device, test_loader, criterion)

torch.save(cnn.state_dict(), 'cnn_mnist_model.pth')