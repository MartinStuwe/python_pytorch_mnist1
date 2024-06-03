import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

list = []

# Define the CNN model class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1featuremap = []
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.4)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=9216, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)


    def forward(self, x):
        x = self.conv1(x)
        self.conv1featuremap.append(x)
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
        output = F.log_softmax(x, dim=1)
        return output

# Load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn_loaded = CNN().to(device)
cnn_loaded.load_state_dict(torch.load('cnn_mnist_model.pth'))
cnn_loaded.eval()

# Load MNIST test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.1307], [0.3081])
])

mnist_testset = MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(mnist_testset, batch_size=64, shuffle=False)

# Function to test the loaded model and visualize images
def test_and_visualize(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    images, predictions, labels = [], [], []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            images.append(data.cpu())
            predictions.append(predicted.cpu())
            labels.append(target.cpu())
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    print(f'Test Loss: {test_loss}, Accuracy: {accuracy}')

    # Flatten the lists
    images = torch.cat(images)
    predictions = torch.cat(predictions)
    labels = torch.cat(labels)

    # Plot the images with their predictions
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.flatten()
    for i in range(16):
        img = images[i].squeeze()
        ax = axes[i]
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Pred: {predictions[i].item()}, True: {labels[i].item()}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Test the loaded model and visualize the results
test_and_visualize(cnn_loaded, device, test_loader, criterion)

model_weights = []
conv_layers = []

"""
model_childs = list(cnn_loaded.children())
counter = 0

for i in range(len(model_childs)):
    if type(model_childs[i] == nn.Conv2d):
        counter += 1
        model_weights.append(model_childs[i].weight)
        conv_layers.append(model_childs[i])


print(f"total conv layers: {counter}")
print(f"conv layers: {conv_layers}")


feature_maps = {}
def hook_fn(m, i, o):
    feature_maps[m] = o
"""

print(cnn_loaded.conv1featuremap[0])
print(cnn_loaded.conv1featuremap[0].size())

feature_map = cnn_loaded.conv1featuremap[0][0].cpu()

fig, axes = plt.subplots(4, 8, figsize=(15,8))
axes = axes.flatten()
for i in range(feature_map.shape[0]):
    ax = axes[i]
    ax.imshow(feature_map[i], cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.show()
