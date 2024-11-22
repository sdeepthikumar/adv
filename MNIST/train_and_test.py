import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import datetime
import os

# Define the neural network model
class SimpleDNN(nn.Module):
    def __init__(self):
        super(SimpleDNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 7 * 7, 16)
        self.fc3 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc3(x)
        return x

# Load and preprocess the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model, loss function, and optimizer
model = SimpleDNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
model.train()
for epoch in range(1):  # Train for 1 epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 1, 28, 28)  # Ensure data is in the correct shape
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        data = data.view(-1, 1, 28, 28)  # Ensure data is in the correct shape
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

test_acc = correct / total

print("Total number of parameters:", sum(p.numel() for p in model.parameters()))
print("Accuracy:", test_acc)
# Model validation checks
assert sum(p.numel() for p in model.parameters()) < 100000, "Model has more than 100,000 parameters"
assert model.fc3.out_features == 10, "Model output shape is incorrect"
assert test_acc > 0.80, "Model accuracy is less than 80%"

# Create the models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the model with a timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
torch.save(model.state_dict(), f"models/model_{timestamp}.pth") 