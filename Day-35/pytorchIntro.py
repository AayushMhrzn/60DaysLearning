# Day 35 - Introduction to PyTorch

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. Load and Normalize CIFAR10 Dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

# 2. Visualize Some Images
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    plt.imshow(torch.permute(img, (1, 2, 0)))
    plt.show()

dataiter = iter(trainloader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images))

# 3. Define a CNN Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)     # in_channels, out_channels, kernel
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # conv1 + relu + pool
        x = self.pool(F.relu(self.conv2(x))) # conv2 + relu + pool
        x = x.view(-1, 16 * 5 * 5)           # flatten
        x = F.relu(self.fc1(x))              # FC1
        x = F.relu(self.fc2(x))              # FC2
        x = self.fc3(x)                      # FC3
        return x

net = Net()

# 4. Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 5. Train the Network (for 1 epoch for demo)
for epoch in range(1):  
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
            running_loss = 0.0

print("Finished Training")

# 6. Save and Load the Model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

net_loaded = Net()
net_loaded.load_state_dict(torch.load(PATH))
net_loaded.eval()
