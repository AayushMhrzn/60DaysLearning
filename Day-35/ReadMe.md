
#  Day 35 — Introduction to PyTorch

Today’s focus was on understanding the **fundamentals of PyTorch**, a powerful deep learning framework widely used for research and production.

---

##  1. PyTorch Tensors

Tensors are the core building blocks of PyTorch, similar to NumPy arrays but with GPU acceleration.

```python
import torch

x = torch.ones(2, 3)
print(x)
print(x.shape)
print(x.device)  # Default: CPU
```

### Key Operations:
- `torch.tensor()` → Create tensor from list/array
- `x.view()` → Reshape tensor
- `x.to(device)` → Move to GPU
- Arithmetic: `+`, `-`, `*`, `/`, `matmul`, etc.

---

##  Loading Datasets and DataLoaders

PyTorch provides utilities to handle image/text datasets.

```python
from torchvision import datasets
from torch.utils.data import DataLoader

train_data = datasets.FakeData(transform=transform)
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
```

You can also use datasets like `MNIST`, `CIFAR10`, `ImageNet`.

---

## Transforms

Transforms help preprocess the data (resizing, normalization, augmentation).

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```
---

##  Building a Neural Network Model

Using `nn.Module` to define a model:

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3*64*64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 3*64*64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```
- torch.nn: Contains high-level building blocks like Linear, Conv2d, etc.
- torch.nn.functional: Contains functions like activation functions (F.relu) that don’t have learnable parameters.
- Subclasses nn.Module, which is the base class for all neural networks in PyTorch.

`self.fc1 = nn.Linear(3*64*64, 128)`
-Input: flattened image of shape 3×64×64 = 12288 pixels.
-Output: 128-dimensional feature vector.
- nn.Linear(in_features, out_features) creates a fully connected layer.

`self.fc2 = nn.Linear(128, 64)`
- Maps from 128 → 64 hidden units.

`self.fc3 = nn.Linear(64, 10)`
- Final output layer: maps 64 → 10 (for 10 class classification).

`forward() Method: Defines Forward Pass`
- Defines how input data flows through the network.

`x = x.view(-1, 3*64*64)`
- Flattens the image tensor.
- -1 lets PyTorch infer the batch size.
- Converts image shape from [batch_size, 3, 64, 64] to [batch_size, 12288].

| Layer | Operation                   | Output Shape       |
| ----- | --------------------------- | ------------------ |
| Input | RGB Image (3×64×64)         | `[batch, 3×64×64]` |
| `fc1` | Linear + ReLU (12288 → 128) | `[batch, 128]`     |
| `fc2` | Linear + ReLU (128 → 64)    | `[batch, 64]`      |
| `fc3` | Linear (64 → 10)            | `[batch, 10]`      |


---

##  Autograd: Automatic Differentiation

PyTorch uses **autograd** to compute gradients automatically for backpropagation.

```python
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()
out.backward()  # computes gradients

print(x.grad)  # prints the gradient d(out)/dx
```

---

##  Optimizing Model Parameters

Use loss functions and optimizers for training:

```python
import torch.optim as optim

model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training loop
for epoch in range(2):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

---

##  Saving and Loading Models

Save and load models for reuse or deployment:

```python
# Save model
torch.save(model.state_dict(), 'model.pth')

# Load model
net_loaded = SimpleNet()
net_loaded.load_state_dict(torch.load('model.pth'))
net_loaded.eval()
```

- `eval()` sets model to evaluation mode (e.g., dropout is disabled).
- Useful for inference or fine-tuning later.

---
