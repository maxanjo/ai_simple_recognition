import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()    # Calls init function of parent class. (nn.Module)
        self.fc1 = nn.Linear(28 * 28, 128)  # Fully connected layer 1
        self.fc2 = nn.Linear(128, 64)       # Fully connected layer 2
        self.fc3 = nn.Linear(64, 10)        # Fully connected layer 3 (output layer)

    def forward(self, x):
        x = x.view(-1, 28 * 28)             # Flatten the input
        x = torch.relu(self.fc1(x))         # Apply ReLU activation to layer 1
        x = torch.relu(self.fc2(x))         # Apply ReLU activation to layer 2
        x = self.fc3(x)                     # Output layer (no activation)
        return x

net = SimpleNN()
# Function to visualize the image
def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)



# Get some random test images
dataiter = iter(testloader)
images, labels = next(dataiter)

# Print images and labels
print('GroundTruth: ', ' '.join('%5s' % labels[j].item() for j in range(8)))

# Visualize the images
imshow(torchvision.utils.make_grid(images[:8]))

# Predict the labels using the trained network
outputs = net(images[:8])
_, predicted = torch.max(outputs, 1)

# Print the predictions
print('Predicted: ', ' '.join('%5s' % predicted[j].item() for j in range(8)))

# Display each image with its prediction
for i in range(8):
    plt.subplot(2, 4, i + 1)
    plt.imshow(images[i].numpy().squeeze(), cmap='gray_r')
    plt.title(f"GT: {labels[i].item()} Pred: {predicted[i].item()}")
    plt.axis('off')
plt.show()
