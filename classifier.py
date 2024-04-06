import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Load and preprocess the MNIST dataset

# Takes an image from the Python Image Library and converts it to a PyTorch Tensor while scaling image pixel values
# from [0, 255] to the range [0, 1]. 
trans_to_tensor = transforms.ToTensor()

# Normalize the pixel values so that they are centered around 0 and have a standard deviation of 0.5
trans_to_normal =transforms.Normalize((0.5,), (0.5,))

# Apply multiple transformations together
transform = transforms.Compose([trans_to_tensor, trans_to_normal])

# Loads in MNIST training data to /data directory
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Loads the data in batches of 64 images, shuffles images before each epoch to generalize
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Loads in test dataset
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Simple nueral network for classifying images
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

net = Net()


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


for epoch in range(5):
    running_loss = 0.0
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')


correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')
