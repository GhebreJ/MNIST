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
# This helps the neural network converge faster and generalize better
trans_to_normal = transforms.Normalize((0.5,), (0.5,))

# Apply multiple transformations together
# Compose is used to chain multiple transformations sequentially
transform = transforms.Compose([trans_to_tensor, trans_to_normal])

# Loads in MNIST training data to /data directory
# The MNIST dataset consists of handwritten digit images and their corresponding labels
# train=True specifies that we want the training dataset
# download=True automatically downloads the dataset if it's not already present
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Loads the data in batches of 64 images, shuffles images before each epoch to generalize
# DataLoader is used to create an iterator that returns batches of images and labels
# batch_size determines the number of samples in each batch
# shuffle=True randomizes the order of the samples in each epoch
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Loads in test dataset
# train=False specifies that we want the test dataset
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Simple neural network for classifying images
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Fully connected layer with 28 * 28 input features and 128 output features
        # 28 * 28 is the size of each flattened MNIST image
        self.fc1 = nn.Linear(28 * 28, 128)
        # Fully connected layer with 128 input features and 10 output features
        # 10 corresponds to the number of classes (digits 0-9)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Reshape the input tensor to a flat vector
        x = x.view(-1, 28 * 28)
        # Apply ReLU activation function to the output of the first fully connected layer
        # ReLU introduces non-linearity and helps the network learn complex patterns
        x = F.relu(self.fc1(x))
        # Pass the output through the second fully connected layer
        x = self.fc2(x)
        # Apply log softmax activation function to the output
        # Log softmax converts the output scores into log probabilities
        return F.log_softmax(x, dim=1)

# Create an instance of the neural network
net = Net()

# Define the loss function
# CrossEntropyLoss combines log softmax and negative log-likelihood loss
criterion = nn.CrossEntropyLoss()

# Define the optimizer
# Adam is an adaptive optimization algorithm that adjusts the learning rate for each parameter
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Training loop
for epoch in range(5):
    running_loss = 0.0
    for images, labels in trainloader:
        # Zero the gradients of the optimizer
        optimizer.zero_grad()
        
        # Forward pass: compute the outputs of the network
        outputs = net(images)
        
        # Compute the loss between the predicted outputs and the true labels
        loss = criterion(outputs, labels)
        
        # Backward pass: compute the gradients of the loss with respect to the network parameters
        loss.backward()
        
        # Update the network parameters using the computed gradients
        optimizer.step()
        
        # Accumulate the running loss
        running_loss += loss.item()
    
    # Print the average loss for the current epoch
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')

# Evaluation loop
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        # Compute the outputs of the network for the test images
        outputs = net(images)
        
        # Get the predicted class labels by taking the argmax of the output probabilities
        _, predicted = torch.max(outputs.data, 1)
        
        # Update the total number of images and the number of correctly classified images
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Print the overall accuracy of the network on the test dataset
print(f'Accuracy: {100 * correct / total}%')