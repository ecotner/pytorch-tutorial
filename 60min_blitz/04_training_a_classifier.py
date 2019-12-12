# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: pytorch-tutorial
#     language: python
#     name: pytorch-tutorial
# ---

# # Training a classifier

# We're going to train an image classifier on the CIFAR10 dataset. Pytorch has some modules that has data loaders for common datasets: `torchvision.datasets` and `torch.utils.data.DataLoader`

# We will do the following steps:
# 1. Load and preprocess the training data
# 2. Define a CNN
# 3. Define a loss function
# 4. Train the network
# 5. Test the network on test data

# ## 1. Loading CIFAR10 data

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Let's download the train/test set data from a remote repository

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                         shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                      download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                        shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
          'horse', 'ship', 'truck')


# Visually inspect some of the training images

# +
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
# -

# ## Define a CNN

# The dataloader makes minibatches with 4 samples, each with
# 3 channels, and (32, 32) size
images.shape

# +
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # (6, 28, 28)
        self.pool = nn.MaxPool2d(2, 2) # (6, 14, 14)
        self.conv2 = nn.Conv2d(6, 16, 5) # (16, 10, 10)
        # Another pool operation here (16, 5, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
net = Net()
# -

# ## Loss function and optimizer

# +
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# -

# ## Train the network

# +
for epoch in range(2):
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
            print('[%d, %5s] loss: %.3f' %
                 (epoch+1, i+1, running_loss/2000))
            running_loss = 0.0

print('Finished training')
# -

# save our model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

# ## Test the network on test data

# display some images from the test set
dataiter = iter(testloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# +
# reload the saved network state
net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)

_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

# +
# See how the network performs on the whole dataset
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
print('Accuracy of the network on 10,000 test images: %d %%' % (
    100 * correct/ total))

# +
# what are classes that performed well?
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
            
for i in range(10):
    print('Accuracy of %5s: %2d %%' % (
        classes[i], 100 * class_correct[i]/class_total[i]))
# -

# ## Train on GPU

device = torch.device("cuda:0")
print(device)

# It's really that simple??
net.to(device)

# +
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = (d.to(device) for d in data)
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5s] loss: %.3f' %
                 (epoch+1, i+1, running_loss/2000))
            running_loss = 0.0

print('Finished training')
# -

