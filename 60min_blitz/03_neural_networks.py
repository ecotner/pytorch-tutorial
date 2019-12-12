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

# # Neural networks

# Neural networks can be constructed with the `torch.nn` package.

# ## Define the network

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# +
class Net(nn.Module):
    """Convolutional neural network
    
    A simple CNN. Assumes a fixed input image size of (24, 24, 1)
    """
    def __init__(self):
        super(Net, self).__init__()
        # Define the weights/layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2, 2))
        # If the size is a square you only need a single scalar
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)
# -

# Recall the number of output neurons $n_{out}$ is related to the number of neurons $n_{in}$, filter size $f$, and stride $s$ by:
# $$ n_{in} = f + s(n_{out}-1) $$
# $$ n_{out} = 1+ \lceil(n_{in} - f)/s\rceil $$

# The learnable parameters of the model
params = list(net.parameters())
print(len(params))
print(params[0].size())

x = torch.randn(1, 1, 32, 32)
y = net(x)

net.zero_grad()
y.backward(torch.randn(1, 10))

net.conv1.weight.grad.shape

# ## Loss function

# +
y = net(x)
Y = torch.randn(10) # some dummy target
Y = Y.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(y, Y)
print(loss)
# -

print(loss.grad_fn) # MSELoss
print(loss.grad_fn.next_functions[0][0]) # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) # ReLU

# ## Backprop

# +
net.zero_grad()

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
# -

# ## Updating the weights

# recall weights are updated via
# $$ W \leftarrow W - \alpha \nabla_W \mathcal{L}$$
# where $W$ are the weights of the network, $\alpha$ is the learning rate, and $\mathcal{L}$ is the loss

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# +
# using built-in optimizers
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
def train():
    optimizer.zero_grad()
    y = net(x)
    loss = criterion(y, Y)
    loss.backward()
    optimizer.step() # does the update
    return loss


# -

for _ in range(15):
    loss = train()
    print(loss.item())


