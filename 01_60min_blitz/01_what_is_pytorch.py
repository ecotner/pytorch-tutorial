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

# # What is PyTorch?

import torch

# ## The torch.Tensor object

# uninitialized matrix
x = torch.empty(5, 3)
print(x)
print(type(x))

# randomly initialized matrix
x = torch.randn(5, 3)
print(x)

# zeros with dtype long (64-bit int?)
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# constructed directly from data
x = torch.tensor([5.5, 3])
print(x)

# +
# create a tensor based on an existing tensor
x = x.new_ones(5, 3, dtype=torch.double)
print(x)

x = torch.randn_like(x, dtype=torch.float)
print(x)
# -

# size/dimensions
print(x.size())
print(x.shape)

# ## Operations

y = torch.rand(5, 3)
print(x + y)

print(torch.add(x, y))

# provide an output tensor as an argument
# kinda like numpy's out arguments
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# Another way of adding
y.add(x)

# post-fixing an operation with `_` mutates a tensor in-place
y.add_(x)
print(y)

# can use numpy-like slicing
print(y[:, 1])
print(y[0,:-1])

# resize tensor
# apparently using .resize() works, but is deprecated
x = torch.randn(4, 4)
print(x.view(16))
print(x.view(-1, 8))
print(x.view(-1, 8).size())

# Using .item() on a scalar returns a python number
x = torch.randn(1)
print(x)
print(x.item())

# ## Converting back and forth between NumPy

# +
a = torch.ones(5)
print(a)

# convert to numpy
b = a.numpy()
print(b)

# and back to Tensor
import numpy as np
c = torch.from_numpy(b)
d = torch.tensor(b)
print(c)
print(d)
# -

# The tensor created with torch.from_numpy()is linked to the numpy array,
# so be careful of mutations. The one created from torch.tensor isn't though
np.add(b, 1, out=b)
print(b)
print(c)
print(d)

# ## CUDA Tensors

# +
# check if CUDA is available
print("CUDA available: ", torch.cuda.is_available())

if torch.cuda.is_available():
    device = torch.device("cuda") # CUDA device object
    y = torch.ones_like(x, device=device) # directly create tensor on GPU
    x = x.to(device) # convert device using .to()
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))
# -


