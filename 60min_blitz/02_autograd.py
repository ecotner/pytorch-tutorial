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

# # Autograd: automatic differentiation

# `autograd` is pytorch's automatic differentiation package, and is defined by how the code is run, not by a static graph like tensorflow (though Eager Mode in TF2 changes things)

import torch
import itertools as it

# Tensors initialized with data do not have their gradients computed automatically
x = torch.tensor([2, 2])
print(x)
x.requires_grad

x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)

# +
# Because y was created as a result of an operation with a tensor
# whose gradient is being tracked, it has a function associated with it

print(y.grad_fn)
# The tensor we created does not
print(x.grad_fn)

# +
# Some more ops
z = y * y * 3
out = z.mean()

print(z, out, sep='\n')

# +
# .requires_grad_() changes a tensor's requires_grad flag in-place (note the
# post-fixed '_')
# Should probably not set the attribute directly...

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)
# -

# ## Gradients

# backprop through a single scalar using .backward() method
print(out)
out.backward()

# +
# This computed the gradient of `out` wrt every tensor in the computational
# graph
# remember y = x + 2, z = y * y * 3, and out = z.mean()
# Since out is a scalar and x is a (2, 2) tensor, that means the gradient
# d(out)/dx is a (2, 2) tensor as well

print(x)
print(x.grad)

# +
# If we take a gradient of a non-scalar tensor, the dimension changes
# For example, the gradient of a (3,) tensor (aka a vector) with another (3,)
# tensor is a (3, 3) tensor

x = torch.randn(3, requires_grad=True)
y = 2 * x
while y.data.norm() < 1000:
    y = y * 2
print(y, end='\n\n')

J = list()
for i in range(y.shape[0]):
    y[i].backward(retain_graph=True)
    J.append(x.grad.clone())
    x.grad.zero_() # Need to zero out the gradient each time
                   # because it is accumulated by default
print(torch.stack(J))
# -

# We can see the above Jacobian is proportional to the identity matrix, which makes sense because $y_i = 2^n x_i$ and doesn't depend on any of the other elements of $\vec{x}$, so $dy_i/dx_j = 0$ for $i \neq j$.

# +
# Dot product of gradient with the vector v
x.grad.zero_() # have to zero the gradient from the 
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v, retain_graph=True)

print(x.grad)
# -

# You can stop autograd from tracking history on Tensors with a context manager
print(x.requires_grad)
print((x ** 2).requires_grad)
with torch.no_grad():
    print((x ** 2).requires_grad)

# or we can get a copy of the tensor that does not require gradients
print(x.requires_grad)
y = x.detach()
print(y.requires_grad)
print(x.eq(y).all())


