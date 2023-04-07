import torch
import numpy as np
import matplotlib.pyplot as plt

x = torch.ones(2,2,requires_grad=True)
print('x:\n',x)
y = torch.eye(2,2,requires_grad=True)
print("y:\n",y)
z = x**2+y**3
z.backward(torch.ones_like(x))
print(x.grad,'\n',y.grad)
