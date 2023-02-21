'''
1D Burgers Problem:

$$
\begin{aligned}
& u_t+u u_x-(0.01 / \pi) u_{x x}=0, \quad x \in[-1,1], \quad t \in[0,1], \\
& u(0, x)=-\sin (\pi x) \\
& u(t,-1)=u(t, 1)=0 .
\end{aligned}
$$
'''

import deepxde as dde
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x_lower = -1
x_upper = 1
t_lower = 0
t_upper = 1

geom = dde.geometry.Interval(x_lower, x_upper)
time = dde.geometry.TimeDomain(t_lower, t_upper)
geomtime = dde.geometry.GeometryXTime(geom, time)

def pde(X, u):
    # X: the input, arrays consist of [x, t]
    # u: the output
    du_dt = dde.grad.jacobian(u, X, j=1)
    du_dx = dde.grad.jacobian(u, X, j=0)
    d2u_dx2 = dde.grad.hessian(u, X, j=0)
    
    return du_dt + u * du_dx - (0.01 / np.pi) * d2u_dx2

def initial_func(X):
    return -np.sin(np.pi * X[:, 0:1])

def dirichlet_boundary(X):
    return np.zeros_like(0)

ic = dde.IC(geomtime, initial_func, lambda _, on_initial:on_initial)
bc = dde.DirichletBC(geomtime, dirichlet_boundary, lambda _, on_boundary: on_boundary)

data = dde.data.TimePDE(
    geomtime,
    pde,
    [ic, bc],
    num_domain=4000,
    num_boundary=2000,
    num_initial=2000,
    num_test=2000
)

net = dde.nn.pytorch.FNN([2, 32, 32, 32, 1], "tanh", "Glorot uniform")
optimizer = "adam"

model = dde.Model(data, net)
model.compile(optimizer, lr=1e-3)

loss_history, train_state = model.train(10000, model_save_path="model/burgers")

# Inference
x = torch.linspace(x_lower, x_upper, 100)
t = torch.linspace(t_lower, t_upper, 100)
xx, tt = torch.meshgrid(x, t, indexing='ij')
xx = xx.reshape(-1, 1)
tt = tt.reshape(-1, 1)
xt = torch.cat([xx, tt], dim=1)
# print(xy.shape)
u_pred = model.predict(xt)
residual = model.predict(xt, operator=pde)

df = pd.DataFrame(np.reshape(u_pred, (100, 100)))
df.to_csv("burgers_predict.csv")


print("Max residual is: ", float(np.max(np.abs(residual))))

plt.imshow(np.reshape(u_pred, (100, 100)))
# plt.imshow(np.reshape(residual, (100, 100)))
plt.colorbar()
plt.tight_layout()
# plt.show()
plt.savefig("burgers_predict.png")
# plt.savefig("burgers_residual.png")