import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
x_lower = 0
x_upper = 1
y_lower = 0
y_upper = 1

x = np.linspace(x_lower, x_upper, 256)
y = np.linspace(y_lower, y_upper, 256)

x_domain = dde.geometry.Interval(x_lower, x_upper)
y_domain = dde.geometry.Interval(y_lower, y_upper)

geom = dde.geometry.geometry_2d.Rectangle([0, 0], [1, 1])

# def gradients(u, x, order=1):
#     if order == 1:
#         return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
#                                    create_graph=True,
#                                    only_inputs=True, )[0]
#     else:
#         return gradients(gradients(u, x), x, order=order - 1)

def pde(X, u):
    # X:(x, y) , the input of the net, u: the output of the net
    du2_d2x = dde.grad.hessian(u, X, j=0)
    du2_d2y = dde.grad.hessian(u, X, j=1)
    return du2_d2x + du2_d2y

def down_boundary(X, on_boundary):
    # X[1]是y分量
    return on_boundary and np.isclose(X[1], 0)
def left_boundary(X, on_boundary):
    return on_boundary and np.isclose(X[0], 0)
def upper_boundary(X, on_boundary):
    return on_boundary and np.isclose(X[1], 1)

def right_boundary(X, on_boundary):
    return on_boundary and np.isclose(X[0], 1)

def dirichlet_boundary(X):
    return np.zeros_like(0)

def neumann_boundary(X):
    return np.sin(X[:, 0:1])

def dirichlet_boundary2(X):
    return np.sin(1) * np.sinh(X[:, 1:2])

def dirichlet_boundary3(X):
    return np.sin(X[:, 0:1]) * np.sinh(1)

bc_dirichlet1 = dde.DirichletBC(geom, dirichlet_boundary, down_boundary)
bc_dirichlet2 = dde.DirichletBC(geom, dirichlet_boundary, left_boundary)
bc_dirichlet3 = dde.DirichletBC(geom, dirichlet_boundary2, right_boundary)
bc_dirichlet4 = dde.DirichletBC(geom, dirichlet_boundary3, upper_boundary)
bc_neumann = dde.NeumannBC(geom, neumann_boundary, down_boundary)

data = dde.data.PDE(
    geom,
    pde,
    [bc_dirichlet1, bc_dirichlet2, bc_dirichlet3, bc_dirichlet4, bc_neumann],
    num_domain= 10000,
    num_boundary= 10000,
    num_test= 1000,
)

net = dde.nn.pytorch.FNN([2, 32, 32, 32, 1], "tanh", "Glorot uniform")
optimizer = "adam"

model = dde.Model(data, net)
model.compile(optimizer, lr=1e-3)

loss_history, train_state = model.train(10000)

# Inference
xc = torch.linspace(0, 1, 100)
xx, yy = torch.meshgrid(xc, xc, indexing='ij')
xx = xx.reshape(-1, 1)
yy = yy.reshape(-1, 1)
xy = torch.cat([xx, yy], dim=1)
# print(xy.shape)
u_pred = model.predict(xy)

df = pd.DataFrame(np.reshape(u_pred, (100, 100)))
df.to_csv("laplace_predict.csv")

print("Max abs error is: ", float(torch.max(torch.abs(torch.from_numpy(u_pred) - torch.sinh(yy) * torch.sin(xx)))))

plt.imshow(np.reshape(u_pred, (100, 100)))
plt.colorbar()
plt.tight_layout()
# plt.show()
plt.savefig("laplace_predict.png")
