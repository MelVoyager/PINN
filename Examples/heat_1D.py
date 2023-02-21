import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt

geom = dde.geometry.Interval(-1, 1)
time = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, time)

def pde(x, y):
    # 这里的x是一个形如(space, time)的二维向量，y是一个数
    dy_xx = dde.grad.hessian(y, x, j = 0)
    dy_t = dde.grad.jacobian(y, x, j = 1)
    
    return dy_t - 0.3 * dy_xx

def func(x):
    return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])

bc = dde.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
ic = dde.IC(geomtime, func, lambda _, on_initial: on_initial)

data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc, ic],
    num_domain= 4000,
    num_boundary= 2000,
    num_initial= 1000,
    num_test= 1000,
    solution= func,
)

net = dde.nn.pytorch.FNN([2, 32, 32, 32, 1], "tanh", "Glorot uniform")
optimizer = "adam"

model = dde.Model(data, net)
model.compile(optimizer, lr=1e-3, metrics=["l2 relative error"])

loss_history, train_state = model.train(10000)

x_data = np.linspace(-1, 1, num=100)
t_data = np.linspace(0, 1, num=100)
test_x, test_y = np.meshgrid(x_data, t_data)
test_domain = np.vstack((np.ravel(test_x), np.ravel(test_y))).T
predict_solution = model.predict(test_domain)
residual = model.predict(test_domain, operator=pde)
residual = np.reshape(residual, (100, 100))
plt.imshow(residual)
plt.colorbar()
plt.tight_layout()
plt.show()