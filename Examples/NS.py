import deepxde as dde
from deepxde.backend import tf
import numpy as np
import matplotlib.pyplot as plt

geom = dde.geometry.geometry_2d.Rectangle([0,0],[1,0.5])
rho = 1000.
mu = 1e-3

def navier_stokes(x,y):
    u, v, p = y[:, 0:1], y[:, 1:2], y[:, 2:]
    u_x = dde.grad.jacobian(y, x, i=0, j=0)
    u_y = dde.grad.jacobian(y, x, i=0, j=1)
    v_x = dde.grad.jacobian(y, x, i=1, j=0)
    v_y = dde.grad.jacobian(y, x, i=1, j=1)
    p_x = dde.grad.jacobian(y, x, i=2, j=0)
    p_y = dde.grad.jacobian(y, x, i=2, j=1)
    u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    u_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    v_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    continuity = u_x + v_y
    x_momentum = rho * (u * u_x + v * u_y) + p_x - mu * (u_xx + u_yy)
    y_momentum = rho * (u * v_x + v * v_y) + p_y - mu * (v_xx + v_yy)
    return [continuity, x_momentum, y_momentum]

def top_wall(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0.5) 

def bottom_wall(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0.0)

def inlet_boundary(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0.0)

def outlet_boundary(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1.0)

def parabolic_velocity(x):
    """Parabolic velocity"""
    return ((-6 * x[:, 1] ** (2)) + (3 * x[:, 1])).reshape(-1, 1)

def zero_velocity_top_wall(x):
    """Zero velocity"""
    return np.zeros_like(0)

def zero_velocity_inlet(x):
    """Zero velocity"""
    return np.zeros_like(0)

def zero_velocity_bottom_wall(x):
    """Zero velocity"""
    return np.zeros_like(0)


inlet_u = dde.icbc.boundary_conditions.DirichletBC(geom, parabolic_velocity, inlet_boundary, component=0)
inlet_v = dde.icbc.boundary_conditions.DirichletBC(geom, zero_velocity_inlet, inlet_boundary, component=1)
top_wall_u = dde.icbc.boundary_conditions.DirichletBC(geom, zero_velocity_top_wall, top_wall, component=0)
top_wall_v = dde.icbc.boundary_conditions.DirichletBC(geom, zero_velocity_top_wall, top_wall, component=1)
bottom_wall_u = dde.icbc.boundary_conditions.DirichletBC(geom, zero_velocity_bottom_wall, bottom_wall, component=0)
bottom_wall_v = dde.icbc.boundary_conditions.DirichletBC(geom, zero_velocity_bottom_wall, bottom_wall, component=1)

data = dde.data.PDE(geom, navier_stokes, [inlet_u, inlet_v, top_wall_u, top_wall_v, bottom_wall_u, bottom_wall_v], num_domain=1000, num_boundary=100, num_test=1000)

net = dde.nn.FNN([2] + [50] * 4 + [3], "tanh", "Glorot uniform")

model = dde.Model(data, net)
model.compile("adam", lr=0.001)
loss_history, train_state = model.train(epochs=5000)
dde.saveplot(loss_history, train_state, issave=True, isplot=True)