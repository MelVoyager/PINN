import functools
import numpy as np
import torch
from scipy import interpolate

import deepxde as dde
from . import baseclass
from ..utils.random import generate_darcy_2d_coef


class WaveHetergeneous(baseclass.BasePDE):

    def __init__(self, bbox=[-1, 1, -1, 1, 0, 10], circ=[(0.3, 0, 0.3)], mu=(-0.5, 0), sigma=-0.3):
        super().__init__()
        # output dim
        self.output_dim = 1
        # geom (notice: no circ are deleted, since the pde is currently not regraded as TimePDE and 3D-CSGDifference is difficult)
        self.bbox = bbox
        self.geom = dde.geometry.Hypercube(xmin=(self.bbox[0], self.bbox[2], self.bbox[4]), xmax=(self.bbox[1], self.bbox[3], self.bbox[5]))

        # PDE
        # self.darcy_2d_coef = generate_darcy_2d_coef(N_res=256, alpha=4, bbox=bbox[0:4])
        self.darcy_2d_coef = np.loadtxt("ref/darcy_2d_coef_256.dat")

        @functools.cache
        def coef(x):
            return torch.Tensor(
                interpolate.griddata(self.darcy_2d_coef[:, 0:2], self.darcy_2d_coef[:, 2], x.detach().cpu().numpy()[:, 0:2], method="nearest")
            )

        def wave_pde(x, u):
            u_xx = dde.grad.hessian(u, x, i=0, j=0) + dde.grad.hessian(u, x, i=1, j=1)
            u_tt = dde.grad.hessian(u, x, i=2, j=2)

            return u_xx - u_tt / coef(x)

        self.pde = wave_pde
        self.set_pdeloss(num=1)

        # TODO: add reference data

        # BCs
        def boundary_t0(x, on_initial):
            return np.isclose(x[2], bbox[4])

        def boundary_condition(x):
            return np.exp(-((x[:, 0:1] - mu[0])**2 + (x[:, 1:2] - mu[1])**2) / (2 * sigma**2))

        self.add_bcs([{
            'component': 0,
            'function': boundary_condition,
            'bc': boundary_t0,
            'type': 'dirichlet'
        }, {
            'component': 0,
            'function': (lambda _: 0),
            'bc': boundary_t0,
            'type': 'neumann'
        }])

        # training config
        self.training_points(domain=4096, boundary=1024)


class WaveEquation1D(baseclass.BaseTimePDE):

    def __init__(self, C=2, bbox=[0, 1, 0, 1]):
        super().__init__()

        # output dim
        self.output_dim = 1
        # geom
        self.bbox = bbox
        self.geom = dde.geometry.Interval(self.bbox[0], self.bbox[1])
        timedomain = dde.geometry.TimeDomain(self.bbox[2], self.bbox[3])
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)

        # 定义方程
        def wave_pde(x, u):
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            u_tt = dde.grad.hessian(u, x, i=1, j=1)

            return u_tt - C * C * u_xx

        self.pde = wave_pde
        self.set_pdeloss(num=1)

        def ref_sol(x):
            return (np.sin(np.pi * x[:, 0:1]) * np.cos(2 * np.pi * x[:, 1:2]) + 0.5 * np.sin(4 * np.pi * x[:, 0:1]) * np.cos(8 * np.pi * x[:, 1:2]))

        self.ref_sol = ref_sol

        self.add_bcs([{
            'component': 0,
            'function': ref_sol,
            'bc': (lambda _, on_boundary: on_boundary),
            'type': 'dirichlet'
        }, {
            'component': 0,
            'function': ref_sol,
            'bc': (lambda _, on_initial: on_initial),
            'type': 'ic'
        }])

        # training config
        self.training_points(domain=4096, boundary=1024, initial=1024)


class WaveEquation2D_Long(baseclass.BaseTimePDE):

    def __init__(self, bbox=[0, 1, 0, 1, 0, 100], a=20, m1=1, m2=1, n1=1, n2=1, p1=1, p2=1):
        super().__init__()

        # output dim
        self.output_dim = 1
        # geom
        self.bbox = bbox
        self.geom = dde.geometry.Rectangle(xmin=[bbox[0], bbox[2]], xmax=[bbox[1], bbox[3]])
        timedomain = dde.geometry.TimeDomain(bbox[4], bbox[5])
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)

        # pde
        INITIAL_COEF_1 = 1
        INITIAL_COEF_2 = 1

        def pde(x, u):
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            u_yy = dde.grad.hessian(u, x, i=1, j=1)
            u_tt = dde.grad.hessian(u, x, i=2, j=2)

            return [u_tt - (u_xx + a * a * u_yy)]

        self.pde = pde
        self.set_pdeloss(num=1)

        # BCs
        def ref_sol(x):
            return (
                INITIAL_COEF_1 * np.sin(m1 * np.pi * x[:, 0:1]) * np.sinh(n1 * np.pi * x[:, 1:2]) * np.cos(p1 * np.pi * x[:, 2:3])
                + INITIAL_COEF_2 * np.sinh(m2 * np.pi * x[:, 0:1]) * np.sin(n2 * np.pi * x[:, 1:2]) * np.cos(p2 * np.pi * x[:, 2:3])
            )

        self.ref_sol = ref_sol

        self.add_bcs([{
            'component': 0,
            'function': ref_sol,
            'bc': (lambda _, on_initial: on_initial),
            'type': 'ic'
        }, {
            'component': 0,
            'function': ref_sol,
            'bc': (lambda _, on_boundary: on_boundary),
            'type': 'dirichlet'
        }])

        # training config
        self.training_points(domain=4096, boundary=1024, initial=1024)
