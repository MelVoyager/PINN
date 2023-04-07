import functools
import numpy as np
import torch
import deepxde as dde

from . import baseclass


class Poisson1D(baseclass.BasePDE):

    def __init__(self, a=1):
        super().__init__()
        # Output Dim
        self.output_dim = 1
        # Domain
        self.bbox = [0, 2 * np.pi / a]
        self.geom = dde.geometry.Interval(*self.bbox)

        # PDE
        def f(x):
            return a**2 * torch.sin(a * x)

        def pde(x, u):
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            return u_xx + f(x)

        self.pde = pde
        self.set_pdeloss(num=1)

        # ref_sol
        def ref_sol(x):
            return np.sin(a * x)

        self.ref_sol = ref_sol

        # bcs
        self.add_bcs([{'component': 0, 'function': (lambda x: 0), 'bc': (lambda x, on_boundary: on_boundary), 'type': 'dirichlet'}])

        self.training_points()


class PoissonClassic(baseclass.BasePDE):

    def __init__(self, scale=8):
        super().__init__()
        # Output Dim
        self.output_dim = 1
        # Domain
        self.bbox = [-scale / 2, scale / 2, -scale / 2, scale / 2]
        self.geom = dde.geometry.Rectangle(xmin=[-scale / 2, -scale / 2], xmax=[scale / 2, scale / 2])
        circ = np.array([[0.3, 0.3, 0.1], [-0.3, 0.3, 0.1], [0.3, -0.3, 0.1], [-0.3, -0.3, 0.1]]) * scale
        for c in circ:
            disk = dde.geometry.Disk(c[0:2], c[2])
            self.geom = dde.geometry.CSGDifference(self.geom, disk)

        # PDE
        def pde(x, u):
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            u_yy = dde.grad.hessian(u, x, i=1, j=1)

            return [u_xx + u_yy]

        self.pde = pde
        self.set_pdeloss(num=1)

        def transform_fn(data):
            data[:, :self.input_dim] *= scale
            return data

        self.load_ref_data("ref/poisson_classic.dat", transform_fn=transform_fn)

        def rec_boundary(x, on_boundary):
            return on_boundary and (
                np.isclose(x[0], self.bbox[0]) or np.isclose(x[0], self.bbox[1]) or np.isclose(x[1], self.bbox[2]) or np.isclose(x[1], self.bbox[3])
            )

        def circ_boundary(x, on_boundary):
            return on_boundary and not rec_boundary(x, on_boundary)

        self.add_bcs([{
            'component': 0,
            'function': (lambda _: 1),
            'bc': rec_boundary,
            'type': 'dirichlet'
        }, {
            'component': 0,
            'function': (lambda _: 0),
            'bc': circ_boundary,
            'type': 'dirichlet'
        }])

        # Training Setting
        self.training_points()  # default


class Poisson2D(baseclass.BasePDE):

    def __init__(self, width=2, height=2, c=0.1, m=1, n=1, k=10):
        super().__init__()
        # output dim
        self.output_dim = 1
        # domain
        self.bbox = [-width / 2, width / 2, -height / 2, height / 2]
        self.geom = dde.geometry.Rectangle(xmin=[-width / 2, -height / 2], xmax=[width / 2, height / 2])

        # PDE
        def pde(x, u):
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            u_yy = dde.grad.hessian(u, x, i=1, j=1)

            def f(xy):
                x, y = xy[:, 0:1], xy[:, 1:2]
                part1 = 4 * torch.pi**2 * m**2 * c * torch.sin(2 * torch.pi * m * x)
                part2 = (2 * k**2 * torch.sinh(k * x)) / (torch.cosh(k * x)**3)
                part3 = 4 * torch.pi**2 * n**2 * (c * torch.sin(2 * torch.pi * m * x) + torch.tanh(k * x))
                return (part1 + part2 + part3) * torch.sin(2 * torch.pi * n * y)

            return [u_xx + u_yy + f(x)]

        self.pde = pde
        self.set_pdeloss(num=1)

        # BCs
        def ref_sol(xy):
            x, y = xy[:, 0:1], xy[:, 1:2]
            return (c * np.sin(2 * np.pi * m * x) + np.tanh(k * x)) * np.sin(2 * np.pi * n * y)

        self.ref_sol = ref_sol

        self.add_bcs([{
            'component': 0,
            'function': ref_sol,
            'bc': (lambda _, on_boundary: on_boundary),
            'type': 'dirichlet',
        }])

        # Training Setting
        self.training_points()  # default


class PoissonBoltzmann2D(baseclass.BasePDE):

    def __init__(self, k=1, mu=(1, 4), A=10, bbox=[-1, 1, -1, 1], circ=[(0.5, 0.5, 0.2), (0.4, -0.4, 0.4), (-0.2, 0.7, 0.1), (-0.6, -0.5, 0.3)]):
        super().__init__()
        # output dim
        self.output_dim = 1
        # geom
        self.bbox = bbox
        self.circ = circ
        geom = dde.geometry.Rectangle(xmin=[bbox[0], bbox[2]], xmax=[bbox[1], bbox[3]])
        for i in range(len(circ)):
            disk = dde.geometry.Disk(circ[i][0:2], circ[i][2])
            geom = dde.geometry.csg.CSGDifference(geom, disk)
        self.geom = geom

        # PDE
        def pde(x, u):
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            u_yy = dde.grad.hessian(u, x, i=1, j=1)

            def f(xy):
                x, y = xy[:, 0:1], xy[:, 1:2]
                return A * (mu[0]**2 + x**2 + mu[1]**2 + y**2) \
                         * torch.sin(mu[0] * torch.pi * x) * torch.sin(mu[1] * torch.pi * y)

            return -u_xx - u_yy + k**2 * u - f(x)

        self.pde = pde
        self.set_pdeloss(num=1)

        self.load_ref_data("ref/poisson_boltzmann2d.dat")

        # Boundary Condition
        def boundary_rec(x, on_boundary):
            return on_boundary and (np.isclose(x[0], bbox[0]) or np.isclose(x[0], bbox[1]) or np.isclose(x[1], bbox[2]) or np.isclose(x[1], bbox[3]))

        def boundary_circle(x, on_boundary):
            return on_boundary and not boundary_rec(x, on_boundary)

        # BCs
        self.add_bcs([{
            'name': 'rec',
            'component': 0,
            'function': (lambda x: 0),
            'bc': boundary_rec,
            'type': 'dirichlet'
        }, {
            'name': 'circ',
            'component': 0,
            'function': (lambda x: 1),
            'bc': boundary_circle,
            'type': 'dirichlet'
        }])
        # Training Config
        self.training_points(domain=4096)


class Poisson3D(baseclass.BasePDE):

    def __init__(
        self,
        bbox=[0, 1, 0, 1, 0, 1],
        interface_z=0.5,
        circ=[(0.4, 0.3, 0.6, 0.2), (0.6, 0.7, 0.6, 0.2), (0.2, 0.8, 0.7, 0.1), (0.6, 0.2, 0.3, 0.1)],
        A=(20, 100),
        m=(1, 10, 5),
        k=(8, 10),
        mu=(1, 1)
    ):
        super().__init__()
        # output dim
        self.output_dim = 1
        # geom
        self.bbox = bbox
        self.circ = circ
        geom = dde.geometry.Hypercube(xmin=self.bbox[0::2], xmax=self.bbox[1::2])
        for i in range(len(circ)):
            sphere = dde.geometry.Sphere(circ[i][0:3], circ[i][3])
            geom = dde.geometry.csg.CSGDifference(geom, sphere)
        self.geom = geom

        # PDE
        def pde(x, u):
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            u_yy = dde.grad.hessian(u, x, i=1, j=1)
            u_zz = dde.grad.hessian(u, x, i=2, j=2)

            def f(xyz):
                x, y, z = xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3]
                xlen2 = x**2 + y**2 + z**2
                part1 = torch.exp(torch.sin(m[0] * x) + torch.sin(m[1] * y) + torch.sin(m[2] * z)) * (xlen2 - 1) / (xlen2 + 1)
                part2 = torch.sin(m[0] * torch.pi * x) + torch.sin(m[1] * torch.pi * y) + torch.sin(m[2] * torch.pi * z)
                return A[0] * part1 + A[1] * part2

            return -torch.where(x[:, 2] < interface_z, mu[0], mu[1]) * (u_xx + u_yy + u_zz) \
                + torch.where(x[:, 2] < interface_z, k[0]**2, k[1]**2) * u - f(x)

        self.pde = pde
        self.set_pdeloss(num=1)

        self.load_ref_data("ref/poisson_3d.dat")

        # BCs
        def boundary_rec(x, on_boundary):
            return on_boundary and (
                np.isclose(x[0], bbox[0]) or np.isclose(x[0], bbox[1]) or np.isclose(x[1], bbox[2]) or np.isclose(x[1], bbox[3])
                or np.isclose(x[2], bbox[4]) or np.isclose(x[2], bbox[5])
            )

        self.add_bcs([{'component': 0, 'function': (lambda x: 0), 'bc': boundary_rec, 'type': 'neumann'}])
        # Training Config
        self.training_points(domain=4096)


class Poisson2DManyArea(baseclass.BasePDE):

    def __init__(self, bbox=[-10, 10, -10, 10], split=(5, 5)):
        super().__init__()
        # output dim
        self.output_dim = 1
        # geom
        self.bbox = bbox
        self.geom = dde.geometry.Rectangle(xmin=[bbox[0], bbox[2]], xmax=[bbox[1], bbox[3]])

        # PDE

        self.a_cof = np.loadtxt("ref/poisson_a_coef.dat")
        self.f_cof = np.loadtxt("ref/poisson_f_coef.dat").reshape(split[0], split[1], 3, 3)
        block_size = np.array([(bbox[1] - bbox[0] + 2e-5) / split[0], (bbox[3] - bbox[2] + 2e-5) / split[1]])

        def domain(x):
            reduced_x = (x - np.array(bbox[::2]) + 1e-5)
            dom = np.floor(reduced_x / block_size).astype("int32")
            return dom, reduced_x - dom * block_size

        def a(x):
            dom, res = domain(x)
            return self.a_cof[dom[0], dom[1]]

        a = np.vectorize(a, signature="(2)->()")

        def f(x):
            dom, res = domain(x)

            def f_fn(coef):
                ans = coef[0, 0]
                for i in range(coef.shape[0]):
                    for j in range(coef.shape[1]):
                        tmp = np.sin(np.pi * np.array((i, j)) * (res / block_size))
                        ans += coef[i, j] * tmp[0] * tmp[1]
                return ans

            return f_fn(self.f_cof[dom[0], dom[1]])

        f = np.vectorize(f, signature="(2)->()")

        @functools.cache
        def get_coef(x):
            x = x.detach().cpu()
            return torch.Tensor(a(x)), torch.Tensor(f(x))

        def pde(x, u):
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            u_yy = dde.grad.hessian(u, x, i=1, j=1)

            a, f = get_coef(x)
            return a * (u_xx + u_yy) + f

        self.pde = pde
        self.set_pdeloss(num=1)

        self.load_ref_data("ref/poisson_manyarea.dat")  # NOTE: this data might have problem

        # BCs
        self.add_bcs([{
            'component': 0,
            'function': (lambda x, y: -y),
            'bc': (lambda _, on_boundary: on_boundary),
            'type': 'robin',
        }])
        # training config
        self.training_points()


class PoissonND(baseclass.BasePDE):

    def __init__(self, dim=5, len=1):
        super().__init__()
        # output dim
        self.output_dim = 1
        # geom
        self.bbox = [0, len] * dim
        self.geom = dde.geometry.Hypercube(xmin=self.bbox[0::2], xmax=self.bbox[1::2])

        # pde
        def pde(x, u):
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            for i in range(1, dim):
                u_xx = u_xx + dde.grad.hessian(u, x, i=i, j=i)

            def f(x):
                return (torch.pi**2) / 4 * torch.sin(torch.pi / 2 * x).sum(axis=1).reshape(-1, 1)

            return [u_xx + f(x)]

        self.pde = pde
        self.set_pdeloss(num=1)

        # bc
        def ref_sol(x):
            return np.sin(np.pi / 2 * x).sum(axis=1).reshape(-1, 1)

        self.ref_sol = ref_sol

        self.add_bcs([{
            'component': 0,
            'function': ref_sol,
            'bc': (lambda _, on_boundary: on_boundary),
            'type': 'dirichlet',
        }])

        # set training config
        self.training_points(domain=4096)
