import torch
from Integral2d import quad_integral
from lengendre import test_func
from tqdm import tqdm
from net_class import MLP

class VPINN:
    # def grid(self):
    #     eps = torch.rand(1).item() * 0.1
    #     x = (torch.linspace(-1, 1 - eps, n) + eps / 2).to(self.device)
    #     y = (torch.linspace(-1, 1, n) + eps / 2).to(self.device)
    #     xx, yy = torch.meshgrid(x, y, indexing='ij')
    #     x = xx.reshape(-1, 1)
    #     y = yy.reshape(-1, 1)
    #     return x.requires_grad_(True), y.requires_grad_(True)

    # (x1,y1)左下,(x2, y2)右上
    def __init__(self, type, f, u, Q, grid_num, boundary_num, test_fcn_num, device):
        self.type =type
        self.f = f
        self.u = u
        self.Q = Q
        self.grid_num = grid_num
        self.boundary_num = boundary_num
        self.test_fcn_num = test_fcn_num
        self.loss = torch.nn.MSELoss()
        self.device = device
        self.net = MLP().to(device)
        
        xs = []
        ys = []
        quad_integral.init(Q)
        x = quad_integral.XX
        y = quad_integral.YY
        for index in range(self.grid_num ** 2):
            x1, y1, x2, y2 = self.index2frame(index, self.grid_num)
            xx = (x.reshape(-1, 1).requires_grad_(True) + 1) / self.grid_num + x1
            yy = (y.reshape(-1, 1).requires_grad_(True) + 1) / self.grid_num + y1
            xs.append(xx)
            ys.append(yy)
        xs = torch.cat(xs, dim=0).view(-1, 1)
        ys = torch.cat(ys, dim=0).view(-1, 1)
        self.grid_xs = xs
        self.grid_ys = ys
        
        xs = []
        ys = []
        for index in range(self.grid_num ** 2):
            x1, y1, x2, y2 = self.index2frame(index, self.grid_num)
            x_r = torch.linspace(x2, x2, self.boundary_num).reshape(-1, 1).to(self.device).requires_grad_(True)
            y_r = torch.linspace(y1, y2, self.boundary_num).reshape(-1, 1).to(self.device).requires_grad_(True)
        
            x_u = torch.linspace(x1, x2, self.boundary_num).reshape(-1, 1).to(self.device).requires_grad_(True)
            y_u = torch.linspace(y2, y2, self.boundary_num).reshape(-1, 1).to(self.device).requires_grad_(True)
            
            x_l = torch.linspace(x1, x1, self.boundary_num).reshape(-1, 1).to(self.device).requires_grad_(True)
            y_l = torch.linspace(y1, y2, self.boundary_num).reshape(-1, 1).to(self.device).requires_grad_(True)
            
            x_d = torch.linspace(x1, x2, self.boundary_num).reshape(-1, 1).to(self.device).requires_grad_(True)
            y_d = torch.linspace(y1, y1, self.boundary_num).reshape(-1, 1).to(self.device).requires_grad_(True)
            
            xs.extend([x_r, x_u, x_l, x_d])
            ys.extend([y_r, y_u, y_l, y_d])
        self.boundary_xs = torch.cat(xs, dim=0)
        self.boundary_ys = torch.cat(ys, dim=0)
        
    def index2frame(self, index, grid_num):
        i = index // grid_num
        j = index % grid_num
        grid_len = 2 / grid_num
        x1 = -1 + i * grid_len
        y1 = -1 + j * grid_len
        x2 = -1 + (i + 1) * grid_len
        y2 = -1 + (j + 1) * grid_len
        return x1, y1, x2, y2
    
    def gradients(self, u, x, order=1):
        if order == 1:
            return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True, )[0]
        else:
            return self.gradients(self.gradients(u, x), x, order=order - 1)
        
    # def bc(self):
    #     xs = []
    #     ys = []
    #     for index in range(self.grid_num ** 2):
    #         x1, y1, x2, y2 = self.index2frame(index, self.grid_num)
    #         x_r = torch.linspace(x2, x2, self.boundary_num).reshape(-1, 1).to(self.device).requires_grad_(True)
    #         y_r = torch.linspace(y1, y2, self.boundary_num).reshape(-1, 1).to(self.device).requires_grad_(True)
        
    #         x_u = torch.linspace(x1, x2, self.boundary_num).reshape(-1, 1).to(self.device).requires_grad_(True)
    #         y_u = torch.linspace(y2, y2, self.boundary_num).reshape(-1, 1).to(self.device).requires_grad_(True)
            
    #         x_l = torch.linspace(x1, x1, self.boundary_num).reshape(-1, 1).to(self.device).requires_grad_(True)
    #         y_l = torch.linspace(y1, y2, self.boundary_num).reshape(-1, 1).to(self.device).requires_grad_(True)
            
    #         x_d = torch.linspace(x1, x2, self.boundary_num).reshape(-1, 1).to(self.device).requires_grad_(True)
    #         y_d = torch.linspace(y1, y1, self.boundary_num).reshape(-1, 1).to(self.device).requires_grad_(True)
            
    #         xs.extend([x_r, x_u, x_l, x_d])
    #         ys.extend([y_r, y_u, y_l, y_d])
    #     x = torch.cat(xs, dim=0)
    #     y = torch.cat(ys, dim=0)
    #     return x, y

    def LaplaceWrapper(self, x, y):
        u = self.net(torch.cat([self.grid_xs, self.grid_ys], dim=1))
        d2x = self.gradients(u, self.grid_xs, 2)
        d2y = self.gradients(u, self.grid_ys, 2)
        laplace_u = d2x + d2y
        result = laplace_u.view(self.grid_num ** 2, self.Q ** 2) \
                * (test_func.test_func(0, x, y).squeeze(-1).unsqueeze(0)).expand(self.grid_num ** 2, self.Q ** 2)
        return result
    
    def fWrapper(self, x, y):
        # print(x)
        # print(xx)
        f = self.f(self.grid_xs, self.grid_ys)
        # print(f.T)
        result = f.view(self.grid_num ** 2, self.Q ** 2) * test_func.test_func(0, x, y).squeeze(-1).unsqueeze(0).expand(self.grid_num ** 2, self.Q ** 2)
        return result
    
    def loss_bc(self):
        prediction = self.net(torch.cat([self.boundary_xs, self.boundary_ys], dim=1))
        solution = self.u(self.boundary_xs, self.boundary_ys)
        return self.loss(prediction, solution)
        
    def loss_interior_1(self):
        int1 = quad_integral.integral(self.LaplaceWrapper) * ((1 / self.grid_num) ** 2)
        int2 = quad_integral.integral(self.fWrapper) * ((1 / self.grid_num) ** 2)
        # int3 = quad_integral.integral(ext_LaplaceWrapper, k1, k2, index) * ((1 /grid_num) ** 2)
        # err1 = torch.abs(f - laplace_u)
        # err2 = torch.abs(f - ext_laplace_u)
        # if k1 - 1 < test_num:
            # loss1[k1-1].append(loss(int1, int2).item())
        
        # print(loss(int1, int2))
        return self.loss(int1, int2)
    
    def train(self, epoch_num=10000, coef=10):
        optimizer = torch.optim.Adam(params=self.net.parameters(), lr=1e-3)
        test_func.init(self.test_fcn_num)
        
        if self.type == 0:
            for i in tqdm(range(epoch_num)):
                optimizer.zero_grad()
                loss = self.loss_interior_1() + coef * self.loss_bc()
                if i % 100 == 0:
                    print(loss)
                loss.backward()
                optimizer.step()
            
        