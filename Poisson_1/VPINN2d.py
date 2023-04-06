import torch
import net_class
from tqdm import tqdm
from Utilities.Integral2d import quad_integral
from Utilities.lengendre import test_func
from net_class import MLP

class VPINN:
    def __init__(self, layer_sizes, f, u=None, type=0, Q=10, grid_num=8, boundary_num=80, test_fcn_num=5, device='cpu', load=None):
        self.type =type
        self.f = f
        self.u = u
        self.Q = Q
        self.grid_num = grid_num
        self.boundary_num = boundary_num
        self.test_fcn_num = test_fcn_num
        self.loss = torch.nn.MSELoss()
        self.device = device
        self.layer_sizes = layer_sizes
        if load:
            self.net = MLP(layer_sizes).to(device)
            self.net = torch.load('./model/'+load).to(device)
        else:
            self.net = MLP(layer_sizes).to(device)
        xs = []
        ys = []
        quad_integral.init(Q,device)
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
    
    # def gradient_u(self, x, y):
    #     du_dx = torch.tensor(0.5 * torch.pi * torch.cos(0.5 * torch.pi * (x + 1)) * torch.sin(0.5 * torch.pi * (y + 1)))
    #     du_dy = torch.tensor(torch.sin(0.5 * torch.pi * (x + 1)) * 0.5 * torch.pi * torch.cos(0.5 * torch.pi * (y + 1)))
    #     return du_dx, du_dy
    
    def DeltaWrapper(self, x, y):
        u = self.net(torch.cat([self.grid_xs, self.grid_ys], dim=1))
        dx = self.gradients(u, self.grid_xs, 1)
        dy = self.gradients(u, self.grid_ys, 1)
        # ext_dx , ext_dy = self.gradient_u(self.grid_xs, self.grid_ys)
        du = torch.cat([dx, dy], dim=1) * self.grid_num
        
        du = du.view(self.grid_num ** 2, self.Q ** 2, 2)
        dv = (test_func.test_func(1, x, y)).view(1, self.Q ** 2, 2)
        result = (du * dv).sum(dim=-1)
        return result.view(-1, self.Q ** 2)

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
    
    def loss_interior_2(self):
        int1 = torch.abs(quad_integral.integral(self.DeltaWrapper)) * ((1 / self.grid_num) ** 2)
        int2 = torch.abs(quad_integral.integral(self.fWrapper)) * ((1 / self.grid_num) ** 2)
        return self.loss(int1, int2)
    
    def train(self, model_name, epoch_num=10000, coef=10):
        optimizer = torch.optim.Adam(params=self.net.parameters())
        test_func.init(self.test_fcn_num)
        
        if self.type == 0:
            for i in tqdm(range(epoch_num)):
                optimizer.zero_grad()
                loss = self.loss_interior_1() + coef * self.loss_bc()
                if i % 100 == 0:
                    print(f'loss={loss.item():.5g}')
                loss.backward(retain_graph=True)
                optimizer.step()
        
        if self.type == 1:
            for i in tqdm(range(epoch_num)):
                optimizer.zero_grad()
                loss = self.loss_interior_2() + coef * self.loss_bc()
                if i % 100 == 0:
                    print(f'loss={loss.item():.5g}')
                loss.backward(retain_graph=True)
                optimizer.step()
        
        torch.save(self.net, './model/'+model_name+f'{self.layer_sizes}'+f'(type={self.type}'+f',Q={self.Q}'+f',grid_num={self.grid_num}'+\
            f',boundary_num={self.boundary_num}'+f',test_fcn={self.test_fcn_num})'+'.pth')
        return self.net
        