import torch
import net_class
import sys
import inspect
import re
from tqdm import tqdm
from Utilities.Integral2d import quad_integral
from Utilities.lengendre import test_func
from net_class import MLP


class VPINN:
    def __init__(self, layer_sizes, pde, bc, transformer=None, Q=10, grid_num=6, test_fcn_num=5, device='cpu', load=None):
        self.pde = pde
        self.bc = bc
        self.Q = Q
        self.grid_num = grid_num
        self.test_fcn_num = test_fcn_num
        self.loss = torch.nn.MSELoss()
        self.device = device
        self.layer_sizes = layer_sizes
        self.load = load
        if load:
            # self.net = MLP(layer_sizes).to(device)
            self.net = torch.load('./model/'+load).to(device)
        else:
            self.net = MLP(layer_sizes).to(device)
            
        # define the grid sample points
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
        
        # pass the boundary sample pointf from arguments
        self.boundary_xs = bc[0].requires_grad_(True)
        self.boundary_ys = bc[1].requires_grad_(True)
        self.boundary_us = bc[2].requires_grad_(True)
        
        # define the test functions
        test_func.init(self.test_fcn_num)
        self.test_fcn0 = test_func.test_func(0, x, y)
        self.test_fcn1 = test_func.test_func(1, x, y)
        
        # check whether the laplace function is used
        source_code = inspect.getsource(pde)
        self.calls_laplace = bool(re.search(r'\bVPINN.laplace\b', source_code))
        if(self.calls_laplace):
            if(transformer is None):
                print('When laplace is used, you are expected to pass the argument of tranformer to show how the laplace conponent is tranformed')
                sys.exit()
        
        # if calls_laplace:
        #     # pde1仅含laplace项
        #     self.pde1 = self.replace_laplace_with_self_and_dummy(pde, True)
        #     # pde2含除laplace之外的其他项
        #     self.pde2 = self.replace_laplace_with_self_and_dummy(pde, False)
        #     self.pde = pde
        # else:
        #     self.pde = pde
        #     self.pde1 = None
        #     self.pde2 = None

    # def replace_laplace_with_self_and_dummy(self, func, is_pde1):
    #     @functools.wraps(func)
    #     def wrapper(*args, **kwargs):
    #         original_laplace = VPINN.laplace
    #         VPINN.laplace = self.Laplace
    #         result_with_self = func(*args, **kwargs)
    #         VPINN.laplace = lambda x: 0
    #         result_with_dummy = func(*args, **kwargs)
    #         VPINN.laplace = original_laplace

    #         return result_with_self - result_with_dummy if is_pde1 else result_with_dummy
    #     return wrapper

    def index2frame(self, index, grid_num):
        i = index // grid_num
        j = index % grid_num
        grid_len = 2 / grid_num
        x1 = -1 + i * grid_len
        y1 = -1 + j * grid_len
        x2 = -1 + (i + 1) * grid_len
        y2 = -1 + (j + 1) * grid_len
        return x1, y1, x2, y2
    
    # just serve as a placeholder
    @staticmethod
    def laplace(x, y, u):
        return torch.zeros_like(x)
    
    @staticmethod
    def gradients(u, x, order=1):
        if order == 1:
            return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True, )[0]
        else:
            return VPINN.gradients(VPINN.gradients(u, x), x, order=order - 1)
    
    def Laplace(self):
        u = self.net(torch.cat([self.grid_xs, self.grid_ys], dim=1))
        dx = VPINN.gradients(u, self.grid_xs, 1)
        dy = VPINN.gradients(u, self.grid_ys, 1)
        
        du = torch.cat([dx, dy], dim=1) * self.grid_num
        du = du.view(self.grid_num ** 2, self.Q ** 2, 2)
        dv = self.test_fcn1.view(self.test_fcn_num ** 2, self.Q ** 2, 2)
        
        du = du.unsqueeze(1).expand(-1, self.test_fcn_num ** 2, -1, -1)
        dv = dv.unsqueeze(0).expand(self.grid_num ** 2, -1, -1, -1)
        
        result = torch.sum(du * dv, dim=-1)
        result = result.view(-1, self.Q ** 2)
        # result = torch.bmm(du.view(-1, 1, self.Q ** 2, 2), dv.view(1, -1, self.Q ** 2, 2)).view(-1, self.Q ** 2, 2)
        # result = (du * dv).sum(dim=-1)
        return result

    def lhsWrapper(self, x=None, y=None, u_in=None):
        u = self.net(torch.cat([self.grid_xs, self.grid_ys], dim=1))
        
        # if self.calls_laplace == False:
        lhs = self.pde(self.grid_xs, self.grid_ys, u)
        # else:
            # lhs = self.pde2(self.grid_xs, self.grid_ys, u)
        
        result = torch.einsum('mc,nc->mnc', \
            lhs.view(self.grid_num ** 2, self.Q ** 2), self.test_fcn0.view(self.test_fcn_num ** 2, self.Q ** 2))
        result = torch.reshape(result, (-1, self.Q ** 2))
        return result
    
    def loss_bc(self):
        prediction = self.net(torch.cat([self.boundary_xs, self.boundary_ys], dim=1))
        solution = self.boundary_us
        return self.loss(prediction, solution)
        
    def loss_interior(self):
        if self.calls_laplace == False:
            int1 = quad_integral.integral(self.lhsWrapper) * ((1 / self.grid_num) ** 2)
        else:
            laplace_conponent = quad_integral.integral(self.Laplace) * ((1 / self.grid_num) ** 2)
            rest = quad_integral.integral(self.lhsWrapper) * ((1 / self.grid_num) ** 2)
            int1 = laplace_conponent + rest
        
        int2 = torch.zeros_like(int1).requires_grad_(True)
        
        return self.loss(int1, int2)
    
    # def loss_interior_2(self):
    #     int1 = -quad_integral.integral(lambda x, y: -self.DeltaWrapper(x, y) + 64 * self.uWrapper(x, y)) * ((1 / self.grid_num) ** 2)
    #     int2 = quad_integral.integral(self.fWrapper) * ((1 / self.grid_num) ** 2)
    #     # int3 = quad_integral.integral(self.LaplaceWrapper) * ((1 / self.grid_num) ** 2)
    #     return self.loss(int1, int2)
    
    def train(self, model_name, epoch_num=10000, coef=10):
        optimizer = torch.optim.Adam(params=self.net.parameters())
        
        
        for i in tqdm(range(epoch_num)):
            optimizer.zero_grad()
            loss = self.loss_interior() + coef * self.loss_bc()
            if i % 100 == 0:
                print(f'loss_interior={self.loss_interior().item():.5g}, loss_bc={self.loss_bc().item():.5g}')
            loss.backward(retain_graph=True)
            optimizer.step()
        
        if model_name:
            torch.save(self.net, './model/'+model_name+f'{self.layer_sizes}'+f',Q={self.Q}'+f',grid_num={self.grid_num}'+\
                f',test_fcn={self.test_fcn_num}'+f',load={self.load}'+f',epoch={epoch_num})'+'.pth')
        return self.net
        

    # def LaplaceWrapper(self):
    #     u = self.net(torch.cat([self.grid_xs, self.grid_ys], dim=1))
    #     d2x = VPINN.gradients(u, self.grid_xs, 2)
    #     d2y = VPINN.gradients(u, self.grid_ys, 2)
    #     laplace_u = d2x + d2y
        
    #     result = torch.einsum('mc,nc->mnc', \
    #         laplace_u.view(self.grid_num ** 2, self.Q ** 2), self.test_fcn0.view(self.test_fcn_num ** 2, self.Q ** 2))
        
    #     result = torch.reshape(result, (-1, self.Q ** 2))
    #     # result = laplace_u.view(self.grid_num ** 2, self.Q ** 2) \
    #     #         * test_func.test_func(0, x, y)
    #     return result
    
    # def fWrapper(self):
    #     f = self.f(self.grid_xs, self.grid_ys)
    #     result = torch.einsum('mc,nc->mnc', \
    #         f.view(self.grid_num ** 2, self.Q ** 2), self.test_fcn0.view(self.test_fcn_num ** 2, self.Q ** 2))
    #     result = torch.reshape(result, (-1, self.Q ** 2))
    #     # result = f.view(self.grid_num ** 2, self.Q ** 2) * test_func.test_func(0, x, y)
    #     return result
    
    # def uWrapper(self):
    #     u = self.net(torch.cat([self.grid_xs, self.grid_ys], dim=1))
    #     result = torch.einsum('mc,nc->mnc', \
    #         u.view(self.grid_num ** 2, self.Q ** 2), self.test_fcn0.view(self.test_fcn_num ** 2, self.Q ** 2))
    #     result = torch.reshape(result, (-1, self.Q ** 2))
    #     return result
    
    # def gradient_u(self, x, y):
    #     du_dx = torch.tensor(0.5 * torch.pi * torch.cos(0.5 * torch.pi * (x + 1)) * torch.sin(0.5 * torch.pi * (y + 1)))
    #     du_dy = torch.tensor(torch.sin(0.5 * torch.pi * (x + 1)) * 0.5 * torch.pi * torch.cos(0.5 * torch.pi * (y + 1)))
    #     return du_dx, du_dy
    
            # if self.type == 1:
        #     for i in tqdm(range(epoch_num)):
        #         optimizer.zero_grad()
        #         loss = self.loss_interior_2() + coef * self.loss_bc()
        #         if i % 100 == 0:
        #             print(f'loss_interior={self.loss_interior_2().item():.5g}, loss_bc={self.loss_bc().item():.5g}')
        #         loss.backward(retain_graph=True)
        #         optimizer.step()