import torch
import functools
import inspect
import regex as re
from tqdm import tqdm
from .Integral2d import quad_integral
from .lengendre import test_func
from .net_class import MLP


class VPINN:
    def __init__(self, layer_sizes, pde, bc, Q=10, grid_num=6, test_fcn_num=5, device='cpu', load=None):
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
            x1, y1, x2, y2 = self.__index2frame(index, self.grid_num)
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
        
         
        laplace_term_pattern = r'\bLAPLACE_TERM\(((?:[^()]|\((?1)\))*)\)'
        # r'LAPLACE_TERM\(([^"]*)\)'
        
        laplace_term = re.search(laplace_term_pattern, source_code)
        calls_laplace = bool(laplace_term)
        self.calls_laplace = calls_laplace

        if calls_laplace:
            self.pde1 = self.__extract_laplace_term(pde, laplace_term.group(1).strip())
            self.pde = pde
        else:
            self.pde = pde
            self.pde1 = None

    def __extract_laplace_term(self, func, laplace_term):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return eval(laplace_term.replace('VPINN.laplace(x, y, u)', 'quad_integral.integral(self.Laplace)'), globals(), {'self': self})
        return wrapper

    def __index2frame(self, index, grid_num):
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
    def LAPLACE_TERM(term):
        return torch.zeros_like(term)
    
    @staticmethod
    def gradients(u, x, order=1):
        if order == 1:
            return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True, )[0]
        else:
            return VPINN.gradients(VPINN.gradients(u, x), x, order=order - 1)
    
    def Laplace(self, x_=None, y_=None, u_=None):
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
        return -result

    def __lhsWrapper(self, x=None, y=None, u_in=None):
        u = self.net(torch.cat([self.grid_xs, self.grid_ys], dim=1))
        
        lhs = self.pde(self.grid_xs, self.grid_ys, u)
        
        result = torch.einsum('mc,nc->mnc', \
            lhs.view(self.grid_num ** 2, self.Q ** 2), self.test_fcn0.view(self.test_fcn_num ** 2, self.Q ** 2))
        result = torch.reshape(result, (-1, self.Q ** 2))
        return result
    
    def __loss_bc(self):
        prediction = self.net(torch.cat([self.boundary_xs, self.boundary_ys], dim=1))
        solution = self.boundary_us
        return self.loss(prediction, solution)
        
    def __loss_interior(self):
        if self.calls_laplace == False:
            int1 = quad_integral.integral(self.__lhsWrapper) * ((1 / self.grid_num) ** 2)
        else:
            laplace_conponent = self.pde1(None, None, None) 
            rest = quad_integral.integral(self.__lhsWrapper)
            int1 = (laplace_conponent + rest) * ((1 / self.grid_num) ** 2)
        
        int2 = torch.zeros_like(int1).requires_grad_(True)
        
        return self.loss(int1, int2)
    
    def train(self, model_name, epoch_num=10000, coef=10):
        optimizer = torch.optim.Adam(params=self.net.parameters())
        
        for i in tqdm(range(epoch_num)):
            optimizer.zero_grad()
            loss = self.__loss_interior() + coef * self.__loss_bc()
            if i % 100 == 0:
                print(f'loss_interior={self.__loss_interior().item():.5g}, loss_bc={self.__loss_bc().item():.5g}')
            loss.backward(retain_graph=True)
            optimizer.step()
        
        if model_name:
            torch.save(self.net, './model/'+model_name+f'{self.layer_sizes}'+f',Q={self.Q}'+f',grid_num={self.grid_num}'+\
                f',test_fcn={self.test_fcn_num}'+f',load={self.load}'+f',epoch={epoch_num})'+'.pth')
        return self.net
        