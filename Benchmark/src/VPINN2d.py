import torch
import functools
import inspect
import regex as re
from tqdm import tqdm
from .Integral2d import quad_integral
from .lengendre import test_func
from .net_class import MLP

class VPINN:
    def __init__(self, layer_sizes, pde, bc, area = [-1, 1, -1, 1], pde2=None, Q=10, grid_num=4,test_fcn_num=5, device='cpu', load=None):
        self.pde = pde
        self.pde2 = pde2
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
        quad_integral.init(Q, device)
        x = quad_integral.XX
        y = quad_integral.YY
        # (x1, y1) stands for the left down point of the rectangle
        # (x2, y2) stands for the right upper point of the rectangle
        x1 = area[0]
        x2 = area[1]
        y1 = area[2]
        y2 = area[3]
        lower_xs = torch.linspace(x1, x2, grid_num + 1)[:-1]
        lower_ys = torch.linspace(y1, y2, grid_num + 1)[:-1]
        xx, yy = torch.meshgrid(lower_xs, lower_ys, indexing='ij')
        x_bias = xx.reshape(-1, 1)
        y_bias = yy.reshape(-1, 1)
        
        regularized_x = (x.reshape(1, -1).requires_grad_(True) + 1) / 2
        regularized_y = (y.reshape(1, -1).requires_grad_(True) + 1) / 2
        
        x_grid_len = (x2 - x1) / grid_num
        y_grid_len = (y2 - y1) / grid_num
        
        xs = regularized_x * x_grid_len + x_bias
        ys = regularized_y * y_grid_len + y_bias
        
        self.grid_xs = xs.reshape(-1, 1)
        self.grid_ys = ys.reshape(-1, 1)
        
        # pass the boundary sample pointf from arguments
        self.boundary_xs = bc[0].requires_grad_(True).to(device)
        self.boundary_ys = bc[1].requires_grad_(True).to(device)
        self.boundary_us = bc[2].requires_grad_(True).to(device)
        
        # define the test functions
        test_func.init(self.test_fcn_num)
        self.test_fcn0 = test_func.test_func(0, x, y)
        self.test_fcn1 = test_func.test_func(1, x, y)
        
        # check whether the laplace function is used
        source_code = inspect.getsource(pde)
        
         
        laplace_term_pattern = r'\bLAPLACE_TERM\(((?:[^()]|\((?1)\))*)\)'
        
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

    def lhsWrapper(self, x=None, y=None, u_in=None):
        u = self.net(torch.cat([self.grid_xs, self.grid_ys], dim=1))
        
        lhs = self.pde(self.grid_xs, self.grid_ys, u)
        
        result = torch.einsum('mc,nc->mnc', \
            lhs.view(self.grid_num ** 2, self.Q ** 2), self.test_fcn0.view(self.test_fcn_num ** 2, self.Q ** 2))
        result = torch.reshape(result, (-1, self.Q ** 2))
        return result
    
    def lhsWrapper2(self, x=None, y=None, u_in=None):
        u = self.net(torch.cat([self.grid_xs, self.grid_ys], dim=1))
        
        lhs = self.pde2(self.grid_xs, self.grid_ys, u)
        
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
            if self.pde2:
                int1 = quad_integral.integral(self.lhsWrapper) * ((1 / self.grid_num) ** 2)
                int2 = quad_integral.integral(self.lhsWrapper2) * ((1 / self.grid_num) ** 2)
            else:
                int1 = quad_integral.integral(self.lhsWrapper) * ((1 / self.grid_num) ** 2)
        else:
            laplace_conponent = self.pde1(None, None, None) 
            rest = quad_integral.integral(self.lhsWrapper)
            int1 = (laplace_conponent + rest) * ((1 / self.grid_num) ** 2)
        
        ref = torch.zeros_like(int1).requires_grad_(True)
        
        if self.pde2:
            return self.loss(int1, ref) + self.loss(int2, ref)
        else:
            return self.loss(int1, ref)
    
    
    def train(self, model_name, epoch_num=10000, coef=10):
        optimizer = torch.optim.Adam(params=self.net.parameters())
            
        for i in tqdm(range(epoch_num)):
            optimizer.zero_grad()
            loss = self.loss_interior() + coef * self.loss_bc()
            
            if i % 100 == 0:
                print(f'loss_interior={self.loss_interior().item():.5g}, loss_bc={self.loss_bc().item():.5g}, coef={coef}')
            loss.backward(retain_graph=True)
            optimizer.step()
        
        if model_name:
            path = (f'./model/{model_name}{self.layer_sizes},Q={self.Q},grid_num={self.grid_num}'
                    f',test_fcn={self.test_fcn_num},epoch={epoch_num}).pth')
            torch.save(self.net, path)
        return self.net
        