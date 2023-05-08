import torch
import functools
import inspect
import regex as re
from tqdm import tqdm
from .Integral import quad_integral3d
from .lengendre import test_func
from .net_class import MLP


class VPINN:
    def __init__(self, layer_sizes, pde, bc1=None, bc2=None, area = [-1, 1, -1, 1, -1, 1], Q=10, grid_num=4,test_fcn_num=5, device='cpu', load=None):
        self.pde = pde
        self.bc1 = bc1
        self.bc2 = bc2
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
        quad_integral3d.init(Q, device)
        x = quad_integral3d.XX
        y = quad_integral3d.YY
        z = quad_integral3d.ZZ
        # (x1, y1) stands for the left down point of the rectangle
        # (x2, y2) stands for the right upper point of the rectangle
        x1 = area[0]
        x2 = area[1]
        y1 = area[2]
        y2 = area[3]
        z1 = area[4]
        z2 = area[5]
        lower_xs = torch.linspace(x1, x2, grid_num + 1)[:-1]
        lower_ys = torch.linspace(y1, y2, grid_num + 1)[:-1]
        lower_zs = torch.linspace(z1, z2, grid_num + 1)[:-1]
        xx, yy, zz = torch.meshgrid(lower_xs, lower_ys, lower_zs,indexing='ij')
        x_bias = xx.reshape(-1, 1).to(device)
        y_bias = yy.reshape(-1, 1).to(device)
        z_bias = zz.reshape(-1, 1).to(device)
        
        regularized_x = (x.reshape(1, -1).requires_grad_(True) + 1) / 2
        regularized_y = (y.reshape(1, -1).requires_grad_(True) + 1) / 2
        regularized_z = (z.reshape(1, -1).requires_grad_(True) + 1) / 2
        
        x_grid_len = (x2 - x1) / grid_num
        y_grid_len = (y2 - y1) / grid_num
        z_grid_len = (z2 - z1) / grid_num
        
        xs = regularized_x * x_grid_len + x_bias
        ys = regularized_y * y_grid_len + y_bias
        zs = regularized_z * z_grid_len + z_bias
        
        self.grid_xs = xs.reshape(-1, 1).to(device)
        self.grid_ys = ys.reshape(-1, 1).to(device)
        self.grid_zs = zs.reshape(-1, 1).to(device)
        
        # pass the boundary sample pointf from arguments
        self.bc1_xs = bc1[0].requires_grad_(True).to(device).reshape(-1,1)
        self.bc1_ys = bc1[1].requires_grad_(True).to(device).reshape(-1,1)
        self.bc1_zs = bc1[2].requires_grad_(True).to(device).reshape(-1,1)
        self.bc1_us = bc1[3].requires_grad_(True).to(device).reshape(-1,1)
        
        if bc2:
            self.bc2_xs = bc2[0].requires_grad_(True).to(device).reshape(-1,1)
            self.bc2_ys = bc2[1].requires_grad_(True).to(device).reshape(-1,1)
            self.bc2_zs = bc2[2].requires_grad_(True).to(device).reshape(-1,1)
            self.bc2_us = bc2[3].requires_grad_(True).to(device).reshape(-1,1)
            self.bc2_operation = bc2[4]
        
        # define the test functions
        test_func.init(self.test_fcn_num)
        self.test_fcn0 = test_func.test_func(0, x, y, z)
        # self.test_fcn1 = test_func.test_func(1, x, y, z)
        
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
            return eval(laplace_term.replace('VPINN.laplace(x, y, u)', 'quad_integral3d.integral(self.Laplace)'), globals(), {'self': self})
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
        return -result

    def lhsWrapper(self):
        u = self.net(torch.cat([self.grid_xs, self.grid_ys, self.grid_zs], dim=1))
        
        lhs = self.pde(self.grid_xs, self.grid_ys, self.grid_zs, u)
        
        result = torch.einsum('mc,nc->mnc', \
            lhs.view(self.grid_num ** 3, self.Q ** 3), self.test_fcn0.view(self.test_fcn_num ** 3, self.Q ** 3))
        result = result.reshape(-1, self.Q ** 3)
        return result
    
    def lhsWrapper2(self):
        u = self.net(torch.cat([self.grid_xs, self.grid_ys, self.grid_zs], dim=1))
        
        lhs = self.pde2(self.grid_xs, self.grid_ys, self.grid_zs, u)
        
        result = torch.einsum('mc,nc->mnc', \
            lhs.view(self.grid_num ** 3, self.Q ** 3), self.test_fcn0.view(self.test_fcn_num ** 3, self.Q ** 3))
        result = torch.reshape(result, (-1, self.Q ** 3))
        return result
    
    def loss_bc1(self):
        prediction = self.net(torch.cat([self.bc1_xs, self.bc1_ys, self.bc1_zs], dim=1))
        solution = self.bc1_us
        return self.loss(prediction, solution)
    
    def loss_bc2(self):
        u = self.net(torch.cat([self.bc2_xs, self.bc2_ys, self.bc2_zs], dim=1))
        prediction = self.bc2_operation(self.bc2_xs, self.bc2_ys, self.bc2_zs,u)
        solution = self.bc2_us
        return self.loss(prediction, solution)
        # return torch.median(torch.abs(prediction - solution))
    
    def loss_interior(self):
        if self.calls_laplace == False:
            int1 = quad_integral3d.integral(self.lhsWrapper) * ((1 / self.grid_num) ** 2)
        else:
            laplace_conponent = self.pde1(None, None, None) 
            rest = quad_integral3d.integral(self.lhsWrapper)
            int1 = (laplace_conponent + rest) * ((1 / self.grid_num) ** 2)
        
        int2 = torch.zeros_like(int1).requires_grad_(True)
        
        return self.loss(int1, int2)
        # return torch.median(torch.abs(int1 - int2))
    
    
    def train(self, model_name, epoch_num=10000, coef=10):
        optimizer = torch.optim.Adam(params=self.net.parameters(), lr=1e-3)
            
        for i in tqdm(range(epoch_num)):
            optimizer.zero_grad()
            if self.bc2:
                loss = self.loss_interior() + coef * self.loss_bc1() + self.loss_bc2()
            else:
                loss = self.loss_interior() + coef * self.loss_bc1()
            
            if i % 1000 == 0:
                if self.bc2:
                    print(f'loss_interior={self.loss_interior().item():.5g}, loss_bc1={self.loss_bc1().item():.5g}, loss_bc2={self.loss_bc2().item():.5g}')
                else:
                    print(f'loss_interior={self.loss_interior().item():.5g}, loss_bc={self.loss_bc1().item():.5g}')
            loss.backward(retain_graph=True)
            optimizer.step()
        
        if model_name and epoch_num != 0:
            path = (f'./model/{model_name}{self.layer_sizes},Q={self.Q},grid_num={self.grid_num}'
                    f',test_fcn={self.test_fcn_num},epoch={epoch_num}).pth')
            torch.save(self.net, path)
        return self.net
        
