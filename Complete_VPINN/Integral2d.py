from GaussJacobiQuadRule_V3 import GaussLobattoJacobiWeights
import basis
import torch
import lengendre
from net_class import MLP
import os, sys

# os.chdir(sys.path[0])
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = torch.device("mps") if torch.has_mps else "cpu"
class Quad_Integral:
    
    def init(self):
        Q = 10
        a, b = 0, 0
        [X, W] = GaussLobattoJacobiWeights(Q, a, b)

        X, Wx = torch.tensor(X,dtype=torch.float32), torch.tensor(W, dtype=torch.float32)
        Y, Wy = X, Wx
        self.XX, self.YY = torch.meshgrid(X, Y, indexing='ij')
        self.Wxx, self.Wyy = torch.meshgrid(Wx, Wy, indexing='ij')
        self.XX = self.XX.reshape(-1, 1).to(device)
        self.YY = self.YY.reshape(-1, 1).to(device)
        self.Wxx = self.Wxx.reshape(-1, 1).to(device)
        self.Wyy = self.Wyy.reshape(-1, 1).to(device)
    
    def integral(self, u, k1, k2, index):
        integral = torch.sum(u(self.XX, self.YY, index, k1, k2) * (self.Wxx * self.Wyy).squeeze(-1).unsqueeze(0).expand(-1, 100))
        return integral

quad_integral = Quad_Integral()
# net = torch.load('ordinary.pth')
# def wrapper1(x, y):
#     return basis.f(x, y) * lengendre.v(0, 2, x, y)

# def wrapper2(x, y):
#     return torch.sum(basis.gradient_u(x, y) * lengendre.v(1, 2, x, y),dim=1, keepdim=True)

# def wrapper3(x, y):
    # xx = torch.tensor(x).reshape(-1, 1).requires_grad_(True)
    # yy = torch.tensor(y).reshape(-1, 1).requires_grad_(True)
    # u = net(torch.cat([xx, yy], dim=1))
    # dx = basis.gradients(u, xx, 1)
    # dy = basis.gradients(u, yy, 1)
    # du = torch.cat([dx, dy], dim=1)
    # return torch.sum(du * lengendre.v(1, 2, x, y),dim=1, keepdim=True)
    
# print(quad_integral(wrapper1),quad_integral(wrapper2),quad_integral(wrapper3))