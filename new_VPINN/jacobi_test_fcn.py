from GaussJacobiQuadRule_V3 import Jacobi
import torch
from lengendre import v
from lengendre import legendre
import matplotlib.pyplot as plt

def Test_fcn_x(n,x):
       test  = Jacobi(n+1,0,0,x) - Jacobi(n-1,0,0,x)
       return test
   
x = torch.linspace(-1, 1, 100)

n = 9
err = legendre(n + 1, x) - legendre(n - 1, x) - Test_fcn_x(n, x)
plt.plot(x, err)
plt.show()