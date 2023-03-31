from GaussJacobiQuadRule_V3 import GaussLobattoJacobiWeights
import basis
import torch

Q = 10
a, b = 0, 0
[X, W] = GaussLobattoJacobiWeights(Q, a, b)

def u(x, y):
    return x**2 + y**2

X, Wx = torch.tensor(X), torch.tensor(W)
Y, Wy = X, Wx
# print(X)
# print(Wx)
XX, YY = torch.meshgrid(X, Y, indexing='ij')
Wxx, Wyy = torch.meshgrid(Wx, Wy, indexing='ij')
integral = torch.sum(u(XX, YY) * (Wxx * Wyy))
# integral = 0
# for i in range(Q):
#     for j in range(Q):
#         x_i, y_j = X[i], X[j]
#         w_i, w_j = W[i], W[j]
#         integral += u(x_i, y_j) * w_i * w_j

print(integral)

