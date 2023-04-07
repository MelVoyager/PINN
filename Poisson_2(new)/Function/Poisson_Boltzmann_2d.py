import torch

sin = torch.sin
cos = torch.cos
pi = torch.pi

def sample_points_on_circle(x, y, r, n):
    angles = torch.linspace(0, 2 * torch.pi, n)  # 在0到2π之间等间隔采样n个角度值
    x_coords = x + r * torch.cos(angles)  # 计算x坐标
    y_coords = y + r * torch.sin(angles)  # 计算y坐标

    return x_coords, y_coords


def f(x, y):
    return 10 * (17 + x ** 2 + y ** 2) * sin(pi * x) * sin(4 * pi * y)


def bc(boundary_num, device='cpu'):
    xs = []
    ys = []
    us = []
    
    # sample on the circle
    circles = [(0.5, 0.5, 0.2), (0.4, -0.4, 0.4), (-0.2, -0.7, 0.1), (-0.6, 0.5, 0.3)]
    for index in range(len(circles)):
        xx, yy = sample_points_on_circle(circles[index][0], circles[index][1], circles[index][2], boundary_num)
        xx = xx.reshape(-1, 1).to(device).requires_grad_(True)
        yy = yy.reshape(-1, 1).to(device).requires_grad_(True)
        xs.append(xx)
        ys.append(yy)
        us.append(torch.ones_like(xx).to(device))
    # circle_xs = torch.cat(xs, dim=0).view(-1, 1)
    # circle_ys = torch.cat(ys, dim=0).view(-1, 1)
    
    # sampel on the rectangle
    x1, y1, x2, y2 = (-1, -1, 1, 1)
    x_r = torch.linspace(x2, x2, boundary_num).reshape(-1, 1).to(device).requires_grad_(True)
    y_r = torch.linspace(y1, y2, boundary_num).reshape(-1, 1).to(device).requires_grad_(True)
            
    x_u = torch.linspace(x1, x2, boundary_num).reshape(-1, 1).to(device).requires_grad_(True)
    y_u = torch.linspace(y2, y2, boundary_num).reshape(-1, 1).to(device).requires_grad_(True)
                
    x_l = torch.linspace(x1, x1, boundary_num).reshape(-1, 1).to(device).requires_grad_(True)
    y_l = torch.linspace(y1, y2, boundary_num).reshape(-1, 1).to(device).requires_grad_(True)
                
    x_d = torch.linspace(x1, x2, boundary_num).reshape(-1, 1).to(device).requires_grad_(True)
    y_d = torch.linspace(y1, y1, boundary_num).reshape(-1, 1).to(device).requires_grad_(True)
                
    xs.extend([x_r, x_u, x_l, x_d])
    ys.extend([y_r, y_u, y_l, y_d])
    us.extend([torch.zeros_like(x_r), torch.zeros_like(x_u), torch.zeros_like(x_l), torch.zeros_like(x_d)])
    
    boundary_xs = torch.cat(xs, dim=0)
    boundary_ys = torch.cat(ys, dim=0)
    boundary_us = torch.cat(us, dim=0)
    return (boundary_xs, boundary_ys, boundary_us)

def in_circle(x, y):
    circles = [(0.5, 0.5, 0.2), (0.4, -0.4, 0.4), (-0.2, -0.7, 0.1), (-0.6, 0.5, 0.3)]
    for i in range(len(circles)):
        if (x - circles[i][0]) ** 2 + (y - circles[i][1]) ** 2 < circles[i][2] ** 2:
            return True
    return False