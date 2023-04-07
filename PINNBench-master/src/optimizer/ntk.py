import deepxde as dde
import torch


class LR_Adaptor_NTK(torch.optim.Optimizer):
    """
    PINN callback for learning rate annealing algorithm of physics-informed neural networks.
    """

    def __init__(self, optimizer, loss_weight, num_pde):
        '''
        loss_weight - initial loss weights
        num_pde - the number of the PDEs (boundary conditions excluded)
        '''

        super().__init__(optimizer.param_groups, defaults={})
        self.optimizer = optimizer
        self.loss_weight = loss_weight
        self.num_pde = num_pde
        self.iter = 0

    @torch.no_grad()
    def step(self, closure):
        self.iter += 1
        with torch.enable_grad():
            _ = closure(skip_backward=True)
            losses = self.losses / torch.as_tensor(self.loss_weight)  # get non_weighted loss from closure
            pde_loss = torch.sum(losses[:self.num_pde])
            boundary_loss = torch.sum(losses[self.num_pde:])

        m_grad_r = []
        self.zero_grad()
        pde_loss.backward(retain_graph=True)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    m_grad_r.append(torch.zeros(p.size))
                else:
                    m_grad_r.append(torch.abs(p.grad).reshape(-1))
        m_grad_r = torch.sum(torch.cat(m_grad_r)**2).item()

        m_grad_b = []
        self.zero_grad()
        boundary_loss.backward(retain_graph=True)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    m_grad_b.append(torch.zeros(p.size))
                else:
                    m_grad_b.append(torch.abs(p.grad).reshape(-1))
        m_grad_b = torch.sum(torch.cat(m_grad_b)**2).item()

        for i in range(self.num_pde):
            self.loss_weight[i] = (m_grad_r + m_grad_b) / m_grad_r
        for i in range(self.num_pde, len(self.loss_weight)):
            self.loss_weight[i] = (m_grad_r + m_grad_b) / m_grad_b

        with torch.enable_grad():
            total_loss = torch.sum(losses * torch.as_tensor(self.loss_weight))
            self.zero_grad()
            total_loss.backward()
        self.optimizer.step()

        # if self.iter % 100 == 0:
        #     print(self.loss_weight)
