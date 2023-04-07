import os
import numpy as np
import torch

os.environ["DDEBACKEND"] = "pytorch"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import deepxde as dde
from src.optimizer import MultiAdam, LR_Adaptor, LR_Adaptor_NTK
from src.pde.burger import Burger1D, Burger2D
from src.pde.poisson import PoissonClassic, Poisson2D, Poisson3D, PoissonBoltzmann2D, Poisson2DManyArea, PoissonND
from src.utils.callbacks import PlotCallback, TesterCallback

pde = Poisson2D()
net = dde.nn.FNN([pde.input_dim] + 5 * [100] + [pde.output_dim], "tanh", "Glorot normal")

loss_weights = np.ones(pde.num_loss)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
# opt = torch.optim.LBFGS(net.parameters(), max_iter=15000)  # remember to call `set_LBFGS_options` to set max_iter
# opt = MultiAdam(net.parameters(), lr=1e-3)
# opt = LR_Adaptor(opt, loss_weights, pde.num_pde, alpha=0.1)
# opt = LR_Adaptor_NTK(opt, loss_weights, pde.num_pde)
callbacks = []
callbacks.append(PlotCallback(log_every=100))
callbacks.append(TesterCallback(log_every=100))

model = pde.create_model(net)
model.compile(opt, loss_weights=loss_weights)
model.train(iterations=500, display_every=100, callbacks=callbacks, model_save_path="runs/debug")
