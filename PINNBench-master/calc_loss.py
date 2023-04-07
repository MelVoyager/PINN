import os

os.environ["DDEBACKEND"] = "pytorch"

import numpy as np
import pandas as pd
from src.optimizer import MultiAdam, LR_Adaptor, LR_Adaptor_NTK
from src.pde.burger import Burger1D, Burger2D
from src.pde.chaotic import KuramotoSivashinskyEquation, GrayScottEquation
from src.pde.heat import HeatComplex, HeatDarcy, HeatLongTime, HeatMultiscale
from src.pde.ns import NSEquation_FourCircles, NSEquation_LidDriven, NSEquation_Long
from src.pde.poisson import PoissonClassic, Poisson2D, Poisson2DManyArea, Poisson3D, PoissonBoltzmann2D, PoissonND
from src.pde.wave import WaveEquation1D, WaveEquation2D_Long, WaveHetergeneous
from src.utils.callbacks import TesterCallback, PlotCallback


def calc_avg(func, path, repeat):
    data = []
    try:
        for i in range(repeat):
            data.append(func(np.loadtxt(path.format(i))))
    except ValueError:
        assert len(data) == 0
        for i in range(repeat):
            data.append(func(open(path.format(i)).readlines()))
    return np.mean(data)


pde_list = \
    [Burger1D, Burger2D] + \
    [KuramotoSivashinskyEquation, GrayScottEquation] + \
    [HeatComplex, HeatDarcy, HeatLongTime, HeatMultiscale] + \
    [NSEquation_FourCircles, NSEquation_LidDriven, NSEquation_Long] + \
    [PoissonClassic, Poisson2D, Poisson2DManyArea, Poisson3D, PoissonBoltzmann2D, PoissonND] + \
    [WaveEquation1D, WaveEquation2D_Long, WaveHetergeneous]

exp_path = 'runs/03.31-20.34.47-all'
repeat = 3

columns = ['pde', 'iter', 'time', 'train loss', 'mse', 'l2rel']
result = []


def extract_time(lines):
    # example: 'train' took 253.845810 s
    for line in lines:
        line = line.strip()
        if line.startswith("'train'"):
            return float(line.split(' ')[2])
    raise ValueError('Could not find training time.')


if __name__ == '__main__':
    for i, pde in enumerate(pde_list):
        iter = 20000
        run_time = calc_avg(extract_time, '{}/{}-{{}}/log.txt'.format(exp_path, i), repeat)
        train_loss = calc_avg(lambda data: data[-1, 1], '{}/{}-{{}}/loss.txt'.format(exp_path, i), repeat)
        try:
            mse = calc_avg(lambda data: data[-1, 2], '{}/{}-{{}}/errors.txt'.format(exp_path, i), repeat)
            l2rel = calc_avg(lambda data: data[-1, 4], '{}/{}-{{}}/errors.txt'.format(exp_path, i), repeat)
        except FileNotFoundError:
            mse = np.nan
            l2rel = np.nan
        result.append([pde.__name__, iter, run_time, train_loss, mse, l2rel])

# save csv
df = pd.DataFrame(result, columns=columns)
df.to_csv(f'{exp_path}/result.csv')
