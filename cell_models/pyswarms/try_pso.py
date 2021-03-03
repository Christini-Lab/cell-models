import copy
import time

# Import modules
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from pyswarms.single.global_best import GlobalBestPSO
from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
from pyswarms.utils.plotters.formatters import Mesher

from cell_models import kernik, protocols
from cell_models.ga import target_objective 

proto = protocols.VoltageClampProtocol()

global TARGET
target_mod = kernik.KernikModel()
TARGET = target_objective.create_target_from_protocol(target_mod, proto)

global CHANNELS
CHANNELS = ['G_Kr', 'G_Ks', 'G_CaL', 'G_Na', 'G_To', 'G_K1']

#global POOL
#POOL = multiprocessing.Pool()

def return_error(ind_dat):
    conductances = ind_dat[0] 
    target = ind_dat[1]

    mod = kernik.KernikModel(updated_parameters=conductances)

    error = target.compare_individual(mod)

    return error


def find_individual_values(vals):
    all_inds = []
    for row in vals:
        row = [10**v for v in row]
        all_inds.append([dict(zip(CHANNELS,row)),
            copy.copy(TARGET)])
    
    t = time.time()
    p = multiprocessing.Pool()
    errors = p.map(return_error, all_inds)
    p.close()

    print(f'Time: {time.time() - t}')

    return errors

# instatiate the optimizer
x_max = 1 * np.ones(len(CHANNELS))
x_min = -1 * np.ones(len(CHANNELS))
bounds = (x_min, x_max)
# c1 = personal best, c2 = global best, w = inertia 
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
optimizer = GlobalBestPSO(n_particles=80, dimensions=len(CHANNELS), options=options, bounds=bounds)

cost, pos = optimizer.optimize(find_individual_values, 80)

plot_cost_history(cost_history=optimizer.cost_history)
plt.show()

import pdb
pdb.set_trace()
