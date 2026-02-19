import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import scipy.stats as st 
import scipy.optimize as op 

import capm
importlib.reload(capm)

position_security ='JNJ'
position_delta_usd = 10
benchmark = '^SPX'
hedge_securities =['META', 'AAPL','MSFT', 'SPY']
# hedge_universe = ['AAPL','MSFT', 'NVDA', '^SPX'] 
hedge_universe = ['XLK','XLV', 'XLP', '^SPX', 'XLY']
regularisation = 0.0

hed = capm.hedge(position_security, position_delta_usd, benchmark, hedge_securities)
hed.compute_betas()
# hed.compute_hedge_weights()
hed.compute_hedge_weights(regularisation)
weights = hed.hedge_weights

# correlaciones
corr = capm.dataframe_correl_beta(position_security, benchmark, hedge_universe)



"""
betas = hed.hedge_betas
target_delta = hed.position_delta_usd # Needed to get a neutral delta
target_beta = hed.position_beta_usd # Needed to get a neutral beta

def cost_function(x, betas, target_delta, target_beta, regularisation):
    dimensions = len(x)
    deltas = np.ones(dimensions)
    f_delta = (np.transpose(deltas).dot(x).item() + target_delta)**2 
    f_beta = (np.transpose(betas).dot(x).item() + target_beta)**2 
    f_penalty = regularisation * (np.sum(x**2))
    f = f_delta + f_beta + f_penalty
    return f

# initial condition
x0 = - target_delta / len(betas) * np.ones(len(betas))  # ← forma (n,)
# due to a new version of optimize, we must have changed the x vector to a 1D array.

# compute optimization
optimal_result = op.minimize(fun = cost_function, x0 = x0,\
                             args = (betas, target_delta, target_beta, regularisation))
                             
hedge_weights = optimal_result.x
print(hedge_weights)
"""                                                                                                                          
                                                                                                                                      