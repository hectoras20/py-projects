import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import scipy.stats as st 

import capm
importlib.reload(capm) 

################
benchmark = 'IVE'
security = 'GCARSOA1_'
m = capm.model(security, benchmark)
m.synchronise_timeseries()
m.plot_timeseries()
m.compute_linear_reg()
m.plot_linear_reg()

h = capm.hedge(security, 10, benchmark, ['AAPL','MSFT'])
h.compute_betas()
h.compute_hedge_weights()

