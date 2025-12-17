import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import scipy.stats as st 

import capm # This class do the import of market_data
importlib.reload(capm) 
import portfolio_class
importlib.reload(portfolio_class) 

# factors = ['^SPX', 'IVW', 'IVE', 'QUAL', 'MTUM', 'SIZE', 'USMV', \
           # 'XLK', 'XLF', 'XLV', 'XLP', 'XLY', 'XLI', 'XLC', 'XLU'] # RESPECCTIVE TO USA FACTORS
# security = 'GCARSOA1'
# factors = ['MTUM_', 'QUAL_', 'IVW_', 'GCARSOA1']
# benchmark = 'IVW_'
security = 'AAPL'
factors = ['MTUM', 'QUAL', 'IVW', 'AAPL']
benchmark = 'IVW'

# Gives the correlation and beta of a security in relation with a list of assets. 
fac = capm.dataframe_factors(security, factors) # Since it is a function out of the class

# Hacer plot de dos activos 
obj = capm.model(security, benchmark)
obj.synchronise_timeseries()
obj.plot_timeseries()

# Correlación entre varios activos - Será un atributo de la clase
corr = portfolio_class.manager(factors, 1)
corr.compute_covariance()