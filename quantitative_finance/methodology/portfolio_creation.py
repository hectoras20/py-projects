import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import importlib
import random
import scipy.optimize as op 

import market_data
importlib.reload(market_data)
import capm
importlib.reload(capm)
import portfolio_class
importlib.reload(portfolio_class)

notional = 10 # Amount to invest 
universe = ['^SPX','^IXIC','^MXX','^STOXX','^GDAXI','^FCHI','^VIX',\
            'XLK','XLF','XLV','XLE','XLC','XLY','XLP','XLI','XLB','XLRE','XLU',\
            'SPY','EWW',\
            'IVW','IVE','QUAL','MTUM','SIZE','USMV',\
            'AAPL','MSFT','NVDA','AMZN','GOOG','META','NFLX',\
            'BRK-B','JPM','V','MA','BAC','MS','GS','BLK',\
            'LLY','JNJ','PG','MRK','ABBV','PFE',\
            'BTC-USD','ETH-USD','SOL-USD','USDC-USD','USDT-USD','DAI-USD',\
            'EURUSD=X','GBPUSD=X','CHFUSD=X','SEKUSD=X','NOKUSD=X','JPYUSD=X','MXNUSD=X']
rics = random.sample(universe, 5)

# initialize the instance - This will be my first class (works as the input class)
portfolio_obj = portfolio_class.manager(rics, notional)

# Commpute var_cov matrix and correlation matrix
portfolio_obj.compute_covariance()

# Compute the desired portfolio - This will be a second class, example: portfolio.output
portf_min_var_l1 = portfolio_obj.compute_portfolio('min_var_l1')
portf_min_var_l2 = portfolio_obj.compute_portfolio('min_var_l2')
port_equi_weight = portfolio_obj.compute_portfolio('equi_weight')
# Testing our new portfolios
portf_long_only = portfolio_obj.compute_portfolio('long_only') # I do not care the return I want to minimize the volatility
portf_markowitz = portfolio_obj.compute_portfolio('markowitz', 0.25)

# RETURNS WITH MARKOWITZ
# target_returns GIVEN, desired, that should be equal to the next line of code
return_target = portf_markowitz.targer_return
# r^T - returns computed = portfolio_class.manager.returns = portfolio_obj.returns
# x = weights, indicating that comes from the markokitz portfolio
return_portfolio_markowitz = np.round(portfolio_obj.returns.dot(portf_markowitz.weights), 6) # r^T * x

# RETURNS WITH LONG ONLY
return_portfolio_long_only = np.round(portfolio_obj.returns.dot(portf_long_only.weights), 6) 

# RETURNS WITH EQUI
return_portfolio_equi = np.round(portfolio_obj.returns.dot(port_equi_weight.weights), 6)

# WITH THESE OUTCOMES YOU CAN DEFINE A TARGET RETURN... you can see the entire returns with self.variances... ¿?


# DATAFRAME 
df_m = pd.DataFrame()
df_m['rics'] = rics
df_m['returns'] = portfolio_obj.returns 
df_m['volatilities'] = portfolio_obj.volatilities
df_m['markowitz_weights'] = portf_markowitz.weights
df_m['markowitz_allocation'] = portf_markowitz.allocate
df_m['long_only_weights'] = portf_long_only.weights
df_m['long_only_allocation'] = portf_long_only.allocate
df_m['equi_weights'] = port_equi_weight.weights
df_m['equi_allocation'] = port_equi_weight.allocate
df_m['min_var_l1_weights'] = portf_min_var_l1.weights
df_m['min_var_l1_allocation'] = portf_min_var_l1.allocate
