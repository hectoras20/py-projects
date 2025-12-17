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

"""
In finance we work with the norm L1, that is why we make normalizations"""

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

notional = 10 # Amount to invest 

df = market_data.sychronise_returns(rics)
mxt = df.drop(columns = 'date')
# Since the volatility that is the sqrt of variance grows like the sqrt of the time, then variance grows linearly in the time that is why we must multiply in this case by 252 and not by sqrt of 252, since we do this with the variance 
matrix_var_cov = np.cov(mxt, rowvar=False) * 252 # Normalization term (252)
mxt_correl = np.corrcoef(mxt, rowvar=False)

# min-var with eigenvalues
eigenvalues, eigenvectors = np.linalg.eigh(matrix_var_cov)
min_var_vector = eigenvectors[:, 0]
"""
One of the changes that we did is NORMALISE the min_var_vector that is the first column in our 
eigenvectors matrix
We could see that min_var_vector variable is origaly in norm L2
We normalize it dividing by the norm L1
Now with this the same variable belongs to the norm L1

min_var_vector /= sum(abs(min_var_vector)) 
The sum of absolute entries is not necessarily equals 1 BEFORE the normalization, 
we make the normalization with this.

BUT, we want to know the weights (how much invest in each asset)... we must apply the normalization to the notional multiplied by the min_var_vector 
"""
min_var_vector = notional * min_var_vector / sum(abs(min_var_vector))


"""
min-var with scipy optimize minize
We are trying to find a way to compute portfolios
"""
# Initial condition = EQUIWEIGHTS - Equiponderado and we are assuming positive entries.
"""
We will do the same (normalization) for the x0 initial condition
"""
x0 = [notional / len(rics)] * len(rics)
# Possible constraints 
l2_norm = [{"type" : "eq", "fun": lambda x: sum(x**2) - 1}] # unitary in norm L2
l1_norm = [{"type" : "eq", "fun": lambda x: sum(abs(x)) - 1}] # unitary in norm L1

# We need previously define a function called portfolio_variance
def portfolio_variance(x, mtx_var_cov):
    variance = np.matmul(np.transpose(x), np.matmul(mtx_var_cov, x)) # WE WILL GET ONE VALUE
    return variance


optimal_result = op.minimize(fun = portfolio_variance, x0 = x0,\
                             args = (matrix_var_cov),\
                             constraints = l1_norm)

"""
We also normalise to the norm L1 this output since we do the optimization with the norm L2
Once we normalize to the norm L1, we can image that we are making an addition of x with y such that is equal to 1.

The equation x+y=1 is the straight line with norm (1,1), so we will have points that goes from (0,1) to (1,0)
A diagonal

We achieve that with:
    optimize_vector /= sum(abs(optimal_result.x))
    
But we need  normalizated to the norm L1 the notional * optimized value, since we are interested in the weights.
"""
optimize_vector = optimal_result.x
optimize_vector = notional * optimize_vector / sum(abs(optimize_vector))


df_weights = pd.DataFrame()
df_weights['rics'] = rics
df_weights['min_var_vec_l2'] = min_var_vector # That is the first column of the eigenvalues matrix, this is originaly in norm L2
df_weights['optimize_l1'] = optimize_vector
df_weights['default'] = x0

