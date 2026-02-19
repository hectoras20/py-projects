"""
Given a portafolio, a rics list
We must return a list of weights = how much do we invest in each asset given X amount
We could give a lot of solutions, we already know the min variance portfolio
"""
# In our previous code... 
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

rics = ['BTC-USD','ETH-USD', 'SOL-USD', 'USDC-USD', 'USDT-USD', 'DAI-USD']
# STABLE COINS = 'USDC-USD', 'USDT-USD'. 'DAI,USD'
# NATIVE COINS = 'BTC-USD','ETH-USD', 'SOL-USD'


df = market_data.sychronise_returns(rics)
mxt = df.drop(columns = 'date')

matrix_var_cov = np.cov(mxt, rowvar=False) * 252 # Normalization term (252), the volatility that is the sqrt of variance grows like the sqrt of the time, then variance grows linearly in the time that is why we must multiply in this case by 252 and not by sqrt of 252, since we do this with the variance 
# Thus, the DIAGONAL entries of the previous matrix created, if we take its sqrt we obtain the volatility 
mxt_correl = np.corrcoef(mxt, rowvar=False)

# Compute eigenvalues (lambda_i) and eigenvectors ()
eigenvalues, eigenvectors = np.linalg.eigh(matrix_var_cov)
variance_explained = eigenvalues / sum(eigenvalues)

"""MIN VARIANCE PORTFOLIO
If we have an amount of 10 million dollarrs, the weights (how much do we need to invest in each asset)
are computed by multiplying the values of the last eigenvector (that is the vector with min variance) by 10 millions 
to know the weights.

If we want to get the norma L_1 we do:
    sum(abs(min_var_vector))
    Is the sum of absolute entries of the FIRST vector, the last vector is the eigenvalue with 
    the most min variance in our portfolio
    - The vector is not unitary in this norm
    - This means that not necessarily the FIRST eignvalue is the min variance vector! 
To get the L2 norm, we do:
    sum(min_var_vector**2"")
    Is the sum of squar entries of the FIRST vector since we are working with the min variance portfolio.    
    - The vector is unitary in this norm
    """
volatility_min = np.sqrt(eigenvalues[0])
volatility_max = np.sqrt(eigenvalues[-1])

# Compute PCA base for 2D visualization
# The first important eignvector (column) is the last one:
pca_vector1 = eigenvectors[:, -1]
pca_vector2 = eigenvectors[:, -2]
# So, the mportant eignvalues for PCA model are:
pca_value1 = eigenvalues[-1]
pca_value2 = eigenvalues[-2]
# Variance explained, where this variable is an array of floats = MATRIX, not a list.
pca_var_explained = variance_explained[-2:].sum() # In this case, we are taking the last values that are with most value. But we could take the last 3, etc.

# Compute min variance portfolio
min_var_vector = eigenvectors[:, 0]
min_var_value = eigenvalues[0]
min_var_explained = variance_explained[0]

"""KEY WORD: OPTIMIZATION"""
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
min-var with scipy optimize minize
We are trying to find a way to compute portfolios
"""
# Initial condition = EQUIWEIGHTS - Equiponderado and we are assuming positive entries.
x0 = [notional / len(rics)] * len(rics)
# Possible constraints 
l2_norm = [{"type" : "eq", "fun": lambda x: sum(x**2) - 1}] # unitary in norm L2
l1_norm = [{"type" : "eq", "fun": lambda x: sum(abs(x)) - 1}] # unitary in norm L1

# We need previously define a function called portfolio_variance
def portfolio_variance(x, mtx_var_cov):
    """We want to minimize the variance, we will do it OVER the L2 norm
    - min (variance) f(x) = z = x^T * W * x 
    where:
        Q is the var_cov_matrix
        x is our initial condition (equi-weights) - vector (the product between vectors must be x * x^T)
    - under the constraints = over the unitary sphere ||x||_2
    """
    # Functions x to minimize (variance)
    variance = np.matmul(np.transpose(x), np.matmul(mtx_var_cov, x)) # WE WILL GET ONE VALUE
    # UNIT TEST FOR VARIANCE FUNCTIONS... If x = min_var_vector we will get the first value of the eigenVALUES variable
    # Constraints
    return variance

# x0 - first argument is already defined
# matrix_var_cov is the next argument must be defined (all the next arguments must be defined here but in this case is only one)
optimal_result = op.minimize(fun = portfolio_variance, x0 = x0,\
                             args = (matrix_var_cov),\
                             constraints = l2_norm)

optimize_vector = optimal_result.x

df_weights = pd.DataFrame()
df_weights['rics'] = rics
df_weights['min var vector'] = min_var_vector # That is the first column of the eigenvalues matrix
df_weights['optimize'] = optimize_vector

"""Conclusions
- The min variance portfolio depends on the norm which we are working

- We could get the same solution but with oposite sign, but it does not care, 
at the end both solutions are valid, the variance is the same we are just changing the direction and both are unitaries, 
since we are working with the norm L2, depends on what we want, for example if we want an specific asset negative
"""




