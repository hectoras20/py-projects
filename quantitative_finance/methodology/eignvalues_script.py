import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import importlib
import market_data
importlib.reload(market_data)
import capm
importlib.reload(capm)

rics = ['BTC-USD','ETH-USD', 'SOL-USD', 'USDC-USD', 'USDT-USD', 'DAI,USD']
# STABLE COINS = 'USDC-USD', 'USDT-USD'. 'DAI,USD'
# NATIVE COINS = 'BTC-USD','ETH-USD', 'SOL-USD'


df = market_data.sychronise_returns(rics)
mxt = df.drop(columns = 'date')

matrix_var_cov = np.cov(mxt, rowvar=False) * 252 # Normalization term (252), the volatility that is the sqrt of variance grows like the sqrt of the time, then variance grows linearly in the time that is why we must multiply in this case by 252 and not by sqrt of 252, since we do this with the variance 
# Thus, the DIAGONAL entries of the previous matrix created, if we take its sqrt we obtain the volatility 
mxt_correl = np.corrcoef(mxt, rowvar=False)

# Compute eigenvalues (lambda_i) and eigenvectors ()
eigenvalues, eigenvectors = np.linalg.eigh(matrix_var_cov)

variance_explained = eigenvalues / sum(eigenvalues) # This will be the percentage of the variance explained

# To check if the matrix of eignvectors is ortogonal... we multiply the matrix by its transpose AND THE OUTPUT MUST BE THE DIAGONAL MATRIX with 1 in the diagonal entries and 0 in other entries.
pord = np.matmul(eigenvectors, np.transpose(eigenvectors))
"""
Notice
Eigenvalues are sorted - The last eignvalues will explain all my portfolio, checking this...
If we check the variance_explained variable we will see how much explain each coordinate, in this case, the last values if we add them, we obtain how much variance explain
We could set a threshold and work with the previous variable.

For example, if we have 6 assets, and the last two eignvalues explain all my university, we reduce the dimension/size of our problem to two 

Remember that the last eignvalues and eignvectors are the most important and useful in the PCA model, these corresponds to the first assets.

Notice that if we want to handle the variance_explianed variable, we must do it as a matrix handling, notice that is an array of float and not a list.
"""

#############
# PCA for 2D 
#############
# Now, let´s have a look to the PCA model and how interpretate the outputs obtained.

# Max and min volaility
# This stablish the range of volaitilty that you can have in your portfolio
volatility_min = np.sqrt(eigenvalues[0])
volatility_max = np.sqrt(eigenvalues[-1])

# Compute PCA base for 2D visualization
# The first important eignvector (column) is the last one:
pca_vector1 = eigenvectors[:, -1]
pca_vector2 = eigenvectors[:, -2]
# Important eignvalues for PCA model
pca_value1 = eigenvalues[-1]
pca_value2 = eigenvalues[-2]
# Variance explained, where this variable is an array of floats = MATRIX, not a list.
pca_var_explained = variance_explained[-2,:].sum

# Compute min variance portfolio
min_var_vector = eigenvectors[:, 0]
min_var_value = eigenvalues[0]
min_var_explained = variance_explained[0]


"""
What we can interpretate in the last block of code - 
If we want to minimize the variance, we must invest in STABLE COINS
If we want to mazimize the vairnace, we must throw the STABLE COINS
With these we can even identify our sectors 
"""






