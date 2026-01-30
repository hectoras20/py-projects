# Markowitz portafolio require variance-covariance matrix
# Beta is a covariance

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import importlib
import market_data
importlib.reload(market_data)
import capm
importlib.reload(capm)

# With the followign rics, we will compute the covariance and variance values
# The dimension of variance-covariance matrix is the number of assets that we pass into the rics variable x these number again. 
rics = ['^MXX', '^SPX']

# 1. We must synchronize all the timeseries of returns ... means that we must fit all the dates between all the assets
# Got the timeseries of each one, we did it with a loop and market-data - load_timeseries function
# But for the variance-covariance matrix we do not need Prices, we only need the returns of each one to compute it
 
# synchronise all the timeseries of returns
df = pd.DataFrame()
dic_timeseries = {}
timestamps = [] # At the end of all the iterations, this will contain the compatible dates between all the rics
# Get intersection of all timestamps
for ric in rics:
    # We got the timeserie with our function created from market-data
    # t is a dataframe with data, close, previous_close and return columns
    t = market_data.load_timeseries(ric)
    
    # This dictonary is created to avoid run twice the timeseries, so we save each dataframe in a dictionary unfiltered, originals.
    dic_timeseries[ric] = t
    # Then we will make a second loop over this dictionary 

    # With t´s we will take the intersection of all the compatible dates
    # The following line only works for the first iteration since we do not have a set yet to make the comparation.
    if len(timestamps) == 0:
        timestamps = list(t['date'].values) # This is the same variable that is out of this block
        
    # This variable works to make the comparation in the following line code
    temp_timestamps = list(t['date'].values)  
    timestamps = list(set(timestamps) & set(temp_timestamps))
        # The first iteration make the comparation between the same timestamp
        # The second one, make the compartion with the previous timestamp compared in the first interation (was between the same) with the new timestamp got it in this new iteraion, the timestamp substracted of the second ric,  in other words

# SYNCHRONIZATION of all the timeseries - Using the dictionary created with original dataframes, unfiltered
for ric in dic_timeseries:
    # We take again a dataframe
    t = dic_timeseries[ric]
    # We make the filter 
    t = t[t['date'].isin(timestamps)]
    # Cleaning the new dataframe
    t = t.sort_values(by='date', ascending=True)
    t = t.dropna()
    t = t.reset_index(drop=True)
    # We overwrite/update the dataframe that we take at the begening with the new dataframe
    dic_timeseries[ric] = t
    
    # Now we create our final dataset that will contain all the neccesary information to create the variance-covariance matrix
    if df.shape[1] == 0:
        df['date'] = timestamps
    df[ric] = t['return']



# 2. Compute the variance-covariance matrix
# Be careful, we already have our dataset with the neccesary information BUT we do not need the first column, this was useful to get the same dates between assets
mxt = df.drop(columns = 'date')
# Then we will use Numpy to compute the variance-covariance matrix, in this case we pass two parameters, the dataframe (called "mxt") and we must indicate IN THIS CASE rowvar = False, since our variables are not the rows from mxt, these are the columns.
matrix_var_cov = np.cov(mxt, rowvar=False)


# 3. Compute the correlation matrix with the matrix filtered, no with the var-corr matrix
# We will use Numpy library again for this step.
# Obs. This function works with the Pearson correlation that we use in the topic of ACPM, so we need to get the same outcomes
mxt_correl = np.corrcoef(mxt, rowvar=False)

# The observation that we made is related with the following line code:
correl = capm.compute_correlation('^MXX', '^SPX')
# So, we can not expect the same values with the correlation matrix since we made a filter of dates, so the lenght of the data used is not the same between both methods. But at least the first decimals will be the same.

# A Quant, when must follow a model? For example in the previous situation we got different correlation values between the ACPM model with the Numpy function used to get the same thing, so which method follow? This will depend on the Quant, if that little difference is unnoticed. 

# We will create a function to avoid large code that may make noise to the customer that will use it.
# It will located in the market_data class

# CONCLUSION 
# At the end we add to rics stablecoins and cripto, we saw the correlation between these and indexes, so at the end we can saw it as 3 SECTORS
# A sector is a block of assets that have correlation at least between them or bussiness model similar
# So, we can have a lot of assets (in rics) and try yo find hiden relations and using machine learning, watching which assets have correlation and more techniques to define sectors 
# In this example we already knew that there was 3 sectores, but we can fine them or create them in base of its correlation
# Some assets could behave as a sector but in a different time behave as some different sector.
