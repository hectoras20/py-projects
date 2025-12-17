import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import scipy.stats as st 

import random_var_opt1_clean as rv_class

import random_var_opt2 as rv_class_simple

importlib.reload(rv_class)
importlib.reload(rv_class_simple)

#####################
# rv_class_simple class without input class
#####################
coeff = 2
rv_type = 'standard_normal'
size=10**6
decimals=4

sim2 = rv_class_simple.simulator_1(coeff = coeff, rv_type = rv_type)
sim2.generate_vector()
sim2.compute_stats()
sim2.plot()

#####################
# rv_class class with input class
#####################

# We create our class into the objet called "inputs", with this we now can set the value of ITS ATTRIBUTES
inputs = rv_class.simulation_inputs() 

# We define a value to each attribution of the class
inputs.df = 2
inputs.scale = 5 # scale in exponential only at the moment
inputs.mean = 5 # mean in 'normal', NOT ADDED YET
inputs.std = 10 # std in ¿normal', NOT ADDED YET
inputs.size = 10**6
inputs.rv_type = 'standard_normal'
inputs.decimals = 5

# We will only use the size attribute if the rv_type is equal to 'normal'
sim1 = rv_class.simulator_2(inputs)
sim1.generate_vector()
sim1.compute_stats()
sim1.plot()

#####################
# Loop of JB normality test
#####################
n = 0 # Number of repetition of the experiment until get a False
is_normal = True
str_title = 'normal'

while is_normal and n<100:    
    x = np.random.standard_normal(size=10**6)
    mu = st.tmean(x) # tmean
    sigma = st.tstd(x) # tstd
    skew = st.skew(x)
    kurt = st.kurtosis(x)
    jb_stat = len(x)/6 * (skew**2 + (1/4)*kurt**2)
    p_value= 1 - st.chi2.cdf(jb_stat, df=2)
    is_normal = (p_value> 0.05) # equivalently jb < 6
    print('n=' + str(n) + ' | is_normal=' + str(is_normal))
    n += 1
    
# This is related with the p_value, if we do the comparation smaller for example, p_value < 0.1 this will take more repetitions until get a False, since the error was reduced.