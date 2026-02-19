# Let's create our class
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st 


class simulation_inputs:
    """
    I CHOOSE THESE PARAMETERS
    This class will contain all the inputs/parameters that I will define to make the process and get my random variable.
    We do not specify the variables that will be computed by the code as mean, std, p_value, jb.
    
    So, if we want to add a new distribution, we must do the following steps:
        1. If is necessary define a new parameter that require the function to create the vector... add it to this class "simulation_inputs"
        2. Add the condition into the "generate_vector" method with its corresponding variables.
        3. If a parameter that was computed into the code now is necessary as a parameter to create the NEW vector (as mean) there is not problem if you call both with the same name.
    """
    def __init__(self):
        self.df = None
        self.scale = None
        self.mean = None # FOR NORMAL DISTR. Not added yet.
        # Here we can add the self.std for NORMAL DISTR. since is a parameter required to create the vector.
        self.rv_type = None
        self.size = None # Len of the vector
        self.decimals = None
        self.evenAddUnnecessaryVariables = None

        

class simulator_2: # Or class simulator() - with parenthesis
    """
    # WITH Input Class - FORMAL - BEST PRACTICES
    """
    def __init__(self, inputs): # Instead of (self, coeff, rv_type, size=10**6, decimals=4) AND I can define the name that I want that will contain the inputs, in this case it is a class.
        """
        The way that we sat this method (builder) remain us the way to declarate variables in java-script
        """
        self.inputs = inputs
        self.str_title = None
        self.vector = None
        self.mean = None # IF WE DO NOT SET THE NORMAL DISTR. THESE VALUE IS COMPUTED INTO THE CODE FOR THE REMAIN DISTRIBUTIONS.
        self.volatility = None
        self.skewness = None
        self.kurtosis = None
        self.jb_stat = None
        self.p_value = None
        self.is_normal = None
        
        
        # These variables WILL BE COMPLETE THEM WITH VALUES IN THE DEVOLPMENT OF THE CODE, that is why these are definied with the keyword "None"
        
        # Let us create our calculator methods
        # DONE: Create a Random Variables
    def generate_vector(self):
        """
        This method does not return something.
        Contains attributes that we can call later into the code
        Normal D. - 
        Student D. - df
        Uniform D - 
        Exponentual D. - scale
        Chi-Squared D. - df
        """
        self.str_title = self.inputs.rv_type
        if self.inputs.rv_type == 'standard_normal':
            self.vector = np.random.standard_normal(self.inputs.size)
            
        elif self.inputs.rv_type == 'student':
            self.vector = np.random.standard_t(self.inputs.df, self.inputs.size)
            # DONE: Add the degrees of freedom to the title
            self.str_title += ' with df = ' + str(self.inputs.df)
            
        elif self.inputs.rv_type == 'uniform':
            self.vector = np.random.uniform(size = self.inputs.size) # 2 arguments predefined by default, we won't change them.
        
        elif self.inputs.rv_type == 'exponential':
            self.vector = np.random.exponential(scale = self.inputs.scale, size = self.inputs.size)
            # DONE: We add the scale, in this case is the same variable 'coeff'
            self.str_title += ' with scale = ' + str(self.inputs.scale)
            
            # This syntax is the same as: str_title = str_title + ' scale: ' + str(self.coeff)
        elif self.inputs.rv_type == 'chi-squared':
            self.vector = np.random.chisquare(df = self.inputs.df, size = self.inputs.size)
            self.str_title += ' with df = ' + str(self.inputs.df)
            
    def compute_stats(self):
        self.mean = st.tmean(self.vector) # Since the self.vector now has a value, the random variable created with the method "generate_vector"
        self.volatility = st.tstd(self.vector) # Remember that volatility is equal to standard desviation.
        self.skewness = st.skew(self.vector)
        self.kurtosis = st.kurtosis(self.vector)
        self.jb_stat = self.inputs.size/6 * (self.skewness**2 + 1/4*self.kurtosis**2)
        self.p_value = 1 - st.chi2.cdf(self.jb_stat, df = 2) # In other words:  = 1 - P(X ≤ jb_stat) =  P(X > jb_stat) 
        self.is_normal = (self.p_value > 0.5) # = JB < 0.6 
        # KEY: la muestra pasa la prueba de normalidad, o al menos no hay evidencia estadísticamente significativa para decir que no es normal.
        
    def plot(self):
        # Add to the string "Title" the following data... title += Data 
        self.str_title += '\n' + 'mean=' + str(np.round(self.mean, self.inputs.decimals)) \
            + '|' + 'volatility=' + str(np.round(self.volatility, self.inputs.decimals)) \
            + '\n' + 'skewness=' + str(np.round(self.skewness, self.inputs.decimals)) \
            + '|' + 'kurtosis=' + str(np.round(self.kurtosis, self.inputs.decimals)) \
            + '\n' + 'JB stat=' + str(np.round (self.jb_stat, self.inputs.decimals)) \
            + '|' + 'p-value=' + str(np.round (self.p_value, self.inputs.decimals)) \
            + '\n' + 'is _normal=' + str(self.is_normal)
        plt.figure()
        plt.hist(self.vector, bins=100)
        plt.title(self.str_title)
        plt.show()
        
                    
                    
                    
                    