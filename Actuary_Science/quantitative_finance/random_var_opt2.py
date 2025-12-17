# Let's create our class
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st 



class simulator_1: # Or class simulator() - with parenthesis
    """
    # WITHOUT Input Class - FORMAL - BEST PRACTICES
    Problems:
        * IS MORE HARD IF WE WANT TO ADD A NEW RANDOM VARIABLE DISTR. WITH THIS CLASS SINCE WE COULD GET TROUBLES WITH THE NAMES, VARIABLES TO ADD AND MORE
        * This class only contains one variable for all the distributions... 
            df=coeff=scale
        And these are problems that we must solve with the Input Class
    """
    def __init__(self, coeff, rv_type, size=10**6, decimals=4):
        """
        The way that we sat this method (builder) remain us the way to declarate variables in java-script
        """
        # As best practices we should write ALL the variables.
        self.coeff = coeff
        self.rv_type = rv_type
        self.size = size # Len of the vector
        self.decimals = decimals
        # I recomend to myself to write all the best practices, for example "__" for private variables.
        # The following variables are not specified into the builder because these emerge from the development of the code:
        self.str_title = None
        self.vector = None
        self.mean = None
        self.volatility = None
        self.skewness = None
        # self.kurtosis = st.kurtosis(self.vector) IS NOT A BEST PRACTICE, WE MUST NOT DO THIS!
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
        self.str_title = self.rv_type
        if self.rv_type == 'standard_normal': # Since the variable standard normal has mean equals to 0 and standaard desviation equals to 1.
            self.vector = np.random.standard_normal(size = self.size) 
            
        elif self.rv_type == 'student':
            self.vector = np.random.standard_t(self.coeff, self.size)
            # DONE: Add the degrees of freedom to the title
            self.str_title += ' df = ' + str(self.coeff)
            
        elif self.rv_type == 'uniform':
            self.vector = np.random.uniform(size = self.size) # 2 arguments predefined by default, we won't change them AND WE NEED TO SPECIFY THAT THE SIZE PARAMETER IS EQUALS TO SELF.SIZE
        
        elif self.rv_type == 'exponential':
            self.vector = np.random.exponential(scale = self.coeff, size = self.size)
            # DONE: We add the scale, in this case is the same variable 'coeff'
            self.str_title += ' scale = ' + str(self.coeff)
            
            # This syntax is the same as: str_title = str_title + ' scale: ' + str(self.coeff)
        elif self.rv_type == 'chi-squared':
            self.vector = np.random.chisquare(df = self.coeff, size = self.size)
            self.str_title += ' df = ' + str(self.coeff)
            
    def compute_stats(self):
        self.mean = st.tmean(self.vector) # Since the self.vector now has a value, the random variable created with the method "generate_vector"
        self.volatility = st.tstd(self.vector) # Remember that volatility is equal to standard desviation.
        self.skewness = st.skew(self.vector)
        self.kurtosis = st.kurtosis(self.vector)
        self.jb_stat = self.size/6 * (self.skewness**2 + 1/4*self.kurtosis**2)
        self.p_value = 1 - st.chi2.cdf(self.jb_stat, df = 2) # In other words:  = 1 - P(X ≤ jb_stat) =  P(X > jb_stat)
        self.is_normal = (self.p_value > 0.5) # KEY: la muestra pasa la prueba de normalidad, o al menos no hay evidencia estadísticamente significativa para decir que no es normal.
        
    def plot(self):
        # Add to the string "Title" the following data... title += Data 
        self.str_title += '\n' + 'mean=' + str(np.round(self.mean, self.decimals)) \
            + ' | ' + 'volatility=' + str(np.round(self.volatility, self.decimals)) \
            + '\n' + 'skewness=' + str(np.round(self.skewness, self.decimals)) \
            + ' | ' + 'kurtosis=' + str(np.round(self.kurtosis, self.decimals)) \
            + '\n' + 'JB stat=' + str(np.round (self.jb_stat, self.decimals)) \
            + ' | ' + 'p-value=' + str(np.round (self.p_value, self.decimals)) \
            + '\n' + 'is _normal=' + str(self.is_normal)
        plt.figure()
        plt.hist(self.vector, bins=100)
        plt.title(self.str_title)
        plt.show()
        
                    
                    
                    
                    