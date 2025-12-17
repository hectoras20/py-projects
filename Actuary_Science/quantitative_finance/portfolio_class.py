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

'''def portfolio_variance(x, mtx_var_cov):
    variance = np.matmul(np.transpose(x), np.matmul(mtx_var_cov, x)) # WE WILL GET ONE VALUE
    return variance

class manager:
    def __init__(self, rics, notional):
        self.rics = rics
        self.notional = notional
        # Some important variables that we could extract from the development of the code. 
        self.mtx_var_cov = None
        self.mxt_correl = None
        self.weights = None
        
    def compute_covariance(self):
        df = market_data.sychronise_returns(self.rics)
        mxt = df.drop(columns = 'date')
        self.mtx_var_cov = np.cov(mxt, rowvar=False) * 252 # Normalization term (252)
        self.mxt_correl = np.corrcoef(mxt, rowvar=False)
        
    def compute_portfolio(self, portfolio_type = 'default'):
        x0 = [self.notional / len(self.rics)] * len(self.rics)
        # Possible constraints 
        l2_norm = [{"type" : "eq", "fun": lambda x: sum(x**2) - 1}] # unitary in norm L2
        l1_norm = [{"type" : "eq", "fun": lambda x: sum(abs(x)) - 1}] # unitary in norm L1
        if portfolio_type == 'min_var_l1':
            optimal_result = op.minimize(fun = portfolio_variance, x0 = x0, args = (self.mtx_var_cov), constraints = (l1_norm))
            weights = optimal_result.x
        elif portfolio_type == 'min_var_l2':
            optimal_result = op.minimize(fun = portfolio_variance, x0 = x0, args = (self.mtx_var_cov), constraints = (l2_norm))
            weights = optimal_result.x
        else: # Is the type is default then we will get the x0 variable
            weights = x0
            
        # Output class (defined at the last of the script)
        optimal_portfolio = output(self.notional, self.rics)
        optimal_portfolio.type = portfolio_type
        # Then I could use the attributes of output class 
        optimal_portfolio.weights = self.notional * weights / sum(abs(weights))
        return optimal_portfolio
        
# We need a new class that will be the output
class output:
    def __init__(self, notional, rics):
        self.rics = rics
        self.notional = notional
        self.type = None
        self.weights = None
'''
def portfolio_variance(x, mtx_var_cov):
    variance = np.matmul(np.transpose(x), np.matmul(mtx_var_cov, x)) # WE WILL GET ONE VALUE
    return variance

class manager:
    def __init__(self, rics, notional):
        self.rics = rics
        self.notional = notional
        # Some important variables that we could extract from the development of the code. 
        self.mtx_var_cov = None
        self.mxt_correl = None
        """
        We will add the following variables due we add more protfolios
        """
        self.returns = None
        self.volatilities = None
        
    def compute_covariance(self):
        decimals = 10
        # We add factor since we are using this value in more code lines... we repeat a process or value more than twice so we must consider it as an encapsulation
        # We add factor since we are using this value in more lines of code... we repeat a process or value more than twice so we should consider it as an encapsulation
        factor = 252
        df = market_data.sychronise_returns(self.rics)
        mxt = df.drop(columns = 'date')
        self.mtx_var_cov = np.cov(mxt, rowvar=False) * factor # Normalization term (252)
        self.mxt_correl = np.corrcoef(mxt, rowvar=False)
        
        #FOR MARKOWITZS
        # We are creating the returns and variance vector transposed to make the dot product with x (x is the vector with min variance)
        #self.returns = np.array([ np.round( np.mean(df[ric]) * factor , decimals ) for ric in self.rics])
        #self.variances = np.array([ np.round( np.std(df[ric]) * np.sqrt(factor) ,decimals) for ric in self.rics])
        returns = []
        volatilities = []
        for ric in self.rics:
            r = np.round(np.mean(df[ric]) * factor, decimals)
            v = np.round(np.std(df[ric]) * np.sqrt(factor), decimals)
            returns.append(r)
            volatilities.append(v)
        self.returns = np.array(returns)
        self.volatilities = np.array(volatilities)
        # For our dataframe
        df_m = pd.DataFrame()
        df_m['rics'] = self.rics
        df_m['returns'] = self.returns 
        df_m['volatilities'] = self.volatilities
        self.dataframe_metrics = df_m
        
        
        
    def compute_portfolio(self, portfolio_type = None, target_return = None): # If we do not define the target_return argument there is not problem. Nothing happens
        """
        portfolio type - to indicate the portfolio desired 
        target_return - for Markowitz portfolio  
        
        We make a cast of weights to arrays since we will need make dot product between two vectors
        """
        decimals = 6 
        # INITIAL CONDITION - That must be normalized
        x0 = [1 / len(self.rics)] * len(self.rics)
        
        # CONSTRAINTS
        l2_norm = [{"type" : "eq", "fun": lambda x: sum(x**2) - 1}] # unitary in norm L2
        l1_norm = [{"type" : "eq", "fun": lambda x: sum(abs(x)) - 1}] # unitary in norm L1
        # FOR MARKOWITZ
        markowitz = [{"type" : "eq", "fun": lambda x: self.returns.dot(x) - target_return}] # (r^T * x = r_return)
        
        # BOUNDARI CONDITIONS for non negative weights - Condiciones de frontera
        """
        List of tuples 
        The first entry of each tuple is equals to 0 (the min) and the second one is the maximal value, when this is None means infinity 
        We got the number of assets tuples
        
        We could generalize it to negative values (short trading) instead of non-negative weights = long trading
        """
        non_negative = [(0, None) for i in range(len(self.rics))] 
        
        # SELECTION OF PORTFOLIO TYPE
        if portfolio_type == 'min_var_l1':
            optimal_result = op.minimize(fun = portfolio_variance, x0 = x0, args = (self.mtx_var_cov), constraints = (l1_norm))
            # We make a "CAST" to numpy arrays
            weights = np.array(optimal_result.x)
            
        elif portfolio_type == 'min_var_l2':
            optimal_result = op.minimize(fun = portfolio_variance, x0 = x0, args = (self.mtx_var_cov), constraints = (l2_norm))
            weights = np.array(optimal_result.x)
            
        # ADDING LONG ONLY PORTFOLIO = NON NEGATIVE WEIGHTS
        elif portfolio_type == 'long_only':
            '''
            We should add boundary conditions to this portfolio 
            In the optimize minime functions we get it with the argument "bounds"
            
            This portfolio minimize the variance 
            In the norm L1 space
            constrained to non negative weights
            
            The sum of the absolute weight entries must equal 1
            '''
            optimal_result = op.minimize(fun = portfolio_variance, x0 = x0, args = (self.mtx_var_cov), constraints = (l1_norm), bounds=non_negative)
            weights = np.array(optimal_result.x)
            
        # MARKOWITZ PORTFOLIO
        elif portfolio_type == 'markowitz':
            '''
            We are still minimizing the variance 
            In the norm L1 space since we stablish use this for finances (but we could use the norm L2 space)
            
            ******* WE MUST SUM TO THE RESTRICTION OF L1 THE NEW RESTRICTION OF MARKOWITZ (r^T * x = r_return) ******
            
            Constrained to non negative weights
            
            NEW CONSTRAINT/Restriction - (r^T * x = r_return) :
                
                dot product of vectors r and x ... r^T * x = portfolio return that MUST BE EQUAL TO the given r_target - objetivo
                
                So, we must give the return_target as a value
                We need a return_vector equals to r^T (self.returns) - We get it with our dataframe of returns already created
                
                - For return we compute the mean of RETURNS - Since we want it annualized we multiply it by the factor
                
                Then, we could do the same for variance, now we give a variance_target and we must compute the variance^T vector, as we did previusly
                - For variance we compute the standard desviation OVER THE RETURNS - Since we want it annualized we multiply it by the square root of factor 
                
            The sum of the absolute weight entries must equal 1
            
            
            SINCE THE PORTFOLIO MUST TAKE NON NEGATIVE WEIGHTS...
            The target return (value) must be between the range of minimal and maximal returns of our universe given.
            To avoid future problems... we solve the following scopes 
            - If we do not recive a target_return value
            - If the target_return is grater than the maximal return 
            - If the targer_return is lesser than the minimal return
            '''
            epsilon = 10**-4 # To avoid calculation problems since we are working in the boundary... we will need this epsilon 
            if target_return == None:
                target_return = np.mean(self.returns)
            elif target_return < np.min(self.returns):
                target_return = np.min(self.returns) + epsilon
            elif target_return > np.max(self.returns):
                target_return = np.max(self.returns) - epsilon
            
            optimal_result = op.minimize(fun = portfolio_variance, x0 = x0, args = (self.mtx_var_cov), constraints = (l1_norm + markowitz), bounds=non_negative)
            weights = np.array(optimal_result.x)
            
        else: # Is the type is default then we will get the x0 variable -  THAT ACTUALLY IS THE EQUIWEIGHT PORTFOLIO
            portfolio_type = 'equi-weight'
            weights = np.array(x0)
            
        # Output class (defined at the last of the script)
        """
        How do we match the outcomes gotted in this class with a new one?
        We must create a object inside this class using the new one
        Notice that no matters that the new class that we want to link is defined after this one
        
        If we FIND a variable imporant to know and develop DURING the process... we could use this class to print or RETRIEVE it.
        An example is what happened with the target_return variable.
        
        
        But using the class we only need the name of the object to call these"""
        optimal_portfolio = output(self.notional, self.rics)
        optimal_portfolio.type = portfolio_type
        # Then I could use the attributes of output class 
        # In the next line of code we delete self.notional * weights since we were not getting the correct values... we were multiplyng the return by the notional which is incorrect, we should get the same value as return
        optimal_portfolio.weights =  weights / sum(abs(weights))
        # The new variables that we noticed that are important to get
        optimal_portfolio.allocate = self.notional * optimal_portfolio.weights
        optimal_portfolio.targer_return = target_return
        
        optimal_portfolio.return_annual = np.round(self.returns.dot(weights),decimals) # Both are vectors, we get a value
        optimal_portfolio.volatility_annual = np.round(np.sqrt( portfolio_variance(weights, self.mtx_var_cov) ) , decimals)
        optimal_portfolio.sharpe_ratio = optimal_portfolio.return_annual / optimal_portfolio.volatility_annual if optimal_portfolio.volatility_annual > 0.0 else 0.0
        
        return optimal_portfolio # return of this function called "compute_portfolio"
        
# We need a new class that will be the output
class output:
    def __init__(self, notional, rics):
        self.rics = rics
        self.notional = notional
        self.type = None
        self.weights = None
        """Adding new variables"""
        self.targer_return = None # We notice that a variable develop during the process was important to retrieve it and therefore, know its value computed.
        self.allocate = None
        self.targer_return = None # WE GIVE THIS VALUE
        self.return_annual = None # This is a vector, we computed it in the previous function, returns_vector^T * optimal_weights. THIS IS THE IDEAL
        self.volatility_annual = None # THIS IS THE EXPECTED VALUE
        self.sharpe_ratio = None
        



