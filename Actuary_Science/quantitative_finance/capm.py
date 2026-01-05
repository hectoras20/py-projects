import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import scipy.stats as st
import scipy.optimize as op 

import market_data
importlib.reload(market_data)


def compute_beta(security, benchmark):
    m = model(security, benchmark)
    m.synchronise_timeseries()
    m.compute_linear_reg()
    return m.beta

def compute_correlation(security, benchmark):
    m = model(security, benchmark)
    m.synchronise_timeseries()
    m.compute_linear_reg()
    return m.correlation

def dataframe_correl_beta(position_security, benchmark, hedge_universe):
    decimals = 5
    df = pd.DataFrame()
    correlations = []
    betas = []
    for hedge_security in hedge_universe:
        correlation = compute_correlation(position_security, hedge_security)
        beta = compute_beta(hedge_security, benchmark)
        correlations.append(correlation)
        betas.append(beta)
    df['hedge security'] = hedge_universe
    df['correlation'] = correlations
    df['beta'] = betas
    df = df.sort_values(by='correlation', ascending=False)
    return df

def dataframe_factors(security, factors):
    decimals = 5
    df = pd.DataFrame()
    correlations = []
    betas = []
    for factor in factors: #THESE ARE THE NEW BENCHMARKS
        correlation = compute_correlation(security, factor)
        beta = compute_beta(security, factor)
        correlations.append(correlation)
        betas.append(beta)
    df['factors'] = factors
    df['correlation'] = correlations
    df['beta'] = betas
    df = df.sort_values(by='correlation', ascending=False)
    return df
        
    
def cost_function(x, betas, target_delta, target_beta, regularisation):
    dimensions = len(x)
    deltas = np.ones(dimensions)
    f_delta = (np.transpose(deltas).dot(x).item() + target_delta)**2 
    f_beta = (np.transpose(betas).dot(x).item() + target_beta)**2 
    f_penalty = regularisation * (np.sum(x**2))
    f = f_delta + f_beta + f_penalty
    return f


class model:
    def __init__(self, security, benchmark, decimals = 5):
        self.security = security
        self.benchmark = benchmark
        self.decimals = decimals
        self.timeseries = None
        self.x = None
        self.y = None
        self.beta = None
        self.alpha = None
        self.p_value = None
        self.correlation = None
        self.r_squared = None
        self.hypothesis_null = None
        self.predictor_linreg = None
        
    def synchronise_timeseries(self):
        self.timeseries = market_data.synchronise_timseries_df(self.security, self.benchmark)
        if self.timeseries.empty:
            print('There is a problem with ', self.security, ' and ', self.benchmark, '. There is not information to match')
        
    def plot_timeseries(self):
        plt.figure(figsize=(12,5))
        plt.title('Timeseries of Close Prices')
        plt.xlabel( 'Time')
        plt.ylabel( 'Prices')
        ax = plt.gca()
        ax1 = self.timeseries.plot(kind='line', x='date', y='close_x', ax=ax, grid=True, color='blue', label=self.benchmark)
        ax2 = self.timeseries.plot(kind='line', x='date', y='close_y' , color='red', secondary_y=True, ax=ax, grid=True, label=self.security)
        ax1.legend(loc=2)
        ax2.legend(loc=1)
        plt.show()
        
    def compute_linear_reg(self):
        # Lineal Regression 
        self.x = self.timeseries['return_x'].values
        self.y = self.timeseries['return_y'].values
        slope_beta, intercept_alpha, correl_r, p_value, standard_error = st.linregress(self.x, self.y)
        self.beta = np.round(slope_beta, self.decimals)
        self.alpha = np.round(intercept_alpha, self.decimals)
        self.p_value = np.round(p_value, self.decimals)
        self.correlation = np.round(correl_r, self.decimals)
        self.r_squared = np.round(correl_r**2, self.decimals)
        self.hypothesis_null = p_value > 0.5
        self.predictor_linreg = intercept_alpha + slope_beta * self.x
        
    def plot_linear_reg(self):
        str_self = 'Linear regression | security ' + self.security \
            + ' | benchmark ' + self.benchmark + '\n' \
            + 'alpha ' + str(self.alpha) \
            + ' | beta (slope) ' + str(self.beta)  + '\n' \
            + 'p-value ' + str(self.p_value) \
            + ' | null-hypothesis ' + str(self.hypothesis_null) + '\n' \
            + 'correl (r-value) ' + str(self.correlation) \
            + ' | r-squared ' + str(self.r_squared)
        str_title = 'Scatterplot of returns ' + '\n' + str_self
        # plt.figure(figsize=(10,10))
        plt.title(str_title)
        plt.scatter(self.x, self.y)
        plt.plot(self.x, self.predictor_linreg, color='green' )
        plt.ylabel(self.security) 
        plt.xlabel(self.benchmark) 
        plt.grid()
        plt.show()
        

class hedge:
    def __init__(self, position_security, position_delta_usd, benchmark, hedge_securities):
        self.position_security = position_security # Name of the assets intended for liquidity sourcing / "Name of the asset that will absorb liquidity."
        self.position_delta_usd = position_delta_usd # Amount absorbed from the asset (POSITIVE)
        self.benchmark = benchmark # Asset that will be used for comparision.
        # The following two elements are still associated to the absorbed asset.
        self.position_beta = None 
        # "Posición en dólares escalada según el beta del activo" - USD position scaled by beta asset
        self.position_beta_usd = None # Amount absorbed (S_0 = position_delta_USD) multiplied by the 'asset return porcentage '(BETA_0) explained by the benckmark
        # Now the following elements are for the hedge
        self.hedge_securities = hedge_securities # 1, 2 - Remember that this model is limited to two assets to ensure a unique solution; at least 2 assets are required.
        self.hedge_betas = [] # B_1, B_2
        self.hedge_weights = None # S_1, S_2
        self.hedge_delta_usd = None # Amount required to reach a neutral delta (NEGATIVE)
        self.hedge_beta_usd = None # Amount required to get a neutral beta
        
    def compute_betas(self):
        self.position_beta = compute_beta(self.position_security, self.benchmark)
        self.position_beta_usd = self.position_beta * self.position_delta_usd
        for security in self.hedge_securities:
            beta = compute_beta(security, self.benchmark)
            self.hedge_betas.append(beta)
        
    def compute_hedge_weights(self, regularisation=0):
        """
        Estás encontrando los pesos óptimos x de cobertura que minimizan el riesgo (medido como distancia al delta y beta neutrales), y para ello necesitas una condición inicial x0 que guíe al optimizador.
        """
        # scipy.optimize.minimize necesita un punto de partida desde el cual comenzar la búsqueda del mínimo.
        x0 = - self.position_delta_usd / len(self.hedge_betas) * np.ones(len(self.hedge_betas)) # the original code is: * np.ones([len(betas), 1])
        # Into the original script, the multiplication is by len(self.hedge_betas)
        optimal_result = op.minimize(fun = cost_function, x0 = x0,\
                                     args = (self.hedge_betas, self.position_delta_usd, self.position_beta_usd , regularisation))
        self.hedge_weights = optimal_result.x
        self.hedge_beta_usd = np.sum(self.hedge_weights)
        self.hedge_delta_usd = np.transpose(self.hedge_betas).dot(self.hedge_weights).item
        
        
    def compute_hedge_weights_model1(self):
        # we create our matrix, starting with the vectors (be careful, AS the vectors could be columns or rows)
        v_deltas = np.ones(len(self.hedge_securities)) # COLUMN vector
        v_betas = self.hedge_betas # COLUMN vector
        # To handle the matrix orientation, we transpose it.
        mtx = np.transpose(np.column_stack([v_deltas, v_betas]))
        # Minus Positions (targets to achieve a neutral delta and beta)
        targets = -np.array([[self.position_beta_usd], [self.position_delta_usd]]) # key: This must be a matrix whose rows are the positions delta and beta dollars
        # Hedge values
        self.hedge_weights = np.linalg.inv(mtx).dot(targets) # sublibrary - linalg, WE NEED INVERT THE MATRIX so that is why we use the function "inv" and then we multiply it by the targets
        self.hedge_beta_usd = np.sum(self.hedge_weights)
        self.hedge_delta_usd = np.transpose(self.hedge_betas).dot(self.hedge_weights).item






