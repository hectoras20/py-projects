import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import scipy.stats as st
import scipy.optimize as op 

from pathlib import Path
import csv

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


def get_names(directory = "market_universe"):
    ruta = Path(directory) # Function from pathlib 
    # To obtain the security name from our entire universe:
    nombres = [f.stem for f in ruta.glob("*.csv")]
    return nombres

def get_correlations(specific_benchmarck = None, specific_rics = None, from_date = 'aaaa-mm-dd', to_date = 'aaaa-mm-dd', orderby = 'correlation', getCSV = False, namefile = 'correlation_output.csv'):
    """
    Compute correlations, betas and R² between one or multiple benchmarks and a
    set of securities.
    
    This method performs a cross-sectional scan of the asset universe by running
    a linear regression for every (security, benchmark) pair and storing the
    resulting statistics in a dataframe.
    
    The method supports flexible configuration:
    
    - Custom list of benchmarks
    - Custom subset of securities
    - Optional time window filtering
    - Optional CSV export of the results
    
    Conceptual note
    ---------------
    In this framework the benchmark is treated as the explanatory variable
    (X) and the security as the dependent variable (Y). Therefore the benchmark
    explains the security's returns, not necessarily the other way around.
    
    Example:
        A stock may statistically explain an index movement,
        but the index does not necessarily explain each individual stock.
    
    Parameters
    ----------
    specific_benchmarck : list | None
        List of benchmarks to evaluate. If None, the instance benchmark
        (self.benchmark) is used.
    
    specific_rics : list | None
        List of securities to analyse. If None, the full universe returned by
        get_names() is used.
    
    from_date : str
        Lower bound of the estimation window (format: 'yyyy-mm-dd').
        If not provided, the earliest available data is used.
    
    to_date : str
        Upper bound of the estimation window (format: 'yyyy-mm-dd').
        If not provided, the latest available data is used.
    
    orderbby : str
        We can get the dataframe sorted by correlation, beta or r2
        
    getCSV : bool
        If True, exports the resulting dataframe to self.namefile.
    
    namefile : str
        Indicates the file name where the correlations will be export.
    
    Output
    ------
    Results are stored in:
    
        self.allCorrelationsDf
    
    Columns
    -------
    benchmark
        Explanatory variable used in the regression.
    
    correlation
        Pearson correlation coefficient.
    
    beta
        Regression slope (sensitivity of the security to the benchmark).
    
    r2
        Coefficient of determination.
    
    security
        Asset analysed.
    
    min_date
        First observation used in the regression.
    
    max_date
        Last observation used in the regression.
    """
    if specific_rics is None:
        ric_names = get_names() # list 
    else:
        ric_names = specific_rics
        
    if specific_benchmarck is None:
        benchmark_names = get_names()
    else:
        benchmark_names = specific_benchmarck # list

    # Creationg of the CSV file with the corrolations 
    # names = [x for x in names if x not in benchmarks]
    df = pd.DataFrame(columns = ["benchmark", "correlation", "beta", "r2", 'security', 'min_date', 'max_date']) # We might extend the infomration indicated to show
    
    # O(n^2)
    for i in ric_names:
        for j in benchmark_names:
            # Getting the data
            info = model(i, j) # i are our securities 
            info.synchronise_timeseries()
            
            # Subsetting of data
            if from_date != 'aaaa-mm-dd' and to_date != 'aaaa-mm-dd':
                    subsetting = (info.timeseries['date'] >= from_date) & (info.timeseries['date'] <= to_date)
                    info.timeseries = info.timeseries.loc[subsetting].reset_index(drop=True)
            elif to_date != 'aaaa-mm-dd':
                subsetting = info.timeseries['date'] <= to_date
                info.timeseries = info.timeseries.loc[subsetting].reset_index(drop=True)
            elif from_date != 'aaaa-mm-dd':
                subsetting = info.timeseries['date'] >= from_date
                info.timeseries = info.timeseries.loc[subsetting].reset_index(drop=True)
            # else, does not make a subsetting and therefore takes all the dates loaded in the database.
            
            if info.timeseries.empty:
                print('There is a problem with the data matching with', info.security)
                continue
        
            # Visualy this will help if the asset is recently added to the market
            min_date = info.timeseries['date'].min()
            max_date = info.timeseries['date'].max()
                
            info.compute_linear_reg()
            # Filling out the dataframe with its correct order
            df.loc[len(df)] = [
                j,
                info.correlation,
                info.beta,
                info.r_squared,
                i,
                min_date,
                max_date]
            
    # Sorting the dataframe by security and correlation
    df = df.sort_values(by=['benchmark'], ascending=[True]).reset_index(drop=True)
    df = df.sort_values(by=['security'], ascending=[True]).reset_index(drop=True)
    df = df.sort_values(by=[orderby], ascending=[True]).reset_index(drop=True)
    # The case when both are None
    
    # To export it
    if getCSV == True:
        with open(namefile, "w", encoding="UTF8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(df.columns.tolist()) # Heading

            writer.writerows(df.to_numpy().tolist()) 
        print('The file ', namefile, ' was updated, if you want to get it in a new file, just call self.namefile and give it a name, it will be created automatically')
        
    return df


def plot_normalized_timeseries(
    rics,
    from_date='aaaa-mm-dd',
    to_date='aaaa-mm-dd',
    base_value=100
):
    """
    Plot normalized price time series for a group of assets.

    Prices are first filtered by the selected date window and then
    normalized so that the first observation within the window equals
    the chosen base_value.

    Price_norm = (Price_t / Price_from) * base_value
    """

    plt.figure(figsize=(12,6))

    for ric in rics:

        t = market_data.load_timeseries(ric)

        # Date filtering
        if from_date != 'aaaa-mm-dd' and to_date != 'aaaa-mm-dd':
            subsetting = (t['date'] >= from_date) & (t['date'] <= to_date)
            t = t.loc[subsetting].reset_index(drop=True)

        elif to_date != 'aaaa-mm-dd':
            subsetting = t['date'] <= to_date
            t = t.loc[subsetting].reset_index(drop=True)

        elif from_date != 'aaaa-mm-dd':
            subsetting = t['date'] >= from_date
            t = t.loc[subsetting].reset_index(drop=True)

        if t.empty:
            print("No data available for", ric)
            continue

        base = t['close'].iloc[0]
        normalized = t['close'] / base * base_value

        plt.plot(t['date'], normalized, label=ric)

    plt.title(f"Normalized Price Series (Base = {base_value})")
    plt.xlabel("Time")
    plt.ylabel(f"Price Index (Base = {base_value})")
    plt.grid(True)
    plt.legend()
    plt.show()
    
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
        
    def synchronise_timeseries(self, extremeValues = False, from_date = 'aaaa-mm-dd', to_date = 'aaaa-mm-dd', log_returns = False, period_returns = 'daily'):
        self.timeseries = market_data.synchronise_timseries_df(self.security, self.benchmark, highVolDays = extremeValues, from_date = from_date, to_date = to_date, log_returns = log_returns, period_returns = period_returns)
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






