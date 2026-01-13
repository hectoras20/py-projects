
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import scipy.stats as st 

from scipy.optimize import minimize
import math
from pathlib import Path

from datetime import datetime
import pandas as pd


# The following two functions works in cases where we use different data from different sources BUT THIS WILL DEPEND ON HOW WE GET THE INFORMATION, THIS IS ONLY AN ADAPTATION FOR A SPECIFIC DATA PRESENTATION.

def corregir_año(año): # El año serán cada valor de la columna a la cual se esté manipulando.
    año = str(año)
    if len(año) == 2:
        return '20' + año  # Asumimos que son años del 2000+
    return año

def corrector(raw_data):
    #Empezamos con la depuración en un nuevo df
    # Renombrar columnas necesarias
    df = raw_data.rename(columns={"Fecha": "Date", "Cierre": "Close"}).copy()
    # Verificación de columnas
    if not {'Date', 'Close'}.issubset(df.columns):
        raise ValueError('The necessary columns do not exist in this data')
    # Normalizar separadores de fecha
    df['Date'] = df['Date'].astype(str).str.replace('/', '.', regex=False)
    # Convertir fechas (quita hora)
    df['Date'] = pd.to_datetime(
        df['Date'],
        format='mixed',
        dayfirst=True
    ).dt.date
    # Limpiar y convertir precios
    df['Close'] = (df['Close']
        .astype(str)
        .str.replace(',', '', regex=False)
        .astype(float)
    ) # These parentheses are useful for improving code readability
    return df[['Date', 'Close']]


# We did not make this function a calculator method because we will use them after for other topics. We want it at hand.
def load_timeseries(ric, highVolDays = False): # highVolDays is an option to filter extreme values
    # directory = '/Users/hectorastudillo/py-proyects/Actuary_Science/projects/quantitative_finance/market_data_c/'
    # path = directory + ric + '.csv'
    path = 'market_universe/' + ric + '.csv'
    raw_data = pd.read_csv(path)
    # Si usamos información de Investing usamos la siguiente linea
    raw_data = corrector(raw_data)
    t = pd.DataFrame()
    t['date'] = pd.to_datetime(raw_data['Date'],format='mixed', dayfirst=True)
    t['close'] = raw_data['Close']
    t = t.sort_values(by='date', ascending=True)
    t['close_previous'] = t['close'].shift(1) # This function shift "recorrer" one cell. 
    t['return'] = t['close'] / t['close_previous'] - 1
    # Filtering the extreme values
    if highVolDays == True:
        t = t[abs(t['return']) >= 0.01]
    t = t.dropna()
    t = t.reset_index(drop=True)
    return t

def synchronise_timseries_df(security, benchmark, highVolDays = False):
    timeseries_x = load_timeseries(benchmark, highVolDays)
    timeseries_y = load_timeseries(security)

    # Los siguientes pasos son necesarios para tener mismas dimensiones en ambos activos, ya que no podemos manipularlas para la reg. lin. si son de diferente tamaño.
    timestamps_x = timeseries_x['date'].values
    timestamps_y = timeseries_y['date'].values
    # Lo que quiero hacer es la intersección de ambas stamstapms, para ello AMBAS LISTAS CREADAS DEBEN DE SER CONJUNTOS, lo hacemos con la funcións set
    timestamps = list(set(timestamps_x) & set(timestamps_y)) # Tal que el resultado de la intersección lo quiero como una lista y no como un conjunto 

    # Ahora hacemos el filtrado para obtener mismas dimensiones, esto lo logramos con Pandas
    timeseries_x = timeseries_x[timeseries_x['date'].isin(timestamps)]
    timeseries_y = timeseries_y[timeseries_y['date'].isin(timestamps)]

    # Re ordenamos ambos subconjuntos 
    timeseries_x = timeseries_x.sort_values(by='date', ascending = True)
    timeseries_y = timeseries_y.sort_values(by='date', ascending = True)

    # Re organizamos el índice de ambos subsets
    timeseries_x = timeseries_x.reset_index(drop = True)
    timeseries_y = timeseries_y.reset_index(drop = True)

    # AHORA DEBEMOS DE CREAR UN DATAFRAME QUE CONTENGA UNICAMENTE LA FECHA, EL CLOSE DE  X, EL CLOSE DE Y y EL RETURN DE AMBOS (rendimiento)
    # Pero la mejor prática es encapsular todo lo anterior dentro de una función! Para después hacer el plot.

    timeseries = pd.DataFrame()
    timeseries['date'] = timeseries_x['date']
    timeseries['close_x'] = timeseries_x['close']
    timeseries['close_y'] = timeseries_y['close']
    timeseries['return_x'] = timeseries_x['return']
    timeseries['return_y'] = timeseries_y['return']
    return timeseries

def sychronise_returns(rics):
    df = pd.DataFrame()
    dic_timeseries = {}
    timestamps = [] 
    for ric in rics:
        t = load_timeseries(ric) # t is a Dataframe
        dic_timeseries[ric] = t # Key = ric_name and Value = dataframe_ric

        if len(timestamps) == 0: # This happens in the first iteration
            timestamps = list(t['date'].values) # We took the date values of the first ric iterated
        temp_timestamps = list(t['date'].values)  # This variable serves as a comparator and will make the interection between dates, the first iteration will keep all the same values.
        timestamps = list(set(timestamps) & set(temp_timestamps)) 
        # Then of the fist iteration, the conditional if, becomes obsolet, the next comparision will be with the previous timestamps variable gotten and the new dates from the next ric allocated in temp_timestamps.

    # SYNCHRONIZATION of all the timeseries - Using the dictionary created with original dataframes, unfiltered
    for ric in dic_timeseries:
        t = dic_timeseries[ric]
        t = t[t['date'].isin(timestamps)]
        t = t.sort_values(by='date', ascending=True)
        t = t.dropna()
        t = t.reset_index(drop=True)
        dic_timeseries[ric] = t
        
        if df.shape[1] == 0:
            df['date'] = timestamps
        df[ric] = t['return']
    return df

def ppcc_distance(lambda_value, df):
    """
    Objective function for Tukey's Lambda distribution fitting using PPCC.

    Parameters
    ----------
    lambda_value : array-like
        Tukey lambda parameter (passed as an array by SciPy).

    df : pandas.DataFrame
        DataFrame containing 'FDA' and 'ranked_return'.

    Returns
    -------
    float
        Squared PPCC distance (1 - PPCC)^2.
    """
    values = []
    lambda_value = lambda_value[0] # Since scipy works with arrays (i.e we pass a lambda value as an array) and we work into our code with a lambda value, NOT as an array, we must do this.
    for i in df['FDA']:
        if lambda_value == 0:
            values.append(math.log(i / (1 - i)))
        else:
            values.append((1 / lambda_value) * (i**lambda_value - (1 - i)**lambda_value))

    tukey_quantile = np.array(values)

    PPCC = np.corrcoef(
        df['ranked_return'],
        tukey_quantile
        )[0, 1] # np.corrcoef returns a matrix, from [0,1] we are only taking the value we are interested in.

    # Queremos PPCC lo más cercano posible a 1 y eso NO es lo mismo que “maximizar PPCC” en términos prácticos de optimización.
    return (1 - PPCC)**2



def classify_lambda_distance(lmbda, tol=0.03):
    """
    Classify a Tukey lambda value into a theoretical distribution.

    Parameters
    ----------
    lmbda : float
    tol : float

    Returns
    -------
    str or numpy.nan
    """
    theoretical = {
        'Cauchy': -1.0,
        'Laplace': -0.12,
        'Hyperbolic Secant': -0.06,
        'Logistic': 0.0,
        'Normal': 0.14,
        'Uniform': 1.0
    }

    dist = {k: abs(lmbda - v) for k, v in theoretical.items()}
    best = min(dist, key=dist.get)

    return best if dist[best] <= tol else np.nan

def get_all_lambda(directory =  "market_universe", tolerance=0.03, printMetrics = False):
    """
    Estimate Tukey lambda parameters for all assets in a directory and classify them.

    Parameters
    ----------
    directory : str
    tolerance : float
    printMetrics : bool

    Returns
    -------
    pandas.DataFrame
    """
    ruta = Path(directory) # Function from pathlib 
    # To obtain the security name from the specific library:
    names = [f.stem for f in ruta.glob("*.csv")]
    
    rows = []
    for ric in names:
        ric_info = distribution_manager(ric)
        ric_info.load_timeseries()
        ric_info.tukey_quantile(tolerance, printMetrics)

        rows.append({
            'ric': ric,
            'lambda': ric_info.lambda_opt
        })

    df_tukey = pd.DataFrame(rows)
    df_tukey['tipo'] = df_tukey['lambda'].apply(classify_lambda_distance, tol = tolerance)
    return df_tukey

    
    
def get_all_kurtosis_skewness(ric, directory = 'market_universe'):
    """
    Since a characteristic of financial data include:
    1. "The empirical distributions of financial returns are leptokurtic, or in other 
    words they have “fat tails”  compared to the tails of the normal distribution (kurtosis = 3)"
        - Extreme desviations from the mean happen more frequently than one would 
        expect with the normal distribution.
        - A Kurotsis of more than 3 means that the probability distriution has fatter
        tails and a sharper peak than the normal distribution.
        - Kurtosis - qué tan FRECUENTES son los valores extremos (±) - how FREQUENT 
        are extreme values
        
    2. The empirical distribution of returns is left-skewed, that is, large negative 
    returns are possible.
        - Skewness → si hay más positivos o más negativos - whether there are 
        more positive or more negative positives
    
    We include this function to obtain all the kurotsis and skewness values of our entire universe of securities.
    
    **** The skewness does not work with the annual mean, works with the "normal/raw" 
    mean and median of the returns, self.vector in the class 
    """
    ruta = Path(directory) # Function from pathlib 
    # To obtain the security name from the specific library:
    names = [f.stem for f in ruta.glob("*.csv")]
    list_df = [load_timeseries(i) for i in names]
    df = pd.DataFrame({'ric': names,
                         '%median' : [np.median(i['return'])*100 for i in list_df],
                         '%mean' : [st.tmean(i['return'])*100 for i in list_df],
                         'kurtosis': [st.kurtosis(i['return']) for i in list_df],
                         'skewness': [st.skew(i['return']) for i in list_df],
                         'sample-size': [i['return'].size for i in list_df],
                         'from': [min(i['date']) for i in list_df],
                         'up to': [max(i['date']) for i in list_df]
                        })
    df['from'] = pd.to_datetime(df['from']).dt.date
    df['to']   = pd.to_datetime(df['to']).dt.date
    # The following constraints does not worth with actual financial data, are useful to theorical topics and understand how the distributions are relative to the normal distribution and that is it.
    # df = df[df['kurtosis']<=3] # A normal distribution has a kurtosis equal to 3
    # df = df[df['skewness']>0] # A normal distribution has a skewness equal to 0, which means it is symmetrical.
    df = df.sort_values(
        by=["kurtosis", "skewness"],
        ascending=[True, False]).reset_index(drop=True)
    return df
    
    
    
    
class distribution_manager:
    def __init__(self, ric, decimals = 5):
        self.ric = ric
        self.decimals = decimals
        self.timeseries = None
        self.str_title = None
        self.vector = None
        self.mean_annual = None
        self.volatility_annual = None
        self.sharpe_ratio = None
        self.var_95 = None
        self.skewness = None
        # self.kurtosis = st.kurtosis(self.vector) IS NOT A BEST PRACTICE, WE MUST NOT DO THIS!
        self.kurtosis = None
        self.jb_stat = None
        self.p_value = None
        self.is_normal = None
        # self.cv = None - Is not recomended use it in return since the return´s mean is near from 0.
        self.lambda_opt = None
        self.ppcc_opt = None
        self.distribution_opt = None
        
    # First method to load the timeserie of the asset, using the isolated function previusly created now is used here.
    def load_timeseries(self):
        """
        We create our timeserie with the isolated function that we create into this script.
        In this functions we get the vector that contains real data, is our random variable
        """
        self.timeseries = load_timeseries(self.ric)
        self.vector = self.timeseries['return'].values
        self.size = len(self.vector)
        self.str_title = self.ric + ' | real data'
        
    def plot_timeseries(self):
        plt.figure()
        self.timeseries.plot(kind ='line', x='date', y = 'close', grid = True, title='Timeseries of close prices for ' + self.ric)
        plt.show()
        
    def compute_stats(self, factor = 252):
        """
        factor is equal to the number of days which the market of the asset is open.
        factor = 252 for indexes and other assets
        factor = 360 for cripto"""
        self.mean_annual = st.tmean(self.vector) * factor
        self.volatility_annual = st.tstd(self.vector) * np.sqrt(factor)
        self.sharpe_ratio = self.mean_annual / self.volatility_annual if self.volatility_annual > 0 else 0.0
        self.var_95 = np.percentile(self.vector, 5)
        self.skewness = st.skew(self.vector)
        self.kurtosis = st.kurtosis(self.vector)
        self.jb_stat = self.size/6 * (self.skewness**2 + 1/4*self.kurtosis**2)
        self.p_value = 1 - st.chi2.cdf(self.jb_stat, df = 2) # In other words:  = 1 - P(X ≤ jb_stat) =  P(X > jb_stat)
        self.is_normal = (self.p_value > 0.5)
        
        
    def plot(self):
        self.str_title += '\n' + 'mean annual=' + str(np.round(self.mean_annual, self.decimals)) \
            + ' | ' + 'volatility annual=' + str(np.round(self.volatility_annual, self.decimals)) \
            + '\n' + 'Sharpe ratio=' + str(np.round(self.sharpe_ratio, self.decimals)) \
            + ' | ' + 'var_95=' + str(np.round(self.var_95, self.decimals)) \
            + '\n' + 'skewness=' + str(np.round(self.skewness, self.decimals)) \
            + ' | ' + 'kurtosis=' + str(np.round(self.kurtosis, self.decimals)) \
            + '\n' + 'JB stat=' + str(np.round (self.jb_stat, self.decimals)) \
            + ' | ' + 'p-value=' + str(np.round (self.p_value, self.decimals)) \
            + '\n' + 'is _normal=' + str(self.is_normal)
            
            # + ' | ' + 'cv=' + str(np.round (self.cv, self.decimals)) \
            # + '\n' + ' =' + 
            
        plt.figure()
        plt.hist(self.vector, bins=100)
        plt.title(self.str_title)
        plt.show()
        
    # Lambda distribution
    def tukey_quantile(self, tolerance = 0.3, printMetrics = True):
        df = pd.DataFrame()
        vector = np.array(self.vector)
        df['rank'] = np.arange(1, len(vector) +1) # +1 since the last number to be indicated in the range is excluded.
        df['ranked_return'] = np.sort(vector)
        df['FDA'] = df['rank'] / max(df['rank'] +1) # +1 to avoid indeterminations
        # lambda_value = lambda_value[0]  # scipy always recive arrays BUT in this case we want to GET the optimize lambda, and not give it.
        initial_lambdas = [-1.0, -0.12, -0.06, 0.0, 0.14, 0.5, 1.0]

        results = []

        for x0 in initial_lambdas:
            res = minimize(fun = ppcc_distance, x0=[x0], args=(df), method='Nelder-Mead')
            results.append(res)

        best_result = min(results, key=lambda r: r.fun)
        # Cada r es un objeto resultado de optimización, osea de la lista results
        # r es un objeto con varios atributos... scipy.optimize.OptimizeResult y .fun es un atributo del objeto OptimizeResult, entre otros más atributos.
        # Usamos min ya que mientras más pequeño r.fun, mejor el ajuste

        self.lambda_opt = best_result.x[0]
        # si el min r.fun = 0.0004
        # Entonces (1 - PPCC)**2 = 0.0004
        # Despejando... PPCC = 1 - sqrt(0.0004) = 0.98
        self.ppcc_opt = 1 - np.sqrt(best_result.fun)
        self.distribution_opt = classify_lambda_distance(self.lambda_opt , tol=tolerance)
        if printMetrics == True:
            print(self.ric, ' has the following data:')
            print("Optimal Lambda Obtained:", self.lambda_opt)
            print("PPCC obtained:", self.ppcc_opt)
            print("Close to a ", self.distribution_opt, ' distribution.')
        
   
    
        
        
        
        
        
        
        
        
        