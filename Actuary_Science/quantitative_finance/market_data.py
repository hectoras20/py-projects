
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import scipy.stats as st 

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
    df = pd.DataFrame()
    df['Date'] = raw_data['Date']# Instead of: raw_data.iloc[:, 0]
    df['Close'] = raw_data['Close']
    # Tratando los formatos 
    # DATE with the format 01.01.2025 there is not problem
    df['Date'] = df['Date'].astype(str).str.replace('/', '.', regex=False)
    # df['Close'] = df['Close'].astype(float) # ESTA SI ESTA BIEN CUANDO NO HAY CANTIDAD EN MILES (SIN COMA, SOLO PUNTOS como 23.53)
    df['Close'] = (df['Close'].astype(str).str.replace(',', '', regex=False).astype(float))
    df[['mes', 'dia', 'año']] = df['Date'].str.split('.', expand=True)
    df['año'] = df['año'].apply(corregir_año) # key: Al usar la función Apply... y usar apply(function) SE ENTIENDE QUE EL ARGUMENTO SON CADA CELDA O VALOR DE LA COLUMNA RESPECTIVA QUE SE VA A CREAR, tipo appli(function(valoresColumna))
    df['Date'] = df[['mes', 'dia', 'año']].astype(str).agg('.'.join, axis=1)
    df.drop(columns=['dia', 'mes'], inplace=True) # inplace para que el cambio sea directo
    return df


# We did not make this function a calculator method because we will use them after for other topics. We want it at hand.
def load_timeseries(ric):
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
    t = t.dropna()
    t = t.reset_index(drop=True)
    return t

def synchronise_timseries_df(security, benchmark):
    timeseries_x = load_timeseries(benchmark)
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
        