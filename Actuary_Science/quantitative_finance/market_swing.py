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
    
    df['Close'] = df['Close'].astype(str).str.replace(',', '.', regex=False)
    df['Close'] = df['Close'].astype(float)
    df[['mes', 'dia', 'año']] = df['Date'].str.split('.', expand=True)
    df['año'] = df['año'].apply(corregir_año) # key: Al usar la función Apply... y usar apply(function) SE ENTIENDE QUE EL ARGUMENTO SON CADA CELDA O VALOR DE LA COLUMNA RESPECTIVA QUE SE VA A CREAR, tipo appli(function(valoresColumna))
    df['Date'] = df[['mes', 'dia', 'año']].astype(str).agg('.'.join, axis=1)
    df.drop(columns=['dia', 'mes'], inplace=True) # inplace para que el cambio sea directo
    return df


# We did not make this function a calculator method because we will use them after for other topics. We want it at hand.
def load_timeseries(ric):
    # directory = '/Users/hectorastudillo/py-proyects/Actuary_Science/projects/quantitative_finance/market_data_c/'
    # path = directory + ric + '.csv'
    path = 'market_data_c/' + ric + '.csv'
    raw_data = pd.read_csv(path)
    # Si usamos información de Investing usamos la siguiente linea
    # raw_data = corrector(raw_data)
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
        t = load_timeseries(ric)
        dic_timeseries[ric] = t

        if len(timestamps) == 0:
            timestamps = list(t['date'].values) 
        temp_timestamps = list(t['date'].values)  
        timestamps = list(set(timestamps) & set(temp_timestamps))

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
        self.size = None
        
        # --- Swing Trading Metrics ---
        self.vol_20d = None
        self.sharpe_20d = None
        self.var_5d = None
        self.cvar_5d = None
        self.atr_14 = None
        self.max_drawdown = None
        
        # Distribution shape (sí útil para swing)
        self.skewness = None
        self.kurtosis = None
        
        # Normality test (opcional pero lo mantenemos)
        self.jb_stat = None
        self.p_value = None
        self.is_normal = None

    def load_timeseries(self):
        """
        We create our timeserie with the isolated function that we create into this script.
        In this function we get the vector that contains real data, is our random variable.
        """
        self.timeseries = load_timeseries(self.ric)
        self.vector = self.timeseries['return'].values
        self.size = len(self.vector)
        self.str_title = self.ric + ' | real data'

    def plot_timeseries(self):
        plt.figure()
        self.timeseries.plot(kind='line', x='date', y='close',
                             grid=True,
                             title='Timeseries of close prices for ' + self.ric)
        plt.show()

    def compute_stats(self):
        """
        Swing trading metrics: we do not annualize.
        We focus on short–mid horizon risk measures.
        """

        # --- Volatility 20-day rolling (último valor) ---
        self.vol_20d = pd.Series(self.vector).rolling(20).std().iloc[-1]

        # --- Sharpe 20 días (no anualizado) ---
        mean_20d = pd.Series(self.vector).rolling(20).mean().iloc[-1]
        self.sharpe_20d = mean_20d / self.vol_20d if self.vol_20d > 0 else 0.0

        # --- VaR y CVaR a 5 días ---
        self.var_5d = np.percentile(self.vector[-5:], 5)
        losses_5d = np.sort(self.vector[-5:])
        self.cvar_5d = losses_5d[losses_5d <= self.var_5d].mean() if np.any(losses_5d <= self.var_5d) else self.var_5d

        # --- ATR 14 días ---
        df = self.timeseries.copy()
        df['previous_close'] = df['close'].shift(1)
        df['true_range'] = np.maximum.reduce([
            df['close'] - df['close'].shift(1),
            abs(df['close'] - df['close'].shift(1)),
            abs(df['close'] - df['previous_close'])
        ])
        self.atr_14 = df['true_range'].rolling(14).mean().iloc[-1]

        # --- Drawdown máximo ---
        cumulative = (1 + pd.Series(self.vector)).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        self.max_drawdown = drawdown.min()

        # --- Distribution shape ---
        self.skewness = st.skew(self.vector)
        self.kurtosis = st.kurtosis(self.vector)

        # --- Normality test ---
        self.jb_stat = self.size/6 * (self.skewness**2 + (1/4)*self.kurtosis**2)
        self.p_value = 1 - st.chi2.cdf(self.jb_stat, df = 2)
        self.is_normal = (self.p_value > 0.5)

    def plot(self):
        self.str_title += '\n' + 'vol_20d=' + str(np.round(self.vol_20d, self.decimals)) \
            + ' | ' + 'sharpe_20d=' + str(np.round(self.sharpe_20d, self.decimals)) \
            + '\n' + 'var_5d=' + str(np.round(self.var_5d, self.decimals)) \
            + ' | ' + 'cvar_5d=' + str(np.round(self.cvar_5d, self.decimals)) \
            + '\n' + 'atr_14=' + str(np.round(self.atr_14, self.decimals)) \
            + ' | ' + 'max_drawdown=' + str(np.round(self.max_drawdown, self.decimals)) \
            + '\n' + 'skewness=' + str(np.round(self.skewness, self.decimals)) \
            + ' | ' + 'kurtosis=' + str(np.round(self.kurtosis, self.decimals)) \
            + '\n' + 'JB stat=' + str(np.round(self.jb_stat, self.decimals)) \
            + ' | ' + 'p-value=' + str(np.round(self.p_value, self.decimals)) \
            + '\n' + 'is_normal=' + str(self.is_normal)

        plt.figure()
        plt.hist(self.vector, bins=100)
        plt.title(self.str_title)
        plt.show()
