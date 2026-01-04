# LET'S THINK LIKE OUR CUSTOMERS!!!
import importlib

import market_data
importlib.reload(market_data)

import capm
importlib.reload(capm)

import portfolio_class_final
importlib.reload(portfolio_class_final)


'''
It’s time to think like our customers... What would they need?
'''

'''
PART 1

I would like to invest in some assets, but I need to know about their metrics and timeseries graph!. 
Which metrics can I guarantee or recommend that you should analyze?

The code is adapted if the customer needs an specific metric that there is not into the code, we could add it easily.

We can achieve this with the "market_data" class.

WARNING: 
   In this case, we assume that we are working with the uploaded data in the repository. 
   However, it can be easily adapted for ANY dataset — this will depend on the company’s data presentation format.
'''

# Let's plot the information about our asset or universe:

ric = ''
# For a given list (an easier way)
# rics = ['GFNORTE', 'USDMXN']

# We can achieve this with the "market_data" class
ric_info = market_data.distribution_manager(ric)

# Getting the asset information.
# Get its graph
ric_info.load_timeseries()
ric_info.plot_timeseries()
# Get its metrics
ric_info.compute_stats()
ric_info.plot()

'''for i in rics:
    ric_info = market_data.distribution_manager(i)
    # Getting its information.
    ric_info.load_timeseries()
    ric_info.plot_timeseries()
    ric_info.compute_stats()
    ric_info.plot()'''

'''
PART 2

What if I want to compare assets in terms of their correlation and how one might affect the other? 
This could be very useful for building the client’s investment strategy.

We achieve this with the 'capm' class creaeting a linear regression
'''
'''ric = 'AMZN'
benchmark = '^SPX'
correlation = capm.model(ric, benchmark)

correlation.synchronise_timeseries()
correlation.compute_linear_reg()
correlation.plot_linear_reg()'''

'''
And what if we want to see the graph of both assets into the same plot?
We achieve it with the same class (capm)
'''
security = 'GCARSOA1'
benchmark = 'USDMXN'

summary = capm.model(security, benchmark)
summary.synchronise_timeseries() 
summary.plot_timeseries()
summary.compute_linear_reg() 
summary.plot_linear_reg()

# Sample: To see if the correlations keeps in historical time 
security = 'GCARSOA12012-15'
benchmark = 'USDMXN2012-15'

sample = capm.model(security, benchmark)
sample.synchronise_timeseries() 
sample.plot_timeseries()
sample.compute_linear_reg() 
sample.plot_linear_reg()

'''
I want to streamline the process to get the correlations, what if...
I want to see all the correlations of my universe of assets in regard to a specific benchmark...
And export it to a file
'''

# Dealing with the original format
'''import pandas as pd
name = 'MARA'
ruta = 'market_universe/' + name + ".csv"
df = pd.read_csv(ruta)
df = df.rename(columns={"Fecha": "Date", "Cierre": "Close"})
df.to_csv(
    ruta,
    index=False,
    encoding="UTF8"
)'''


from pathlib import Path

ruta = Path("market_universe")

nombres = [f.stem for f in ruta.glob("*.csv")]


import pandas as pd

# El siguiente bloque de código dependerá de como estan presentados los datos en crudo al momento de obtenerlos, en este caso al ser obtenido de Investing se manipulan de la siguiente manera
for i in nombres:
    ruta = 'market_universe/' + i + ".csv"
    df = pd.read_csv(ruta)
    df = df.rename(columns={"Fecha": "Date", "Cierre": "Close"})
    # CHECK
    if 'Date' and 'Close' not in df.columns:
        print(i)
    df.to_csv(
        ruta,
        index=False,
        encoding="UTF8"
    )

benchmarks = ['USDMXN', 'USDEUR', 'DJI', 'SPX', 'NASDAQ', 'MSCI_W', 'MSCI_EM', 'VIX', 'BRENT', 'MXX', 'inflationusa', 'inflationmex', 'XAU_USD', 'SOX', 'XLV']

# nombres = [x for x in nombres if x not in benchmarks]

df = pd.DataFrame(columns = ["security", "correlation", "beta", "r2", 'benchmark'])
for j in benchmarks:
    for i in nombres:
        benchmark = j
        security = i
        info = capm.model(security, benchmark)
        info.synchronise_timeseries()
        info.compute_linear_reg()
        df.loc[len(df)] = [
            i,
            info.correlation,
            info.beta,
            info.r_squared,
            j
        ]

df = df.sort_values(
    by=["benchmark", "correlation"],
    ascending=[False, False]
).reset_index(drop=True)


import csv
archivo = "correlation_output.csv"
with open(archivo, "w", encoding="UTF8", newline="") as file:
    writer = csv.writer(file)

    # encabezado
    writer.writerow(df.columns.tolist())

    # múltiples líneas
    writer.writerows(df.to_numpy().tolist())


"""
PORTFOLIO
How much should I invest in each asset?
Be careful with leveraged assets! Instead of this, use the base asset.
"""
rics = ['SOX', 'VIX']
notional = 5000
portfolio = portfolio_class_final.manager(rics, notional)

portfolio.compute_covariance()
long_port = portfolio.compute_portfolio('long_only')
marko_port = portfolio.compute_portfolio('markowitz', 0.20)
long_allocate = long_port.allocate
marko_allocate = marko_port.allocate

"""What if i want to see with which benchmark a security is more explained?
We have two ways to get it 
1. Do again the calculatiosn fixing the security and runing into all the reamining securities but this means do again what does the previous code again
2. Extract/filter the first column by the name of the security and sort it by correlation or another metric
We will do the second one as follows
"""

ric_corr = ''
df_security = df[df['security'] == ric_corr].sort_values(by=["correlation"], ascending=False).reset_index(drop=True)

"""
The will be an extra manipulation for USDMXN's information...
We will make the linear regression with filtered data
In this case we will take the days with a return grater than or equals to 1% = 0.01...

|USD return| > 1%

* La mayoría de los días el USD se mueve poco
En esos días:
- GCARSO se mueve por sus propios factores
- La relación se diluye → R² bajo
Pero eso NO significa que no haya edge
Significa que el edge aparece solo en días especiales

We will add the following code later as a method OR simply modifing the CAPM´s methods in its respective class.

TO MAKE A BETTTER CODE WE MUST USE PERCENTILES INSTEAD OF A FIXED PERCENTAJE AS 1% TO MAKE THE TESTING IN HIGH VOLATILITY DAYS
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import scipy.stats as st 

from datetime import datetime
import pandas as pd

ric = 'GFNORTEO'
benchmark = 'USDMXN'
def load_timeseries(ric, highVolDays = False):
    # directory = '/Users/hectorastudillo/py-proyects/Actuary_Science/projects/quantitative_finance/market_data_c/'
    # path = directory + ric + '.csv'
    path = 'market_universe/' + ric + '.csv'
    raw_data = pd.read_csv(path)
    # Si usamos información de Investing usamos la siguiente linea
    # raw_data = corrector(raw_data)
    t = pd.DataFrame()
    t['date'] = pd.to_datetime(raw_data['Date'],format='mixed', dayfirst=True)
    t['close'] = raw_data['Close']
    t = t.sort_values(by='date', ascending=True)
    t['close_previous'] = t['close'].shift(1) # This function shift "recorrer" one cell. 
    t['return'] = t['close'] / t['close_previous'] - 1
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

df = synchronise_timseries_df(ric, benchmark, True)

def compute_linear_reg(timeseries, decimals = 2):
    # Lineal Regression 
    x = timeseries['return_x'].values
    y = timeseries['return_y'].values
    slope_beta, intercept_alpha, correl_r, p_value, standard_error = st.linregress(x, y)
    beta = np.round(slope_beta, decimals)
    alpha = np.round(intercept_alpha, decimals)
    p_value = np.round(p_value, decimals)
    correlation = np.round(correl_r, decimals)
    r_squared = np.round(correl_r**2, decimals)
    hypothesis_null = p_value > 0.5
    predictor_linreg = intercept_alpha + slope_beta * x
    # PLOT
    str_self = 'Linear regression | security ' + ric \
        + ' | benchmark ' + benchmark + '\n' \
        + 'alpha ' + str(alpha) \
        + ' | beta (slope) ' + str(beta)  + '\n' \
        + 'p-value ' + str(p_value) \
        + ' | null-hypothesis ' + str(hypothesis_null) + '\n' \
        + 'correl (r-value) ' + str(correlation) \
        + ' | r-squared ' + str(r_squared)
    str_title = 'Scatterplot of returns ' + '\n' + str_self
    # plt.figure(figsize=(10,10))
    plt.title(str_title)
    plt.scatter(x, y)
    plt.plot(x, predictor_linreg, color='green' )
    plt.ylabel(ric) 
    plt.xlabel(benchmark) 
    plt.grid()
    plt.show()
    
compute_linear_reg(df)