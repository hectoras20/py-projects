# LET'S THINK LIKE OUR CUSTOMERS!!!
import importlib

import market_data
importlib.reload(market_data)

import capm
importlib.reload(capm)

import portfolio_class_final
importlib.reload(portfolio_class_final)

import csv
from pathlib import Path
import pandas as pd


'''
It’s time to think like our customers... What would they need?
Firstly I would like to analyze how the assets in my universe relate to a different probability distributions
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
ric = 'USDMXN'

# We can achieve this with the "market_data" class
ric_info = market_data.distribution_manager(ric)

# Get the asset information.
# Load information
ric_info.load_timeseries()
# Get its possible distribution
ric_info.tukey_quantile(tolerance = 0.05)
# Get its graph
ric_info.plot_timeseries()
# Get its metrics
ric_info.compute_stats()
ric_info.plot()

'''
# For a given list (an easier way)
# rics = ['GFNORTE', 'USDMXN']
for i in rics:
    ric_info = market_data.distribution_manager(i)
    # Getting its information.
    ric_info.load_timeseries()
    ric_info.plot_timeseries()
    ric_info.compute_stats()
    ric_info.plot()'''
    
    
################################################################################################################################################################

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
security = 'MXX'
benchmark = 'BIMBOA'


summary = capm.model(security, benchmark, 6)
summary.synchronise_timeseries(extremeValues = False) 
summary.plot_timeseries()
summary.compute_linear_reg() 
summary.plot_linear_reg()
summary.get_all_correlations()

# Sample: To see if the correlations keeps in historical time 
security = 'GCARSOA12012-15'
benchmark = 'USDMXN2012-15'

sample = capm.model(security, benchmark)
sample.synchronise_timeseries() 
sample.plot_timeseries()
sample.compute_linear_reg() 
sample.plot_linear_reg()



################################################################################################################################################################

'''
PART 3
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


ruta = Path("market_universe")

nombres = [f.stem for f in ruta.glob("*.csv")]


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

benchmarks = ['BIMBOA'] # ['USDMXN', 'USDEUR', 'DJI', 'SPX', 'NASDAQ', 'VIX', 'BRENT', 'MXX', 'SOX', 'XLV']

# nombres = [x for x in nombres if x not in benchmarks]

df = pd.DataFrame(columns = ["security", "correlation", "beta", "r2", 'benchmark'])
for j in benchmarks:
    for i in nombres:
        benchmark = j
        security = i
        info = capm.model(security, benchmark)
        info.synchronise_timeseries()
        if info.timeseries.empty:
            print('There is a problem with ', info.security)
            continue
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

"""What if i want to see with which benchmark explain more a security?
We have two ways to get it 
1. Do again the calculatiosn fixing the security and runing into all the reamining securities but this means do again what does the previous code again
2. Extract/filter the first column by the name of the security and sort it by correlation or another metric
We will do the second one as follows
"""

ric_corr = 'BIMBOA'
df_security = df[df['security'] == ric_corr].sort_values(by=["correlation"], ascending=False).reset_index(drop=True)


################################################################################################################################################################

"""
There will be an extra manipulation for USDMXN's information...
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

ric = 'BIMBOA'
benchmark = 'MXX'
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
#### Calculating returns per security
def load_timeseries(ric, highVolDays = False):
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
    if highVolDays == True:
        t = t[abs(t['return']) >= 0.01]
    t = t.dropna()
    t = t.reset_index(drop=True)
    return t
t = load_timeseries(benchmark)

def synchronise_timseries_df(security, benchmark, highVolDays = False):
    timeseries_x = load_timeseries(benchmark, highVolDays) # We set here the option because the benchmark leads the asset´s performance.
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
'''Ends the USDMXN manage
That can ve extended '''

####################################################################################################################################









####################################################################################################################################
#### Let's get the Tukey's lambda distribution of each security
# We will try to get the security distribution with the Tukey statements
ric = 'BBVA'
df_info = load_timeseries(ric)
df_info['Rank'] = np.arange(1, len(df_info['return']) +1) 
df_info['Ranked_Return'] = np.sort(np.array(df_info['return']))
df_info['FDA'] = df_info['Rank'] / max(df_info['Rank'] +1)

# =SI(lambda=0, LN(p/(1-p)), (1/lambda)  * (p^lambda - (1-p)^lambda))
import math
values = []
lambda_value = 0.14
for i in df_info['FDA']:
    if lambda_value == 0:
        values.append(math.log(i/(1-i)))
    else:
        values.append((1/lambda_value) * (i**lambda_value - (1-i)**lambda_value))
df_info['Tukey_quantile'] = values
PPCC = np.corrcoef(df_info['Ranked_Return'], df_info['Tukey_quantile'])

####### Finding the lambda value that leads to the most correlation with scipy
import numpy as np
import math

lambda_value = [0.14]
def ppcc_distance(lambda_value, df_info):
    lambda_value = lambda_value[0]  # scipy siempre pasa arrays

    values = []
    for i in df_info['FDA']:
        if lambda_value == 0:
            values.append(math.log(i / (1 - i)))
        else:
            values.append((1 / lambda_value) * (i**lambda_value - (1 - i)**lambda_value))

    tukey_quantile = np.array(values)

    PPCC = np.corrcoef(
        df_info['Ranked_Return'],
        tukey_quantile
    )[0, 1] # Recuerda que corrcoef devuelve una matriz y simplemente estamos tomando un valor que nos interesa con indexación 

    # Queremos PPCC lo más cercano posible a 1 y eso NO es lo mismo que “maximizar PPCC” en términos prácticos de optimización.
    return (1 - PPCC)**2
    # Distancia matemática
    # La forma más simple de medir “qué tan lejos está PPCC de 1” es: distancia=∣1−PPCC∣
    # Y elevar al cuadrado Evita valores negativos, penaliza más los errores grandes y es suave para optimización, por eso en estadística y ML se usa error cuadrático.
    # KEY: “Quiero que PPCC = 1” ENTONCES “Minimizo la distancia al cuadrado respecto a 1”

min_distance_to_opt = ppcc_distance(lambda_value, df_info)

#Solver
from scipy.optimize import minimize


'''result = minimize(
    ppcc_distance,  # siempre requiere de una función previa para funcionar 
    x0=[-0.14],              # valor inicial - es dónde empiezas a explorar la geometría del PPCC
    # Ya que usar Normal como punto neutroes muy común en práctica cuantitativa por punto “equilibrado”, muy estable numéricamente y fácil interpretación
    args=(df_info,),    # simplemente pasamos los argumetnso de la función de arriba, es simple estructura de scipy que sigue 
    method='Nelder-Mead'     # no necesita derivadas, ideal aquí, y proviene de scipy predeterminadamente, pero podemos crear nuestros propios métodos ajustados 
)
    # Resultado óptimo
lambda_opt = result.x[0]
ppcc_opt = 1 - np.sqrt(result.fun)
    
# Referente a x_0 como valor inicial, si queremos algo simple y sólido: 
    # x0 = [0.0] logística como centro
    # Algo robusto y serio: Multi-start con tu tabla
    # Algo óptimo y limpio: differential_evolution en [-1, 1]
    
# Al final obtenemos el mismo valor optimo para lambda con el siguiente código
# Esto depende de un enfoque quant'''

initial_lambdas = [-1.0, -0.12, -0.06, 0.0, 0.14, 0.5, 1.0]

results = []

for x0 in initial_lambdas:
    res = minimize(
        ppcc_distance, # The user-provided objective function must return a scalar value.
        x0=[x0],
        args=(df_info,),
        method='Nelder-Mead'
    )
    results.append(res)

best_result = min(results, key=lambda r: r.fun)
# Cada r es un objeto resultado de optimización, osea de la lista results
# r es un objeto con varios atributos... scipy.optimize.OptimizeResult y .fun es un atributo del objeto OptimizeResult, entre otros más atributos.
# Usamos min ya que mientras más pequeño r.fun, mejor el ajuste

lambda_opt = best_result.x[0]
# si el min r.fun = 0.0004
# Entonces (1 - PPCC)**2 = 0.0004
# Despejando... PPCC = 1 - sqrt(0.0004) = 0.98
ppcc_opt = 1 - np.sqrt(best_result.fun)


print("Lambda óptimo:", lambda_opt)
print("PPCC alcanzado:", ppcc_opt)




# Gráfica del histograma
import matplotlib.pyplot as plt
import numpy as np
"""
Este es EL gráfico principal para entender Tukey-Lambda.
Qué estás viendo
Eje X: probabilidades acumuladas (FDA)
Eje Y: cuantiles teóricos Tukey(λ)

Interpretación
La curvatura depende de λ
Si λ = 0 → logística (log(p/(1-p)))
λ > 0 → colas más delgadas
λ < 0 → colas más pesadas (NO recomendado para retornos financieros sin cuidado)
Aquí ves literalmente la forma de la distribución
"""
plt.figure(figsize=(8,5))
plt.plot(df_info['FDA'], df_info['Tukey_quantile'], marker='o', linestyle='-')
plt.xlabel('FDA (p)')
plt.ylabel('Tukey Quantile')
plt.title(f'Tukey-Lambda Distribution (λ = {lambda_value:.4f})')
plt.grid(True)
plt.show()

"""
Este es el equivalente visual del PPCC.
Interpretación
Si los puntos forman una línea recta → excelente ajuste
Curva en S → colas mal ajustadas
Apertura en extremos → problemas en colas

PPCC no es magia: esto es lo que está midiendo

"""
plt.figure(figsize=(6,6))
plt.scatter(df_info['Tukey_quantile'], df_info['Ranked_Return'])
plt.xlabel('Tukey Quantile')
plt.ylabel('Ranked Returns')
plt.title('PPCC Plot (Empirical vs Tukey Quantiles)')
plt.grid(True)
plt.show()

"""
Comparar varios valores de λ (
Aquí es donde entiendes de verdad λ.
| λ         | Efecto                   |
| --------- | ------------------------ |
| λ < 0     | Colas MUY pesadas        |
| λ = 0     | Logística                |
| 0 < λ < 1 | Ajuste típico financiero |
| λ grande  | Distribución casi lineal |

λ controla colas y curtosis
"""
lambdas = [-0.5, 0, 0.3, 0.8]

plt.figure(figsize=(8,5))

for lam in lambdas:
    tukey = np.where(
        lam == 0,
        np.log(df_info['FDA'] / (1 - df_info['FDA'])),
        (1 / lam) * (df_info['FDA']**lam - (1 - df_info['FDA'])**lam)
    )
    plt.plot(df_info['FDA'], tukey, label=f'λ = {lam}')

plt.xlabel('FDA (p)')
plt.ylabel('Tukey Quantile')
plt.title('Effect of λ on Tukey-Lambda Distribution')
plt.legend()
plt.grid(True)
plt.show()

'''
PPCC vs λ (el “Solver visual”)
Este conecta TODO.
Interpretación
El máximo es tu λ óptimo
Si la curva es plana → muchos λ similares
Si es puntiaguda → modelo muy sensible
Esto es EXACTAMENTE lo que hace Solver “por detrás”.
'''
lambda_grid = np.linspace(0.001, 1.5, 100)
ppcc_values = []

for lam in lambda_grid:
    tukey = (1 / lam) * (df_info['FDA']**lam - (1 - df_info['FDA'])**lam)
    corr = np.corrcoef(df_info['Ranked_Return'], tukey)[0,1]
    ppcc_values.append(corr)

plt.figure(figsize=(8,5))
plt.plot(lambda_grid, ppcc_values)
plt.axvline(lambda_opt, linestyle='--')
plt.xlabel('λ')
plt.ylabel('PPCC')
plt.title('PPCC as a function of λ')
plt.grid(True)
plt.show()


plt.figure(figsize=(8,5))
plt.hist(df_info['Ranked_Return'], bins=100, density=True, alpha=0.6)
plt.title('Histogram of Ranked Returns')
plt.grid(True)
plt.show()



###### pruebas para clase

def ppcc_distance(lambda_value, df):
    # We deal with an error of dimensions, we were operating with arrays instead of values.
    values = []
    for i in df['FDA']:
        if lambda_value[0] == 0:
            values.append(math.log(i / (1 - i)))
        else:
            values.append((1 / lambda_value[0]) * (i**lambda_value[0] - (1 - i)**lambda_value[0]))

    tukey_quantile = np.array(values)

    PPCC = np.corrcoef(
        df['ranked_return'],
        tukey_quantile
        )[0, 1] # np.corrcoef returns a matrix, from [0,1] we are only taking the value we are interested in.

    # Queremos PPCC lo más cercano posible a 1 y eso NO es lo mismo que “maximizar PPCC” en términos prácticos de optimización.
    return (1 - PPCC)**2


vector = np.array(ric_info.vector)

#########
def tukey_quantile(vector):
    df = pd.DataFrame()
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

    lambda_opt = best_result.x[0]
    # si el min r.fun = 0.0004
    # Entonces (1 - PPCC)**2 = 0.0004
    # Despejando... PPCC = 1 - sqrt(0.0004) = 0.98
    ppcc_opt = 1 - np.sqrt(best_result.fun)

    print("Lambda óptimo:", lambda_opt)
    print("PPCC alcanzado:", ppcc_opt)
    
PRUEBA = tukey_quantile(vector)

"""
AND NOW THE BEST PART
Let´s see if our universe follows a specific distribution with this method previously created
"""
rows = []

for ric in nombres:
    ric_info = market_data.distribution_manager(ric)
    ric_info.load_timeseries()
    ric_info.tukey_quantile()

    rows.append({
        'ric': ric,
        'lambda': ric_info.lambda_opt
    })

df_tukey = pd.DataFrame(rows)

# Ayuda visual 

def classify_lambda(lmbda):
    if -1.10 <= lmbda <= -0.90:
        return 'Cauchy'

    elif -0.13 <= lmbda <= -0.11:
        return 'Laplace'

    elif -0.07 <= lmbda <= -0.05:
        return 'Hyperbolic Secant'

    elif -0.01 <= lmbda <= 0.01:
        return 'Logistic'

    elif 0.15 <= lmbda <= 0.13:
        return 'Normal'

    elif 0.90 <= lmbda <= 1.10:
        return 'Uniform'

    else:
        return np.nan
    
def classify_lambda(lmbda):
    if -1.05 <= lmbda <= -0.95:
        return 'Cauchy'

    elif -0.15 <= lmbda <= -0.09:
        return 'Laplace'

    elif -0.08 <= lmbda <= -0.04:
        return 'Hyperbolic Secant'

    elif -0.02 <= lmbda <= 0.02:
        return 'Logistic'

    elif 0.12 <= lmbda <= 0.16:
        return 'Normal'

    elif 0.95 <= lmbda <= 1.05:
        return 'Uniform'

    else:
        return np.nan

df_tukey['tipo'] = df_tukey['lambda'].apply(classify_lambda)


def classify_lambda_distance(lmbda, tol=0.03):
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


df_tukey['tipo'] = df_tukey['lambda'].apply(classify_lambda_distance)

# Más estricto
df_tukey['tipo'] = df_tukey['lambda'].apply(
    lambda x: classify_lambda_distance(x, tol=0.02))

# Más flexible
df_tukey['tipo'] = df_tukey['lambda'].apply(
    lambda x: classify_lambda_distance(x, tol=0.05)
)


    
################################################################################################################


'''
Now we want to manipulate the dates
This could be useful for take a single sample, get many samples as we want,
get monthly/weekly returns of a single asset or two'''
