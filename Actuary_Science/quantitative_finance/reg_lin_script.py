import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import scipy.stats as st 

import market_data
importlib.reload(market_data) 

import capm
importlib.reload(capm) 

benchmark = 'SPX' # Mercado, es un índice VARIABLE X
security = 'GCARSOA1' # Asset VARIABLE Y

timeseries_x = market_data.load_timeseries(benchmark)
timeseries_y = market_data.load_timeseries(security)

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


# PLOT
# plt.gca() - Usado cuando el data contenido dentro de nuestro plot esta relacionado a dos variables x,y distintas, como tal de distinta información o escala, creo que es el que por el lado derecho de neustro plot muestra la escala por ejemplo de x y por el lado izquierdo la de y

plt.figure(figsize=(12,5))
plt.title('Timeseries of Close Prices')
plt.xlabel( 'Time')
plt.ylabel( 'Prices')
ax = plt.gca()
ax1 = timeseries.plot(kind='line', x='date', y='close_x', ax=ax, grid=True, color='blue', label=benchmark)
ax2 = timeseries.plot(kind='line', x='date', y='close_y' , color='red', secondary_y=True, ax=ax, grid=True, label=security)
ax1.legend(loc=2)
ax2.legend(loc=1)
plt.show()


# La volatilidad es alta cuando los precios tienden a caer pero es más estable durante la subida de los precios. ESTO SE LOGRA VER CON EL VIX Y EL INDICE SPX


# Lineal Regression 
x = timeseries['return_x'].values
y = timeseries['return_y'].values

slope_beta, intercept_alpha, correl_r, p_value, standard_error = st.linregress(x,y)
# SI el P value es la hipotesis nula entonces la pendiente es 0 - The p-value for a hypothesis is that the slope is zero
r_squared = correl_r**2 # Que tanto la linea explica el modelo, mayor R^2 el modelo es más compacto y un R^2 mas chico significa QUE NO EXPLICÓ NADA DE LA VARIACIÓN por lo tanto no hay información.
# R^2 se usa como un criterio, como una metrica de que tan buena o mala es una regresion lineal.
# El r^2 en este caso nos permitirá explicar cual tiene mejor correlación, con que mercado nuestro asset tiene mejor correlación.

hypothesis_null = p_value > 0.5# Recordemos que la hipótesis nula necesita un p_value MAYOR A 0.5

"""
Si tengo un p_value < 0.5 como 0.0 entonces rechazo la hipotesis nula, por ende no es de distribución normal
Entonces puedo decir con 95% de confianza que el BETA NO ES 0
Es estadísticamente significativo
Y si el alpha de la misma forma es significamente chico PERO NO ES 0
Entonces esto estaría satisfaciendo la Teoría de Mercados Eficientes en el cual alpha debe de ser 0 porque no peudo vencer sistematicamente al mercado, en este caso alpha es muy chico, lo cual es valido aún"""

# YA TENEMOS EL MODELO AHPRA HACEMOS EL PREDICT PERO ESTO NO ES MAS QUE LA FORMULA DE y^= a + bx entonces lo podemos hacer manualmente el predict
predictor_linreg = intercept_alpha + slope_beta * x

# Ahora haremos el plot de la regresión Lineal

# plot linear regression
# x = timeseries['return x'].values
# y = timeseries['return_y'].values
str_self = 'Linear regression | security ' + security \
    + ' | benchmark ' + benchmark + '/n' \
    + 'alpha (intercept) ' + str(np.round(intercept_alpha, 6)) \
    + ' | beta (slope) ' + str(np.round(slope_beta, 6)) + '\n' \
    + 'p-value ' + str(np.round(p_value, 6)) \
    + ' | null-hypothesis ' + str(hypothesis_null) + '\n' \
    + 'correl (r-value) ' + str(np.round(correl_r, 6)) \
    + ' | r-squared ' + str(np.round(r_squared, 6))
str_title = 'Scatterplot of returns ' + '/n' + str_self
plt.figure(figsize=(10,10))
plt.title(str_title)
plt.scatter(x, y)
plt.plot(x, predictor_linreg, color='green' )
plt.ylabel(security) 
plt.xlabel(benchmark) 
plt.grid()
plt.show()



