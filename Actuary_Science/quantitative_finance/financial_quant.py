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

ric = 'VIX'
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
security = 'SOXL'
benchmark = 'USDMXN'

summary = capm.model(security, benchmark)
summary.synchronise_timeseries() 
summary.plot_timeseries()
summary.compute_linear_reg() 
summary.plot_linear_reg()

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

benchmarks = ['USDMXN', 'USDEUR', 'DJI', 'SPX', 'NASDAQ', 'MSCI_W', 'MSCI_EM', 'VIX', 'BRENT', 'MXX', 'inflationusa', 'inflationmex', 'XAU_USD', 'SOX']

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

ric_corr = 'SOXL'
df_security = df[df['security'] == ric_corr].sort_values(by=["correlation"], ascending=False).reset_index(drop=True)

