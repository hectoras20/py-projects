# busqueda de factores potenciales...
# Realizamos un análisis cualitativo de los posibles factores para diferentes activos

# Detro del notebook de apt realizamos un análisis mas profundo el cual retomaremos bajo un nuevo sistema dinámico de codigo usando yahoo finance el cual nos permitirá hacer simulaciones y encontrar de manera más rapida nuestos factores

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_yahoo_timeseries(
    tickers,
    from_date="2024-01-01",
    to_date=None,
    price_col="Close",
    returns=True,
    log_returns=True,
    auto_adjust=True
):
    """
    Descarga datos desde Yahoo Finance y grafica:
    
    1. Rendimientos diarios
    2. O precios de cierre
    
    Parámetros:
    -----------
    tickers : list
        Lista de tickers Yahoo Finance.
        Ejemplo:
        ["AAPL", "GCARSOA1.MX", "^MXX"]

    from_date : str
        Fecha inicial.

    to_date : str
        Fecha final.

    price_col : str
        Columna de precio.
        Normalmente "Close".

    returns : bool
        True  -> grafica rendimientos
        False -> grafica precios

    log_returns : bool
        Solo aplica si returns=True

        True  -> rendimientos logarítmicos
        False -> rendimientos simples

    auto_adjust : bool
        Ajustar por dividendos/splits.
    """

    # --------------------------------------------------------
    # DESCARGA DATOS
    # --------------------------------------------------------

    data = yf.download(
        tickers=tickers,
        start=from_date,
        end=to_date,
        auto_adjust=auto_adjust,
        progress=False
    )

    prices = data[price_col].copy()

    # Si solo hay un ticker
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    prices = prices.dropna(how="all")

    # --------------------------------------------------------
    # CALCULAR RETURNS O USAR PRECIOS
    # --------------------------------------------------------

    if returns:

        if log_returns:

            series_to_plot = np.log(prices / prices.shift(1))

            title = "Logarithmic Daily Returns"
            ylabel = "Log Return"

        else:

            series_to_plot = prices.pct_change()

            title = "Simple Daily Returns"
            ylabel = "Simple Return"

        series_to_plot = series_to_plot.dropna(how="all")

    else:

        series_to_plot = prices / prices.iloc[0] * 100
    
        title = "Normalized Closing Prices"
        ylabel = "Price Index Base 100"

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    plt.figure(figsize=(12, 6))

    for ticker in series_to_plot.columns:

        plt.plot(
            series_to_plot.index,
            series_to_plot[ticker],
            label=ticker
        )

    if returns:
        plt.axhline(0, linewidth=1)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)

    plt.grid(True)
    plt.legend()

    plt.show()

    return series_to_plot

nvda_ecosystem = [
    # Core AI
    "NVDA","AMD","AVGO","MRVL","MU","TSM","ARM","QCOM","INTC",

    # Foundry
    "ASML","AMAT","LRCX","KLAC","AXTI","AMKR",

    # Networking
    "ANET","CSCO","CIEN","FN","LITE","COHR",

    # Datacenter
    "EQIX","DLR","VRT","ETN","SMCI",

    # Hyperscalers
    "MSFT","AMZN","GOOGL","META","ORCL",

    # Energy
    "CEG","VST","SMR","TLN",

    # ETFs
    "SMH","SOXX","QQQ","XLK","VGT","BOTZ",

    # Macro
    "^GSPC","^IXIC","^SOX","^TNX","DX-Y.NYB","^VIX"
]

nvda_core_ai = [
    "NVDA",     # NVIDIA
    "AMD",      # AMD
    "AVGO",     # Broadcom
    "MRVL",     # Marvell
    "ARM",      # ARM Holdings
    "TSM",      # TSMC
]

core_ai = [
    "NVDA",
    "AMD",
    "AVGO",
    "MRVL",
    "MU",
    "TSM",
    "ASML",
    "ARM",
    "QCOM",
    "INTC",
    "AMAT",
    "LRCX",
    "KLAC"
]

memory_ai = [
    "NVDA",
    "MU",
    "000660.KS",
    "005930.KS"
]

foundry = [
    "TSM",
    "ASML",
    "AMAT",
    "LRCX",
    "KLAC",
    "AXTI",
    "AMKR"
]

networking_ai = [
    "NVDA",
    "ANET",
    "CSCO",
    "MRVL",
    "CIEN",
    "FN",
    "LITE",
    "COHR"
]

returns_log = plot_yahoo_timeseries(
    tickers= networking_ai, # ["AAPL", "GCARSOA1.MX", "^MXX"],
    from_date="2026-01-01",
    to_date="2026-05-19",
    returns=False,
    log_returns=True)

datacenter = [
    "NVDA",
    "EQIX",
    "DLR",
    "VRT", # APROVECHAMIENTO DE VRT MAS NO HAY UNA CORRELACION DIRECTA, CON SU CAIDA NVDA APROVECHA Y SE RECUPERA PARA MANTENERSE, CUANDO SUBE MUCHO NVDA RESBALA PERO SIGUE SU RITMO
    "ETN",
    "SMCI"
]

hyperscalers = [
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "ORCL"
] # NO SIRVIÓ

energy_ai = [
    "CEG",
    "VST",
    "SMR",
    "TLN",
    "ETN"
] # NO SIRVIÓ

cooling_power = [
    "NVDA",
    "VRT",
    "ETN", # MAYOR EXPLICACION OBSERVADA EN VRT CON NVDA 
    "SU.PA"
]

returns_log = plot_yahoo_timeseries(
    tickers= cooling_power, # ["AAPL", "GCARSOA1.MX", "^MXX"],
    from_date="2026-01-01",
    to_date="2026-05-19",
    returns=False,
    log_returns=True)

##################
######### GCARSO
##################

gcarso_core = [
    "GCARSOA1.MX",
    "AMX",
    "BIMBOA.MX",
    "WALMEX.MX",
    "FEMSAUBD.MX"
]

gcarso_infra = [
    "GCARSOA1.MX",
    "ICA.MX",
    "PINFRA.MX",
    "IDEALB-1.MX",
    "CEMEXCPO.MX",
    "GCC.MX"
]

gcarso_materials = [
    "GCARSOA1.MX",
    "GMEXICOB.MX",
    "PE&OLES.MX",
    "CEMEXCPO.MX",
    "X.MX",   # si tienes ADR/local
    "FCX"
]

gcarso_energy = [
    "GCARSOA1.MX",
    "VISTA",
    "PBR",
    "XOM",
    "SLB",
    "HAL"
]

returns_log = plot_yahoo_timeseries(
    tickers= gcarso_energy, # ["AAPL", "GCARSOA1.MX", "^MXX"],
    from_date="2026-01-01",
    to_date="2026-05-19",
    returns=False,
    log_returns=True)

gcarso_retail = [
    "GCARSOA1.MX",
    "GSANBORB-1.MX",
    "WALMEX.MX",
    "LIVEPOLC-1.MX",
    "FEMSAUBD.MX", ###KEY FOR FORECAST THE PRICE OF THE OTHER TICKERS
    "CHEDRAUIB.MX"
]

gcarso_telecom = [
    "GCARSOA1.MX",
    "AMX",
    "TMUS",
    "VZ",
    "T",
    "CSCO"
] # INDEPENDIENTES ENTRE ELLOS CON VZ QUIZÁ UNA CORRELACION INVERSA

gcarso_industrial = [
    "GCARSOA1.MX",
    "ALFAA.MX",
    "KOFUBL.MX",
    "VESTA.MX",
    "ETN",
    "HON"
]

gcarso_realestate = [
    "GCARSOA1.MX",
    "VESTA.MX",
    "FUNO11.MX",
    "DANHOS13.MX",
    "FMTY14.MX",
    "FIBRAPL14.MX"
] # DRANHOS12 Y FIBRAPL PODRIAN EXPLICARLO BIEN YA QUE DAN ESE FEELING DE PREDECIR EL PRECIO DE GCARSO

gcarso_macro = [
    "GCARSOA1.MX",
    "^MXX",
    "MXN=X",
    "^TNX",
    "^IRX",
    "CL=F",
    "HG=F",
    "^VIX"
]

gcarso_ecosystem = [

    # Core
    "GCARSOA1.MX","AMX",

    # Infraestructura
    "PINFRA.MX","IDEALB-1.MX","CEMEXCPO.MX","GCC.MX",

    # Materiales
    "GMEXICOB.MX","PE&OLES.MX","FCX",

    # Energía
    "VISTA","PBR","XOM","SLB","HAL",

    # Retail
    "GSANBORB-1.MX","WALMEX.MX","LIVEPOLC-1.MX","FEMSAUBD.MX",

    # Industrial
    "ALFAA.MX","VESTA.MX","ETN","HON",

    # Real estate
    "FUNO11.MX","DANHOS13.MX","FMTY14.MX","FIBRAPL14.MX",

    # ETFs
    "EWW","XLI","XLB","PAVE",

    # Macro
    "^MXX","MXN=X","CL=F","HG=F","^VIX"
]

returns_log = plot_yahoo_timeseries(
    tickers= gcarso_macro, # ["AAPL", "GCARSOA1.MX", "^MXX"],
    from_date="2026-01-01",
    to_date="2026-05-19",
    returns=False,
    log_returns=True)

###### APT CODE

