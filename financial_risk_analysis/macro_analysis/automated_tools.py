import pandas as pd
import matplotlib.pyplot as plt
import importlib
import csv
from pathlib import Path
import numpy as np
import math
import scipy.stats as st
import yfinance as yf            # Descarga de precios desde Yahoo Finance

from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import boxcox
from scipy.stats import boxcox_llf
from scipy.stats import t
from scipy import stats
from sklearn.preprocessing import PowerTransformer


##### BLOQUE PARA MANIPULAR DATOS Y PODERNOS UTILIZAR
def corregir_año(año): # El año serán cada valor de la columna a la cual se esté manipulando.
    año = str(año)
    if len(año) == 2:
        return '20' + año  # Asumimos que son años del 2000+
    return año

def corrector(clean_data, columna_interes = str):
    #Empezamos con la depuración en un nuevo df
    # Renombrar columnas necesarias
    # REVISAR df = clean_data.rename(columns={"Date": "Fecha", "Close": "Cierre", "Precio": "Cierre"}).copy()
    df = clean_data.copy()
    # df = df.dropna()
    # Verificación de columnas
    if not {'Fecha'}.issubset(df.columns):
        raise ValueError('The necessary columns do not exist in this data')
    # Normalizar separadores de fecha
    df['Fecha'] = df['Fecha'].astype(str).str.replace('/', '.', regex=False)
    # Convertir fechas (quita hora)
    df['Fecha'] = pd.to_datetime(
        df['Fecha'],
        format='mixed',
        dayfirst=True
    ).dt.date
    
    # Columna de interes
    df[columna_interes] = (
        df[columna_interes]
        .astype(str)
        .str.replace(',', '', regex=False)
        .replace('N/E', None)  # o np.nan
        .astype(float)
        .round(2))
    return df


def load_timeseries(ticker, columna='Cierre', tipo_var=None,
                       from_date=None, to_date=None, log_returns=False):
    """
    Equivalente a load_timeseries() pero descarga datos desde Yahoo Finance
    en lugar de leer un CSV local.

    Devuelve exactamente el mismo formato que load_timeseries():
        Fecha | Cierre | return_<ticker>   (si columna == 'Cierre')
        Fecha | Tasa   | dif_bps_<ticker>  (si columna == 'Tasa')

    Parámetros
    ----------
    ticker     : str   — ticker de Yahoo Finance (ej. '^GSPC', 'AAPL', '^IRX')
    columna    : str   — 'Cierre' para precios, 'Tasa' para tasas (^IRX, CETES)
    tipo_var   : str   — None | 'lag' | 'dec_base' | 'yoy'  (igual que load_timeseries)
    from_date  : str   — 'YYYY-MM-DD'
    to_date    : str   — 'YYYY-MM-DD'
    log_returns: bool  — True = log-rendimientos, False = rendimientos simples

    Nota sobre ^IRX (T-Bill):
        Yahoo lo publica como tasa anual en porcentaje (ej. 5.25 = 5.25 % anual).
        Cuando columna == 'Tasa' esta función asume ese formato y lo almacena
        directamente en la columna 'Tasa'. La conversión a semanal/mensual se
        delega a quien llame a esta función (mismo comportamiento que load_timeseries
        con archivos de tasas de Banxico o Investing).
    """
    # ── 1. Descarga ──────────────────────────────────────────────────────────
    raw = yf.download(
        tickers     = ticker,
        start       = from_date,
        end         = to_date,
        auto_adjust = True,    # precios ajustados por splits y dividendos
        progress    = False
    )

    if raw.empty:
        raise ValueError(f"Yahoo Finance no devolvió datos para '{ticker}'")

    # ── 2. Extraer precio de cierre y normalizar columnas ────────────────────
    # yf.download puede devolver MultiIndex si se piden varios tickers;
    # aquí siempre pedimos uno solo, pero lo manejamos de forma defensiva.
    if isinstance(raw.columns, pd.MultiIndex):
        cierre_raw = raw['Close'][ticker]
    else:
        cierre_raw = raw['Close']

    # ── 3. Construir DataFrame con el mismo esquema que load_timeseries ──────
    t = pd.DataFrame()
    t['Fecha']  = cierre_raw.index          # DatetimeIndex → columna Fecha
    t['Fecha']  = pd.to_datetime(t['Fecha']).dt.normalize()  # quita hora

    # Asignar a la columna con el nombre que espera el resto del código
    t[columna]  = cierre_raw.values

    # Eliminar NaN: días en que el mercado estuvo cerrado o sin dato
    # (festivos, suspensiones, huecos de Yahoo Finance)
    t = t.dropna(subset=[columna])

    t = t.sort_values('Fecha').reset_index(drop=True)

    # ── 4. Variaciones (mismo bloque lógico que load_timeseries) ─────────────
    if tipo_var == 'lag':
        t['Var%'] = (t[columna] / t[columna].shift(1) - 1) * 100

    elif tipo_var == 'dec_base':
        t['year']  = t['Fecha'].dt.year
        t['month'] = t['Fecha'].dt.month
        dec_values = t[t['month'] == 12].set_index('year')[columna]
        t['base']  = t['year'].map(lambda y: dec_values.get(y - 1))
        t['base']  = t['base'].fillna(
            t.groupby('year')[columna].transform('first')
        )
        t['Var%']  = (t[columna] / t['base'] - 1) * 100

    elif tipo_var == 'yoy':
        t['Var%'] = (t[columna] / t[columna].shift(12) - 1) * 100

    t = t.dropna()
    t = t.reset_index(drop=True)

    # ── 5. Filtrado de fechas ─────────────────────────────────────────────────
    if from_date is not None:
        t = t[t['Fecha'] >= pd.to_datetime(from_date)]
    if to_date is not None:
        t = t[t['Fecha'] <= pd.to_datetime(to_date)]

    # ── 6. Rendimientos (precios) ─────────────────────────────────────────────
    if columna in ['Cierre', 'Close']:
        name_close = columna
        t['close_previous'] = t[name_close].shift(1)
        if not log_returns:
            t['return_' + ticker] = t[name_close] / t['close_previous'] - 1
        else:
            t['return_' + ticker] = np.log(t[name_close] / t['close_previous'])

    # ── 7. Diferencias en bps (tasas) ─────────────────────────────────────────
    if columna in ['Tasa', 'Diferencia']:
        t['rate_previous']    = t[columna].shift(1)
        t['dif_bps_' + ticker] = (t[columna] - t['rate_previous']) * 100

    t = t.dropna()
    t = t.reset_index(drop=True)
    return t


def synchronise_timeseries(info1, info2,
                               columna1='Cierre', columna2='Cierre',
                               name1=None, name2=None,
                               dif=False, suma=False,
                               from_date='aaaa-mm-dd', to_date='aaaa-mm-dd',
                               log_returns=False, model_type='macro'):
    """
    Equivalente a synchronise_timseries_df() pero usando load_timeseries.

    info1, info2 pueden ser:
      - str  → ticker de Yahoo Finance (se descarga automáticamente)
      - DataFrame → ya cargado (mismo comportamiento que la versión original)

    El merge se hace con pd.merge(on='Fecha', how='inner') en lugar del
    enfoque set() del original, lo que es más robusto con DatetimeIndex y
    evita pérdida de datos por discrepancias de tipo.

    Huecos (días sin dato en uno de los dos activos, ej. festivos) se
    eliminan automáticamente por el inner join + dropna() final.
    """
    # ── Cargar series ─────────────────────────────────────────────────────────
    if isinstance(info1, str):
        timeseries_x = load_timeseries(info1, columna=columna1,
                                           from_date=(None if from_date == 'aaaa-mm-dd' else from_date),
                                           to_date=(None if to_date == 'aaaa-mm-dd' else to_date),
                                           log_returns=log_returns)
        name1 = name1 or info1
    else:
        timeseries_x = info1.copy()
        name1 = name1 or columna1

    if isinstance(info2, str):
        timeseries_y = load_timeseries(info2, columna=columna2,
                                           from_date=(None if from_date == 'aaaa-mm-dd' else from_date),
                                           to_date=(None if to_date == 'aaaa-mm-dd' else to_date),
                                           log_returns=log_returns)
        name2 = name2 or info2
    else:
        timeseries_y = info2.copy()
        name2 = name2 or columna2

    # ── Asegurar tipo datetime en Fecha ────────────────────────────────────────
    timeseries_x['Fecha'] = pd.to_datetime(timeseries_x['Fecha']).dt.normalize()
    timeseries_y['Fecha'] = pd.to_datetime(timeseries_y['Fecha']).dt.normalize()

    # ── Seleccionar columna de retorno/tasa según model_type ─────────────────
    def _pick_col(ts, info, columna, model_type):
        if model_type in ['macro', 'statistics']:
            if columna in ['Tasa', 'Diferencia']:
                return 'dif_bps_' + info
            else:
                return 'return_' + info
        elif model_type == 'learning':
            return columna

    col_x = _pick_col(timeseries_x, info1 if isinstance(info1, str) else columna1,
                      columna1, model_type)
    col_y = _pick_col(timeseries_y, info2 if isinstance(info2, str) else columna2,
                      columna2, model_type)

    # ── Reducir a columnas necesarias antes del merge ─────────────────────────
    # Esto evita colisiones de columnas con el mismo nombre
    x_slim = timeseries_x[['Fecha', col_x]].rename(columns={col_x: name1})
    y_slim = timeseries_y[['Fecha', col_y]].rename(columns={col_y: name2})

    # ── Merge inner por fecha (intersección, más robusto que set()) ───────────
    # inner join descarta automáticamente días donde falta dato en cualquiera
    # de las dos series (festivos, suspensiones, huecos de Yahoo Finance)
    timeseries = pd.merge(x_slim, y_slim, on='Fecha', how='inner')

    # Eliminar cualquier hueco residual (NaN que pudieran quedar)
    timeseries = timeseries.dropna()

    # ── Columnas opcionales ───────────────────────────────────────────────────
    if dif:
        timeseries['Diferencia'] = timeseries[name1] - timeseries[name2]
    if suma:
        timeseries['Suma'] = timeseries[name1] + timeseries[name2]

    # ── Ordenar y resetear índice ─────────────────────────────────────────────
    timeseries = timeseries.sort_values('Fecha').reset_index(drop=True)

    # ── Filtrado de fechas (si no se pasó en la descarga) ────────────────────
    if from_date != 'aaaa-mm-dd' and to_date != 'aaaa-mm-dd':
        mask = (timeseries['Fecha'] >= from_date) & (timeseries['Fecha'] <= to_date)
        timeseries = timeseries.loc[mask].reset_index(drop=True)
    elif to_date != 'aaaa-mm-dd':
        timeseries = timeseries.loc[timeseries['Fecha'] <= to_date].reset_index(drop=True)
    elif from_date != 'aaaa-mm-dd':
        timeseries = timeseries.loc[timeseries['Fecha'] >= from_date].reset_index(drop=True)

    return timeseries


def target_column_yf(ticker):
    """
    Equivalente a target_column() pero para Yahoo Finance.
    Siempre retorna 'Cierre' porque load_timeseries usa ese nombre
    para precios. Para tasas (^IRX u otros) retorna 'Tasa'.

    Tickers de tasas reconocidos: ^IRX, ^TNX, ^TYX, ^FVX, ^IRX
    (Treasury Bills y Bonds de Yahoo Finance)
    """
    tasas_yf = {'^IRX', '^TNX', '^TYX', '^FVX', '^TWO'}
    if ticker.upper() in tasas_yf:
        return 'Tasa'
    return 'Cierre'


def plot_timeseries(df, ric1 = str,ric2 = str, secondary_y = True):
    plt.figure(figsize=(12,5))
    plt.title('Timeseries')
    plt.xlabel( 'Time')
    plt.ylabel( 'Prices')
    ax = plt.gca()
    ax1 = df.plot(kind='line', x='Fecha', y=ric1, ax=ax, grid=True, color='blue', label=ric1)
    ax2 = df.plot(kind='line', x='Fecha', y=ric2 , color='red', secondary_y= secondary_y, ax=ax, grid=True, label=ric2)
    ax1.legend(loc=2)
    ax2.legend(loc=1)
    plt.show()
    
def compute_linear_reg(df, x = str, y = str, decimals = 6):
    x_data = df[x]
    y_data = df[y]
    # Lineal Regression 
    slope_beta, intercept_alpha, correl_r, p_value, standard_error = st.linregress(x_data, y_data)
    beta = np.round(slope_beta, decimals)
    alpha = np.round(intercept_alpha, decimals)
    p_value = np.round(p_value, decimals)
    correlation = np.round(correl_r, decimals)
    r_squared = np.round(correl_r**2, decimals)
    hypothesis_null = p_value > 0.5
    predictor_linreg = intercept_alpha + slope_beta * x_data
    str_self = 'Linear regression | y ' + y \
        + ' | x ' + x + '\n' \
        + 'alpha ' + str(alpha) \
        + ' | beta (slope) ' + str(beta)  + '\n' \
        + 'p-value ' + str(p_value) \
        + ' | null-hypothesis ' + str(hypothesis_null) + '\n' \
        + 'correl (r-value) ' + str(correlation) \
        + ' | r-squared ' + str(r_squared)
        
    str_title = 'Scatterplot of returns ' + '\n' + str_self
    ## plt.figure(figsize=(10,10))
    plt.title(str_title)
    plt.scatter(x_data, y_data)
    plt.plot(x_data, predictor_linreg, color='green' )
    plt.ylabel(x) 
    plt.xlabel(y) 
    plt.grid()
    plt.show()

# Quiero guardar la información que consulte en un archivo existente o crear uno nuevo
def cargar_archivo(df, archivo = str):
    base_path = Path(__file__).resolve().parents[1]  # sube al root del proyecto
    output_dir = base_path / "macro_analysis/macro_data"

    output_dir.mkdir(parents=True, exist_ok=True)  # por si no existe

    path = output_dir / f"{archivo}.csv"

    with open(path, "w", encoding="UTF8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(df.columns.tolist())
        writer.writerows(df.to_numpy().tolist())

    print(f"Archivo guardado en: {path}")



def promedio_mensual(df, columna_interes):
    """
    Calcula el promedio mensual (mes-año) de las tasas y variaciones.
    Retorna un DataFrame con frecuencia mensual.
    """
    df = df.copy()
    
    # Crear identificador Año-Mes
    df['YearMonth'] = df['Fecha'].dt.to_period('M')
    
    # Agrupar por mes y calcular promedios
    df_mensual = (
        df
        .groupby('YearMonth')
        .agg({
            columna_interes : 'mean'
        })
        .reset_index()
    )
    
    # Convertir YearMonth a fecha (primer día del mes)
    df_mensual['Fecha'] = df_mensual['YearMonth'].dt.to_timestamp()
    
    # Orden final
    df_mensual = df_mensual.sort_values('Fecha').reset_index(drop=True)
    # df_mensual.drop('Fecha', axis=1, inplace=True)
    
    return df_mensual

def plot_timeseries_one(csv_name = str, columna = str, from_date = None, to_date = None):
    info = load_timeseries(csv_name, columna, from_date = from_date, to_date = to_date)
    plt.figure()
    info.plot(kind ='line', x='Fecha', y = columna, grid = True, title='Timeseries')
    plt.show()


def plot_two_timeseries(info1 = str, info2 = str, columna1 = str, columna2 = str, from_date=None, to_date=None, titulo = 'Timeseries', secondary_y = True):
    plt.figure(figsize=(12,5)) 
    plt.title(titulo) 
    plt.xlabel( 'Time') 
    plt.ylabel( 'Rate/Prices') 
    ax = plt.gca() 
    df1 = load_timeseries(info1, columna1, from_date = from_date, to_date = to_date) 
    df2 = load_timeseries(info2, columna2, from_date = from_date, to_date = to_date) 
    ax1 = df1.plot(kind='line', x='Fecha', y=columna1, ax=ax, grid=True, color='blue', label=info1) 
    ax2 = df2.plot(kind='line', x='Fecha', y=columna2 , color='red', secondary_y = secondary_y, ax=ax, grid=True, label=info2) 
    ax1.legend(loc=2) 
    ax2.legend(loc=1) 
    plt.show()
    
def plot_normalized_timeseries_def_columns(rics, columns = list, from_date='aaaa-mm-dd', to_date='aaaa-mm-dd', base_value=100, legend = 'rics'): 
    """ 
    Plot normalized price time series for a group of assets. Prices are first filtered by the selected date window and then normalized so that the first observation within the window equals the chosen base_value. Price_norm = (Price_t / Price_from) * base_value length(rics) must be equal to the length of columnas """
    plt.figure(figsize=(12,6)) 
    for a, b in zip(rics, columns): 
        t = load_timeseries(a, b) # Date filtering 
        if from_date != 'aaaa-mm-dd' and to_date != 'aaaa-mm-dd': 
            subsetting = (t['Fecha'] >= from_date) & (t['Fecha'] <= to_date) 
            t = t.loc[subsetting].reset_index(drop=True) 
        elif to_date != 'aaaa-mm-dd': 
            subsetting = t['Fecha'] <= to_date 
            t = t.loc[subsetting].reset_index(drop=True) 
        elif from_date != 'aaaa-mm-dd': 
            subsetting = t['Fecha'] >= from_date 
            t = t.loc[subsetting].reset_index(drop=True) 
            
        if t.empty: 
            print("No data available for", a) 
            continue 
        base = t[b].iloc[0] 
        normalized = t[b] / base * base_value
        plt.plot(t['Fecha'], normalized, label=a) if legend == 'rics' else plt.plot(t['Fecha'], normalized, label=b)
        
    plt.title(f"Normalized Price Series (Base = {base_value})") 
    plt.xlabel("Time") 
    plt.ylabel(f"Price Index (Base = {base_value})") 
    plt.grid(True) 
    plt.legend() 
    plt.show()

def target_column(name_file = str):
    base_path = Path(__file__).parent
    data_path = base_path / "macro_data"
    path = data_path / f"{name_file}.csv"
    df = pd.read_csv(path)
    columns = df.columns
    targets = ['diferencia', 'cierre', 'close', 'tasa', 'rate', 'price']
    for col in columns:
        if col.lower() in targets:
            return col




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




def plot_yahoo_timeseries(
    tickers,
    from_date="2024-01-01",
    to_date=None,
    price_col="Close",
    returns=False,       # Cambiado a False por defecto para ver precios directos
    log_returns=True,
    auto_adjust=True
):
    """
    Descarga datos desde Yahoo Finance y grafica en dos subgráficos:
    1. Precios de cierre (o Rendimientos) de los activos.
    2. Volumen de operaciones de los activos.
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

    # --------------------------------------------------------
    # PROCESAMIENTO DE PRECIOS / RENDIMIENTOS
    # --------------------------------------------------------
    prices = data[price_col].copy()

    # Si solo hay un ticker, asegurar estructura DataFrame
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0] if isinstance(tickers, list) else tickers)

    prices = prices.dropna(how="all")

    if returns:
        if log_returns:
            series_to_plot = np.log(prices / prices.shift(1))
            title_top = "Logarithmic Daily Returns"
            ylabel_top = "Log Return"
        else:
            series_to_plot = prices.pct_change()
            title_top = "Simple Daily Returns"
            ylabel_top = "Simple Return"
        series_to_plot = series_to_plot.dropna(how="all")
    else:
        # Precios normalizados base 100 para poder comparar múltiples activos fácilmente
        series_to_plot = prices / prices.iloc[0] * 100
        title_top = "Normalized Closing Prices"
        ylabel_top = "Price Index Base 100"

    # --------------------------------------------------------
    # PROCESAMIENTO DE VOLUMEN
    # --------------------------------------------------------
    volume_data = data["Volume"].copy()
    if isinstance(volume_data, pd.Series):
        volume_data = volume_data.to_frame(name=tickers[0] if isinstance(tickers, list) else tickers)
    
    volume_data = volume_data.dropna(how="all")

    # --------------------------------------------------------
    # PLOT (SUBPLOTS CONFIGURACIÓN)
    # --------------------------------------------------------
    # Creamos 2 subgráficos compartiendo el mismo eje X (Fechas)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # --- Subgráfico 1: Precios o Rendimientos (Superior) ---
    for ticker in series_to_plot.columns:
        ax1.plot(
            series_to_plot.index,
            series_to_plot[ticker],
            label=f"{ticker} - Price/Return"
        )
    
    if returns:
        ax1.axhline(0, linewidth=1, color="black", linestyle="--")

    ax1.set_title(title_top)
    ax1.set_ylabel(ylabel_top)
    ax1.grid(True)
    ax1.legend()

    # --- Subgráfico 2: Volumen (Inferior) ---
    for ticker in volume_data.columns:
        ax2.plot(
            volume_data.index,
            volume_data[ticker],
            label=f"{ticker} - Volume",
            linestyle="--" # Línea discontinua para diferenciar visualmente el volumen
        )

    ax2.set_title("Trading Volume per Asset")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Volume (Shares / Contracts)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout() # Ajusta los márgenes automáticamente
    plt.show()

    # Retornamos un diccionario con ambos DataFrames procesados por si los necesitas
    return {"prices_returns": series_to_plot, "volume": volume_data}


import mplfinance as mpf

def plot_advanced_market_profile(
    ticker,
    from_date="2024-01-01",
    to_date=None,
    vwap_window=20,       # Periodo para el VWAP acumulado
    num_std=2,            # Desviaciones estándar para las bandas VWAP
    price_color="blue",   # Color base para el activo (velas) y su VWAP
    vol_profile_bins=30   # Cuántas barras horizontales tendrá el perfil
):
    """
    Descarga datos de Yahoo Finance y grafica para UN activo:
    1. Gráfico de velas (Open, High, Low, Close) en una misma gama de color.
    2. Canal de VWAP con bandas de desviación estándar y sombreado transparente.
    3. Perfil de Volumen (Volume Profile) manual inyectado horizontalmente.
    """
    # --------------------------------------------------------
    # 1. DESCARGA Y LIMPIEZA DE DATOS
    # --------------------------------------------------------
    data = yf.download(
        tickers=ticker,
        start=from_date,
        end=to_date,
        auto_adjust=False, # Requerido False para mantener High/Low puros
        progress=False
    )
    
    if data.empty:
        print(f"No se encontraron datos para el ticker {ticker}")
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    # --------------------------------------------------------
    # 2. CÁLCULO MATEMÁTICO DEL VWAP Y SUS BANDAS
    # --------------------------------------------------------
    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
    tp_v = typical_price * data["Volume"]
    
    rolling_tp_v = tp_v.rolling(window=vwap_window).sum()
    rolling_vol = data["Volume"].rolling(window=vwap_window).sum()
    
    data["VWAP"] = rolling_tp_v / rolling_vol
    
    rolling_variance = ((typical_price - data["VWAP"]) ** 2 * data["Volume"]).rolling(window=vwap_window).sum() / rolling_vol
    data["VWAP_Std"] = np.sqrt(rolling_variance)
    
    data["VWAP_Upper"] = data["VWAP"] + (num_std * data["VWAP_Std"])
    data["VWAP_Lower"] = data["VWAP"] - (num_std * data["VWAP_Std"])
    
    data = data.dropna()

    # --------------------------------------------------------
    # 3. ESTILOS DE VELAS Y LÍNEAS ADICIONALES
    # --------------------------------------------------------
    custom_market_colors = mpf.make_marketcolors(
        up=price_color, down=price_color,
        edge=price_color, wick=price_color,
        volume='gray', inherit=True
    )
    
    custom_style = mpf.make_mpf_style(
        marketcolors=custom_market_colors,
        gridstyle="--",
        y_on_right=False # Forzar que las escalas de precio queden a la izquierda
    )

    # Añadimos las líneas de VWAP
    additional_plots = [
        mpf.make_addplot(data["VWAP"], color=price_color, width=2.0, label="VWAP"),
        mpf.make_addplot(data["VWAP_Upper"], color=price_color, width=1.0, linestyle=":", alpha=0.5),
        mpf.make_addplot(data["VWAP_Lower"], color=price_color, width=1.0, linestyle=":", alpha=0.5)
    ]

    # --------------------------------------------------------
    # 4. GRAFICADO CON INTERCEPCIÓN DE FIGURA (Matplotlib)
    # --------------------------------------------------------
    # Usamos returnfig=True para poder manipular los ejes directamente
    fig, axlist = mpf.plot(
        data,
        type="candle",
        style=custom_style,
        addplot=additional_plots,
        volume=True,
        title=f"\nMarket Profile: {ticker} (VWAP Channels & Volume Profile)",
        ylabel="Price",
        ylabel_lower="Volume Traded",
        figsize=(14, 9),
        returnfig=True # <- Devuelve la tupla (figura, lista_de_ejes)
    )

    # axlist[0] corresponde al cuadro del gráfico de precios principal
    ax_price = axlist[0]

    # --- A. Canal sombreado transparente para las bandas del VWAP ---
    ax_price.fill_between(
        range(len(data)), # Usamos rango numérico interno de las posiciones de x en mplfinance
        data["VWAP_Lower"],
        data["VWAP_Upper"],
        color=price_color,
        alpha=0.10, # Muy transparente para no tapar las velas
        label="VWAP Band"
    )

    # --- B. Inyección Manual del Perfil de Volumen (Volume Profile) ---
    # Determinamos el tamaño de cada división (bucket de precios)
    price_min = data['Low'].min()
    price_max = data['High'].max()
    bin_size = (price_max - price_min) / vol_profile_bins

    # Agrupamos los volúmenes según el nivel de precios más cercano
    volume_profile = data['Volume'].groupby(
        data['Close'].apply(lambda x: bin_size * round(x / bin_size, 0))
    ).sum()

    # Extraemos los ejes X e Y del histograma horizontal
    vp_prices = volume_profile.index.values
    vp_volumes = volume_profile.values

    # Escalamos el volumen de las barras para que cubran un espacio armónico del gráfico sin saturarlo
    max_visible_width = len(data) * 0.25 # Máximo 25% del ancho de la gráfica
    scaled_volumes = (vp_volumes / vp_volumes.max()) * max_visible_width

    # Pintamos el Perfil de Volumen desde el extremo derecho hacia la izquierda
    # Para ponerlo al extremo derecho, restamos el ancho al valor máximo de X
    right_edge_x = len(data) - 1

    ax_price.barh(
        vp_prices,
        width=-scaled_volumes,       # Valor negativo para que crezca hacia la izquierda
        left=right_edge_x,          # Anclado al borde derecho del gráfico
        height=bin_size * 0.85,      # Espaciado leve entre barras horizontales
        align='center',
        color='gray',
        alpha=0.25,                 # Sutil para que funcione de fondo
        edgecolor='gray'
    )

    # Forzar actualización de la leyenda en pantalla
    ax_price.legend(loc="upper left")
    
    mpf.show()
    return data


def plot_advanced_market_profile_focus_vol(
    ticker,
    from_date="2024-01-01",
    to_date=None,
    vwap_window=20,       # Periodo para el VWAP acumulado
    num_std=2,            # Desviaciones estándar para las bandas VWAP
    price_color="blue",   # Color base para el activo (velas) y su VWAP
    vol_profile_bins=30   # Cuántas barras horizontales tendrá el perfil
):
    """
    Descarga datos de Yahoo Finance y grafica para UN activo:
    1. Gráfico de velas y Canal de VWAP con bandas de desviación estándar.
    2. Perfil de Volumen (Volume Profile) manual inyectado horizontalmente.
    3. Análisis integrado de Volumen Relativo (RelVol) y Rotación de Float.
    """
    # --------------------------------------------------------
    # 1. DESCARGA DE DATOS Y OBTENCIÓN DEL FLOAT
    # --------------------------------------------------------
    data = yf.download(
        tickers=ticker,
        start=from_date,
        end=to_date,
        auto_adjust=False, 
        progress=False
    )
    
    if data.empty:
        print(f"No se encontraron datos para el ticker {ticker}")
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    # Obtener información del Float mediante el objeto Ticker de yfinance
    try:
        ticker_info = yf.Ticker(ticker).info
        float_shares = ticker_info.get("floatShares", None)
    except Exception:
        float_shares = None

    # --------------------------------------------------------
    # 2. CÁLCULO MATEMÁTICO DE VWAP, RELVOL Y ROTACIÓN
    # --------------------------------------------------------
    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
    tp_v = typical_price * data["Volume"]
    
    rolling_tp_v = tp_v.rolling(window=vwap_window).sum()
    rolling_vol = data["Volume"].rolling(window=vwap_window).sum()
    
    data["VWAP"] = rolling_tp_v / rolling_vol
    
    rolling_variance = ((typical_price - data["VWAP"]) ** 2 * data["Volume"]).rolling(window=vwap_window).sum() / rolling_vol
    data["VWAP_Std"] = np.sqrt(rolling_variance)
    
    data["VWAP_Upper"] = data["VWAP"] + (num_std * data["VWAP_Std"])
    data["VWAP_Lower"] = data["VWAP"] - (num_std * data["VWAP_Std"])
    
    # --- Métricas Avanzadas de Volumen ---
    # Promedio diario normal del volumen (Media móvil simple del volumen)
    data["Vol_Avg"] = data["Volume"].rolling(window=vwap_window).mean()
    # Volumen Relativo (RelVol)
    data["RelVol"] = data["Volume"] / data["Vol_Avg"]
    
    # Rotación del Float (%) si el dato está disponible
    if float_shares:
        data["Float_Rotation_%"] = (data["Volume"] / float_shares) * 100
    else:
        data["Float_Rotation_%"] = np.nan

    data = data.dropna()

    # --------------------------------------------------------
    # 3. IDENTIFICACIÓN Y REPORTE DE VOLÚMENES ANORMALES
    # --------------------------------------------------------
    # Filtrar días con volumen anormalmente alto (RelVol >= 1.5)
    high_vol_days = data[data["RelVol"] >= 1.5].copy()
    
    # Construir el reporte/Dataframe analítico solicitado
    report_rows = []
    for date, row in high_vol_days.iterrows():
        date_str = date.strftime("%Y-%m-%d")
        rel_vol_val = round(row["RelVol"], 2)
        
        # Determinar la categoría de volumen según tus reglas
        vol_type = "Moderadamente Alto" if rel_vol_val <= 2.0 else "Muy Alto (Foco de Mercado)"
        
        # Determinar el impacto en precio (Subida vs Caída)
        price_change = row["Close"] - row["Open"]
        if price_change > 0:
            bias = "Subida con Convicción (Respaldo comprador institucional/retail)"
        else:
            bias = "Caída con Presión / Pánico (Fuerte liquidación de posiciones)"
            
        # Incluir análisis de rotación de Float
        float_txt = ""
        if not np.isnan(row["Float_Rotation_%"]):
            rot_val = round(row["Float_Rotation_%"], 1)
            float_txt = f" | Rotación de Float: {rot_val}%"
            if rot_val >= 100:
                float_txt += " [ALERTA: Manos cambiadas por completo]"

        reason = f"Volumen {vol_type} (X {rel_vol_val} del promedio). {bias}{float_txt}"
        
        report_rows.append({
            "Fecha (Año-Mes-Día)": date_str,
            "Volumen Diario": int(row["Volume"]),
            "RelVol": rel_vol_val,
            "Rotación Float (%)": round(row["Float_Rotation_%"], 2) if float_shares else "N/A",
            "Razón del Impacto": reason
        })
        
    df_reporte_volumen = pd.DataFrame(report_rows)
    
    # Mostrar el dataframe analítico en la consola al terminar el día simulado
    print(f"\n=== REPORTE DE DÍAS CON VOLUMEN ANORMALMENTE ALTO PARA {ticker.upper()} ===")
    if not df_reporte_volumen.empty:
        # Ordenamos de mayor a menor volumen relativo para destacar las fechas críticas
        df_reporte_volumen = df_reporte_volumen.sort_values(by="RelVol", ascending=False).reset_index(drop=True)
        print(df_reporte_volumen.to_string(index=False))
    else:
        print("No se registraron días con RelVol superior a 1.5 en este rango de fechas.")

    # --------------------------------------------------------
    # 4. CONFIGURACIÓN ESTILOS Y GRÁFICO
    # --------------------------------------------------------
    custom_market_colors = mpf.make_marketcolors(
        up=price_color, down=price_color,
        edge=price_color, wick=price_color,
        volume='gray', inherit=True
    )
    
    custom_style = mpf.make_mpf_style(
        marketcolors=custom_market_colors,
        gridstyle="--",
        y_on_right=False
    )

    additional_plots = [
        mpf.make_addplot(data["VWAP"], color=price_color, width=2.0, label="VWAP"),
        mpf.make_addplot(data["VWAP_Upper"], color=price_color, width=1.0, linestyle=":", alpha=0.5),
        mpf.make_addplot(data["VWAP_Lower"], color=price_color, width=1.0, linestyle=":", alpha=0.5)
    ]

    # Modificación del título dinámico si es Low Float
    title_text = f"\nMarket Profile: {ticker} (VWAP & Volume Profile)"
    if float_shares and float_shares < 20_000_000:
        title_text += " - ¡ALERTA: ACTIVO LOW FLOAT!"

    fig, axlist = mpf.plot(
        data,
        type="candle",
        style=custom_style,
        addplot=additional_plots,
        volume=True,
        title=title_text,
        ylabel="Price",
        ylabel_lower="Volume Traded",
        figsize=(14, 9),
        returnfig=True
    )

    ax_price = axlist[0]

    # Canal sombreado para bandas VWAP
    ax_price.fill_between(
        range(len(data)), 
        data["VWAP_Lower"],
        data["VWAP_Upper"],
        color=price_color,
        alpha=0.10,
        label="VWAP Band"
    )

    # Inyección Manual del Perfil de Volumen
    price_min = data['Low'].min()
    price_max = data['High'].max()
    bin_size = (price_max - price_min) / vol_profile_bins

    volume_profile = data['Volume'].groupby(
        data['Close'].apply(lambda x: bin_size * round(x / bin_size, 0))
    ).sum()

    vp_prices = volume_profile.index.values
    vp_volumes = volume_profile.values

    max_visible_width = len(data) * 0.25 
    scaled_volumes = (vp_volumes / vp_volumes.max()) * max_visible_width
    right_edge_x = len(data) - 1

    ax_price.barh(
        vp_prices,
        width=-scaled_volumes,       
        left=right_edge_x,         
        height=bin_size * 0.85,      
        align='center',
        color='gray',
        alpha=0.25,                 
        edgecolor='gray'
    )
    
    mpf.show()
    return df_reporte_volumen



def plot_advanced_market_profile_visual_performance(
    ticker,
    from_date="2024-01-01",
    to_date=None,
    vwap_window=20,           # Periodo para el VWAP acumulado y promedio de volumen
    num_std=2,                # Desviaciones estándar para las bandas VWAP
    price_color="blue",       # Color base para el activo (velas) y su VWAP
    vol_profile_bins=30,      # Cuántas barras horizontales tendrá el perfil
    mark_high_vol_days=False, # True para marcar automáticamente las fechas de alto volumen
    custom_dates_to_mark=None # Lista de strings ['YYYY-MM-DD'] para marcar fechas manualmente
):
    """
    Descarga datos de Yahoo Finance y grafica para UN activo:
    1. Gráfico de velas y Canal de VWAP con bandas de desviación estándar.
    2. Perfil de Volumen (Volume Profile) manual inyectado horizontalmente.
    3. Análisis integrado de Volumen Relativo (RelVol) y Rotación de Float.
    4. Marcado dinámico y tolerante de líneas verticales para fechas clave de volumen.
    """
    # --------------------------------------------------------
    # 1. DESCARGA DE DATOS Y OBTENCIÓN DEL FLOAT
    # --------------------------------------------------------
    data = yf.download(
        tickers=ticker,
        start=from_date,
        end=to_date,
        auto_adjust=False, 
        progress=False
    )
    
    if data.empty:
        print(f"No se encontraron datos para el ticker {ticker}")
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    try:
        ticker_info = yf.Ticker(ticker).info
        float_shares = ticker_info.get("floatShares", None)
    except Exception:
        float_shares = None

    # --------------------------------------------------------
    # 2. CÁLCULO MATEMÁTICO DE VWAP, RELVOL Y ROTACIÓN
    # --------------------------------------------------------
    typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
    tp_v = typical_price * data["Volume"]
    
    rolling_tp_v = tp_v.rolling(window=vwap_window).sum()
    rolling_vol = data["Volume"].rolling(window=vwap_window).sum()
    
    data["VWAP"] = rolling_tp_v / rolling_vol
    
    rolling_variance = ((typical_price - data["VWAP"]) ** 2 * data["Volume"]).rolling(window=vwap_window).sum() / rolling_vol
    data["VWAP_Std"] = np.sqrt(rolling_variance)
    
    data["VWAP_Upper"] = data["VWAP"] + (num_std * data["VWAP_Std"])
    data["VWAP_Lower"] = data["VWAP"] - (num_std * data["VWAP_Std"])
    
    data["Vol_Avg"] = data["Volume"].rolling(window=vwap_window).mean()
    data["RelVol"] = data["Volume"] / data["Vol_Avg"]
    
    if float_shares:
        data["Float_Rotation_%"] = (data["Volume"] / float_shares) * 100
    else:
        data["Float_Rotation_%"] = np.nan

    # Limpieza de valores nulos iniciales generados por la ventana móvil (Rolling Window)
    data = data.dropna().copy()

    # --- SOLUCIÓN AL ERROR INDEXERROR (VALIDACIÓN DE TAMAÑO) ---
    if data.empty or len(data) < 2:
        print(f"\n[ERROR] No hay suficientes datos históricos para {ticker} usando una ventana de {vwap_window} días.")
        print(f"Prueba expandiendo el rango inicial (ej. from_date='2024-11-01') para acumular el historial requerido.")
        return None

    # --------------------------------------------------------
    # 3. IDENTIFICACIÓN Y REPORTE DE VOLÚMENES ANORMALES
    # --------------------------------------------------------
    high_vol_days = data[data["RelVol"] >= 1.5].copy()
    
    report_rows = []
    detected_dates_str = [] 
    
    for date, row in high_vol_days.iterrows():
        date_str = date.strftime("%Y-%m-%d")
        detected_dates_str.append(date_str)
        rel_vol_val = round(row["RelVol"], 2)
        
        vol_type = "Moderadamente Alto" if rel_vol_val <= 2.0 else "Muy Alto (Foco de Mercado)"
        price_change = row["Close"] - row["Open"]
        
        if price_change > 0:
            bias = "Subida con Convicción (Respaldo comprador)"
        else:
            bias = "Caída con Presión / Pánico (Liquidación masiva)"
            
        float_txt = ""
        if not np.isnan(row["Float_Rotation_%"]):
            rot_val = round(row["Float_Rotation_%"], 1)
            float_txt = f" | Rotación de Float: {rot_val}%"
            if rot_val >= 100:
                float_txt += " [ALERTA: Volteo de Float]"

        reason = f"Volumen {vol_type} (X {rel_vol_val} del promedio). {bias}{float_txt}"
        
        report_rows.append({
            "Fecha (Año-Mes-Día)": date_str,
            "Volumen Diario": int(row["Volume"]),
            "RelVol": rel_vol_val,
            "Rotación Float (%)": round(row["Float_Rotation_%"], 2) if float_shares else "N/A",
            "Razón del Impacto": reason
        })
        
    df_reporte_volumen = pd.DataFrame(report_rows)
    
    print(f"\n=== REPORTE DE DÍAS CON VOLUMEN ANORMALMENTE ALTO PARA {ticker.upper()} ===")
    if not df_reporte_volumen.empty:
        df_reporte_volumen = df_reporte_volumen.sort_values(by="Fecha (Año-Mes-Día)", ascending=False).reset_index(drop=True)
        print(df_reporte_volumen.to_string(index=False))
    else:
        print("No se registraron días con RelVol superior a 1.5 en este rango de fechas para este activo.")

    # --------------------------------------------------------
    # 4. GESTIÓN DE LÍNEAS VERTICALES (Vlines)
    # --------------------------------------------------------
    vlines_dict = None
    dates_to_draw = []

    if mark_high_vol_days:
        dates_to_draw.extend(detected_dates_str)
        
    if custom_dates_to_mark is not None:
        if isinstance(custom_dates_to_mark, list):
            dates_to_draw.extend(custom_dates_to_mark)
        else:
            dates_to_draw.append(str(custom_dates_to_mark))

    dates_to_draw = list(set(dates_to_draw))
    valid_dates = [pd.to_datetime(d) for d in dates_to_draw if pd.to_datetime(d) in data.index]

    if valid_dates:
        vlines_dict = dict(vlines=valid_dates, colors='red', linewidths=1.2, linestyle='--')

    # --------------------------------------------------------
    # 5. CONFIGURACIÓN ESTILOS Y GRÁFICO (CORREGIDO)
    # --------------------------------------------------------
    custom_market_colors = mpf.make_marketcolors(
        up=price_color, down=price_color,
        edge=price_color, wick=price_color,
        volume='gray', inherit=True
    )
    
    custom_style = mpf.make_mpf_style(
        marketcolors=custom_market_colors,
        gridstyle="--",
        y_on_right=False
    )

    additional_plots = [
        mpf.make_addplot(data["VWAP"], color=price_color, width=2.0, label="VWAP"),
        mpf.make_addplot(data["VWAP_Upper"], color=price_color, width=1.0, linestyle=":", alpha=0.5),
        mpf.make_addplot(data["VWAP_Lower"], color=price_color, width=1.0, linestyle=":", alpha=0.5)
    ]

    title_text = f"\nMarket Profile: {ticker} (VWAP & Volume Profile)"
    if float_shares and float_shares < 20_000_000:
        title_text += " - ¡ALERTA: ACTIVO LOW FLOAT!"

    # Generamos los parámetros dinámicos para evitar el fallo por valor None
    plot_kwargs = dict(
        data=data,
        type="candle",
        style=custom_style,
        addplot=additional_plots,
        volume=True,
        title=title_text,
        ylabel="Price",
        ylabel_lower="Volume Traded",
        figsize=(14, 9),
        returnfig=True
    )

    # Solo insertamos la clave vlines si contiene datos válidos
    if vlines_dict is not None:
        plot_kwargs['vlines'] = vlines_dict

    # Desempaquetado seguro
    fig, axlist = mpf.plot(**plot_kwargs)
    ax_price = axlist[0]

    # Canal sombreado para bandas VWAP
    ax_price.fill_between(
        range(len(data)), 
        data["VWAP_Lower"],
        data["VWAP_Upper"],
        color=price_color,
        alpha=0.10,
        label="VWAP Band"
    )

    # Inyección Manual del Perfil de Volumen
    price_min = data['Low'].min()
    price_max = data['High'].max()
    bin_size = (price_max - price_min) / vol_profile_bins

    volume_profile = data['Volume'].groupby(
        data['Close'].apply(lambda x: bin_size * round(x / bin_size, 0))
    ).sum()

    vp_prices = volume_profile.index.values
    vp_volumes = volume_profile.values

    max_visible_width = len(data) * 0.25 
    scaled_volumes = (vp_volumes / vp_volumes.max()) * max_visible_width
    right_edge_x = len(data) - 1

    ax_price.barh(
        vp_prices,
        width=-scaled_volumes,       
        left=right_edge_x,         
        height=bin_size * 0.85,      
        align='center',
        color='gray',
        alpha=0.25,                 
        edgecolor='gray'
    )
    
    mpf.show()
    return df_reporte_volumen


def detect_volume_confluence_bombs(
    ticker,
    from_date="2024-01-01",
    to_date=None,
    vol_profile_bins=50,
    volume_z_threshold=2.5,  # Cuántas desviaciones estándar para considerar un "pico"
    hvn_percentile=70,  # Top % de barras del profile que consideramos "Zona de Alto Volumen"
):
    """Analiza un activo buscando confluencias entre picos de volumen en el tiempo

    y zonas de alta densidad en el Volume Profile (HVN).
    Devuelve un DataFrame con las señales y emite alertas para los últimos 3
    días.
    """
    # --------------------------------------------------------
    # 1. DESCARGA DE DATOS
    # --------------------------------------------------------
    data = yf.download(
        tickers=ticker,
        start=from_date,
        end=to_date,
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        print(f"[-] No se encontraron datos para el ticker {ticker}")
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    df = data.copy()

    # --------------------------------------------------------
    # 2. ANÁLISIS 1: PICOS DE VOLUMEN (Z-SCORE)
    # --------------------------------------------------------
    # Usamos una ventana de 20 días para medir la anomalía del volumen actual
    df["Vol_Mean"] = df["Volume"].rolling(window=20).mean()
    df["Vol_Std"] = df["Volume"].rolling(window=20).std()
    # Z-Score: Mide a cuántas desviaciones estándar está el volumen de hoy
    df["Vol_ZScore"] = (df["Volume"] - df["Vol_Mean"]) / df["Vol_Std"]

    # Marcamos si cumple la condición de pico matemático
    df["Is_Volume_Peak"] = df["Vol_ZScore"] >= volume_z_threshold

    # --------------------------------------------------------
    # 3. ANÁLISIS 2: RECONOCIMIENTO DEL VOLUME PROFILE (ZONAS HVN)
    # --------------------------------------------------------
    price_min = df["Low"].min()
    price_max = df["High"].max()
    bin_size = (price_max - price_min) / vol_profile_bins

    # Generamos el perfil de volumen acumulado de todo el periodo
    # Mapeamos cada cierre a su respectivo "bin" o cubeta de precio
    df["Price_Bin"] = df["Close"].apply(
        lambda x: bin_size * round(x / bin_size, 0)
    )
    volume_profile = df.groupby("Price_Bin")["Volume"].sum()

    # Identificamos el umbral para ser considerado High Volume Node (HVN)
    # Las zonas que están en el percentil superior (ej. top 30% con más volumen)
    hvn_threshold = np.percentile(volume_profile.values, hvn_percentile)

    # Creamos un set de precios que son considerados "Zonas Explosivas de Carga"
    hvn_bins = volume_profile[volume_profile >= hvn_threshold].index.tolist()

    # Verificamos si el precio de cada día cerró dentro de un nodo de alto volumen
    df["Is_In_HVN"] = df["Price_Bin"].isin(hvn_bins)

    # --------------------------------------------------------
    # 4. CONFLUENCIA ("LA BOMBA") Y SCORE DE INTENSIDAD
    # --------------------------------------------------------
    # La condición requiere que ocurran ambas cosas el mismo día
    df["Bomba_Detectada"] = df["Is_Volume_Peak"] & df["Is_In_HVN"]

    # Creamos un score armónico: Z-Score de volumen multiplicado por la importancia del precio
    # Si no hay confluencia, el score es 0
    max_vp_vol = volume_profile.max()
    df["Bomba_Score"] = 0.0

    for idx, row in df.iterrows():
        if row["Bomba_Detectada"]:
            # Proporción de volumen que tiene ese nivel de precio (de 0 a 1)
            bin_weight = volume_profile[row["Price_Bin"]] / max_vp_vol
            # Score = Multiplicación de la fuerza del impacto por el peso de la zona histórica
            df.at[idx, "Bomba_Score"] = round(
                row["Vol_ZScore"] * bin_weight, 2
            )

    # --------------------------------------------------------
    # 5. MONITOREO DE LOS ÚLTIMOS 3 DÍAS Y ALERTAS
    # --------------------------------------------------------
    print(f"\n=== ANÁLISIS DE CONFLUENCIA PARA {ticker.upper()} ===")
    print(
        f"Periodo analizado: {df.index.min().strftime('%Y-%m-%d')} al {df.index.max().strftime('%Y-%m-%d')}"
    )

    ultimos_3_dias = df.tail(3)
    bomba_reciente = ultimos_3_dias[ultimos_3_dias["Bomba_Detectada"] == True]

    if not bomba_reciente.empty:
        print(
            f"\n🔥 [ALERTA] ¡SE DETECTÓ UNA BOMBA EN LOS ÚLTIMOS 3 DÍAS PARA {ticker}! 🔥"
        )
        for fecha, fila in bomba_reciente.iterrows():
            print(f"  • Fecha: {fecha.strftime('%Y-%m-%d')}")
            print(f"    - Precio Cierre: ${fila['Close']:.2f}")
            print(
                f"    - Pico de Volumen (Z-Score): {fila['Vol_ZScore']:.2f} σ"
            )
            print(f"    - Score de la Bomba: {fila['Bomba_Score']} pts")
            print(
                f"    - Nota: El precio está absorbiendo volumen masivo en una zona de alta aceptación histórica."
            )
    else:
        print(
            f"\n[-] No se detectaron patrones de 'Bomba' activos en los últimos 3 días para {ticker}."
        )
        # Mostramos cuándo fue la última vez que ocurrió para dar contexto
        todas_las_bombas = df[df["Bomba_Detectada"] == True]
        if not todas_las_bombas.empty:
            ultima_fecha = todas_las_bombas.index[-1]
            print(
                f"    La última confluencia registrada fue el: {ultima_fecha.strftime('%Y-%m-%d')} (Score: {todas_las_bombas['Bomba_Score'].iloc[-1]})"
            )

    # Limpiamos el DataFrame final para devolver solo lo valioso para tu análisis
    columnas_interes = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Vol_ZScore",
        "Is_Volume_Peak",
        "Is_In_HVN",
        "Bomba_Detectada",
        "Bomba_Score",
    ]

    return df[columnas_interes]

def scan_market_for_bombs(lista_tickers, dias_atras=3, vol_profile_bins=50):
    """Escanea una lista de activos y devuelve un DataFrame consolidado

    únicamente con aquellos que activaron la 'bomba' de confluencia
    en los últimos días especificados.
    """
    nominados = []

    print(
        f"🔍 Iniciando escaneo de {len(lista_tickers)} activos en busca de confluencias institucionales...\n"
    )

    for ticker in lista_tickers:
        try:
            # Reutilizamos la función base ajustando los parámetros
            # Se usa un bloque try/except para evitar caídas si un ticker falla o no tiene datos
            df_activo = detect_volume_confluence_bombs(
                ticker=ticker,
                from_date="2024-01-01",
                vol_profile_bins=vol_profile_bins,
                volume_z_threshold=2.5,
                hvn_percentile=70,
            )

            if df_activo is not None and not df_activo.empty:
                # Filtrar solo los últimos días del activo
                ultimos_dias = df_activo.tail(dias_atras)
                bombas_encontradas = ultimos_dias[
                    ultimos_dias["Bomba_Detectada"] == True
                ]

                # Si el activo arrojó señales en este periodo, extraemos sus datos básicos
                for fecha, fila in bombas_encontradas.iterrows():
                    nominados.append(
                        {
                            "Ticker": ticker,
                            "Fecha_Señal": fecha.strftime("%Y-%m-%d"),
                            "Precio_Cierre": round(fila["Close"], 2),
                            "Vol_ZScore": round(fila["Vol_ZScore"], 2),
                            "Bomba_Score": fila["Bomba_Score"],
                        }
                    )
        except Exception as e:
            # Mantiene el bucle corriendo de forma silenciosa ante errores de descarga puntuales
            continue

    # Convertimos la lista de diccionarios en el DataFrame consolidado de salida
    if nominados:
        df_nominados = pd.DataFrame(nominados)
        # Ordenamos de mayor a menor según el Bomba_Score para ver las mejores oportunidades arriba
        df_nominados = df_nominados.sort_values(
            by="Bomba_Score", ascending=False
        ).reset_index(drop=True)
        print(
            f"\n🎉 [PROCESO COMPLETADO] Escaneo finalizado. Se encontraron {len(df_nominados)} señales válidas."
        )
        return df_nominados
    else:
        print(
            "\n[-] Escaneo finalizado. Ningún activo de la lista cumplió los criterios de confluencia en los últimos 3 días."
        )
        return pd.DataFrame(
            columns=[
                "Ticker",
                "Fecha_Señal",
                "Precio_Cierre",
                "Vol_ZScore",
                "Bomba_Score",
            ]
        )




def calcular_indicadores_base(data, stoch_k=5, stoch_d=9, stoch_s=2, rsi_per=14, roc_per=12, atr_per=14, adx_per=14):
    """Calcula todas las series matemáticas requeridas para los indicadores."""
    df = data.copy()
    
    # 1. Medias Móviles
    df['EMA_89'] = df['Close'].ewm(span=89, adjust=False).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # 2. Oscilador Estocástico Completo (%K, %D, %D Suavizado)
    low_min = df['Low'].rolling(window=stoch_k).min()
    high_max = df['High'].rolling(window=stoch_k).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=stoch_d).mean()
    df['Stoch_Smooth'] = df['Stoch_D'].rolling(window=stoch_s).mean()
    
    # 3. RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_per).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_per).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 4. ROC (Rate of Change)
    df['ROC'] = ((df['Close'] - df['Close'].shift(roc_per)) / df['Close'].shift(roc_per)) * 100
    
    # 5. ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=atr_per).mean()
    
    # 6. ADX (Average Directional Index)
    up_move = df['High'].diff()
    down_move = df['Low'].shift() - df['Low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(window=adx_per).mean() / df['ATR'])
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(window=adx_per).mean() / df['ATR'])
    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di))
    df['ADX'] = pd.Series(dx, index=df.index).rolling(window=adx_per).mean()
    
    return df.dropna()




def calcular_indicadores_avanzados(data, stoch_k=5, stoch_d=9, stoch_s=2):
    """Calcula todas las series matemáticas incluyendo MACD, Bollinger y Medias."""
    df = data.copy()
    
    # --- 1. NUEVAS MEDIAS MÓVILES SOLICITADAS ---
    # EMAs (9, 21, 54, 89)
    df['EMA_9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA_21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA_54'] = df['Close'].ewm(span=54, adjust=False).mean()
    df['EMA_89'] = df['Close'].ewm(span=89, adjust=False).mean()
    # SMAs (10, 50, 200)
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # --- 2. BANDAS DE BOLLINGER (Basadas en EMA 20 + 2 Desviaciones Estándar) ---
    df['BB_Base'] = df['Close'].ewm(span=20, adjust=False).mean() # Tu regla pide EMA 20
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Base'] + (2 * bb_std)
    df['BB_Lower'] = df['BB_Base'] - (2 * bb_std)
    
    # --- 3. MACD (12, 26, 9) ---
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # --- 4. OSCILADOR ESTOCÁSTICO (Aquí está guardado) ---
    low_min = df['Low'].rolling(window=stoch_k).min()
    high_max = df['High'].rolling(window=stoch_k).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=stoch_d).mean()
    df['Stoch_Smooth'] = df['Stoch_D'].rolling(window=stoch_s).mean()
    
    return df.dropna()



def plot_trend_and_macd(df_calculado, ticker, usar_emas=True):
    """
    Grafica las velas, canales de Bollinger, volumen y el panel inferior MACD.
    usar_emas = True -> Grafica EMAs (9, 21, 54, 89)
    usar_emas = False -> Grafica SMAs (10, 50, 200)
    """
    df = df_calculado.copy()
    additional_plots = []
    
    # 1. Configurar cuáles medias se inyectan en el gráfico de precio
    if usar_emas:
        additional_plots += [
            mpf.make_addplot(df['EMA_9'], color='cyan', width=1.0, label='EMA 9'),
            mpf.make_addplot(df['EMA_21'], color='blue', width=1.0, label='EMA 21'),
            mpf.make_addplot(df['EMA_54'], color='darkorange', width=1.2, label='EMA 54'),
            mpf.make_addplot(df['EMA_89'], color='red', width=1.5, label='EMA 89')
        ]
    else:
        additional_plots += [
            mpf.make_addplot(df['SMA_10'], color='lightblue', width=1.0, label='SMA 10'),
            mpf.make_addplot(df['SMA_50'], color='purple', width=1.2, label='SMA 50'),
            mpf.make_addplot(df['SMA_200'], color='black', width=1.8, label='SMA 200')
        ]
        
    # 2. Inyectar las Bandas de Bollinger al gráfico principal (Líneas punteadas)
    additional_plots += [
        mpf.make_addplot(df['BB_Upper'], color='gray', linestyle='--', width=1.0),
        mpf.make_addplot(df['BB_Lower'], color='gray', linestyle='--', width=1.0)
    ]
    
    # 3. Inyectar el MACD en un subgráfico independiente (Panel 2)
    additional_plots += [
        mpf.make_addplot(df['MACD'], panel=2, color='blue', width=1.2, label='MACD (12,26)'),
        mpf.make_addplot(df['MACD_Signal'], panel=2, color='orange', width=1.0, linestyle='-', label='Señal (9)'),
        mpf.make_addplot(df['MACD_Hist'], panel=2, type='bar', color='gray', alpha=0.5, label='Hist')
    ]
    
    # Estilo visual de las velas
    custom_colors = mpf.make_marketcolors(up='green', down='red', edge='inherit', wick='inherit', volume='silver')
    custom_style = mpf.make_mpf_style(marketcolors=custom_colors, gridstyle='--', y_on_right=False)
    
    # 4. Construcción y renderizado del gráfico
    fig, axlist = mpf.plot(
        df,
        type='candle',
        style=custom_style,
        addplot=additional_plots,
        volume=True, # El volumen se dibuja automáticamente en el Panel 1 intermediario
        title=f"\nEstrategia de Tendencia, Volatilidad y MACD: {ticker}",
        ylabel="Precio",
        ylabel_lower="Volumen",
        figsize=(14, 10),
        returnfig=True
    )
    
    # Interceptamos el eje principal de precios para rellenar el canal de Bollinger
    ax_price = axlist[0]
    ax_price.fill_between(
        range(len(df)),
        df['BB_Lower'],
        df['BB_Upper'],
        color='gray',
        alpha=0.05 # Sombreado ultra transparente para el canal de volatilidad
    )
    
    # Forzar la aparición de la simbología descriptiva
    ax_price.legend(loc="upper left")
    if len(axlist) > 2:
        axlist[2].legend(loc="upper left") # Leyenda del panel MACD
        
    mpf.show()


# LOS SIGUIENTES SON AMS FACILES DE USAR que las previas 3 funciones... 
'''
'''



def plot_advanced_trend_profile(ticker, from_date="2025-01-01", to_date=None, vol_profile_bins=30):
    """
    GRAFICA EL PRECIO STRUCTURAL:
    - Velas Japonesas + Volumen Tradicional.
    - Medias Móviles: EMA 89, SMA 50, SMA 200.
    - Retroceso de Fibonacci (calculado sobre el rango visual).
    - Perfil de Volumen en el extremo derecho.
    """
    # 1. Descarga de datos
    data = yf.download(ticker, start=from_date, end=to_date, auto_adjust=False, progress=False)
    if data.empty:
        return print(f"No hay datos para {ticker}")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)

    # 2. Cálculo de Medias Móviles
    data["EMA_89"] = data["Close"].ewm(span=89, adjust=False).mean()
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["SMA_200"] = data["Close"].rolling(window=200).mean()
    data = data.dropna(subset=["EMA_89"]) # Limpieza inicial básica

    # 3. Configuración de líneas adicionales en el gráfico principal
    additional_plots = [
        mpf.make_addplot(data["EMA_89"], color="darkorange", width=1.5, label="EMA 89 (Mensual)"),
        mpf.make_addplot(data["SMA_50"], color="blue", width=1.5, label="SMA 50"),
        mpf.make_addplot(data["SMA_200"], color="purple", width=1.5, label="SMA 200")
    ]

    custom_market_colors = mpf.make_marketcolors(up="green", down="red", edge="inherit", wick="inherit", volume="gray")
    custom_style = mpf.make_mpf_style(marketcolors=custom_market_colors, gridstyle="--", y_on_right=False)

    # 4. Graficado con interceptor de Matplotlib
    fig, axlist = mpf.plot(
        data, type="candle", style=custom_style, addplot=additional_plots, volume=True,
        title=f"\nPerfil de Tendencia Estructural y Niveles: {ticker}", ylabel="Precio",
        ylabel_lower="Volumen", figsize=(14, 9), returnfig=True
    )
    ax_price = axlist[0]

    # 5. Cálculo e inyección de Retrocesos de Fibonacci (Rango Visible)
    high_max = data["High"].max()
    low_min = data["Low"].min()
    diff = high_max - low_min
    
    niveles_fibo = {
        "0.0% (Max)": high_max,
        "38.2%": high_max - 0.382 * diff,
        "50.0%": high_max - 0.500 * diff,
        "61.8%": high_max - 0.618 * diff,
        "100.0% (Min)": low_min
    }
    
    colores_fibo = ["red", "orange", "green", "blue", "purple"]
    for i, (nombre, precio) in enumerate(niveles_fibo.items()):
        ax_price.axhline(precio, color=colores_fibo[i], linestyle="-.", alpha=0.6, lw=1)
        ax_price.text(0, precio, f" Fibo {nombre}: {precio:.2f}", color=colores_fibo[i], va="bottom", fontsize=9)

    # 6. Perfil de Volumen (Histograma Horizontal Derecho)
    price_min, price_max = data["Low"].min(), data["High"].max()
    bin_size = (price_max - price_min) / vol_profile_bins
    volume_profile = data["Volume"].groupby(data["Close"].apply(lambda x: bin_size * round(x / bin_size, 0))).sum()
    
    vp_prices = volume_profile.index.values
    scaled_volumes = (volume_profile.values / volume_profile.max()) * (len(data) * 0.20)
    right_edge_x = len(data) - 1

    ax_price.barh(vp_prices, width=-scaled_volumes, left=right_edge_x, height=bin_size * 0.8, color="gray", alpha=0.2)
    ax_price.legend(loc="upper left")
    mpf.show()


def plot_advanced_oscillators(ticker, from_date="2025-01-01", to_date=None, rsi_p=14, est_k=5, est_d=9, est_s=2, adx_p=14, roc_p=12, atr_p=14):
    """
    CALCULA Y GRAFICA EL PANEL DE OSCILADORES:
    - RSI / Estocástico (5,9,2) / ADX / ROC / ATR.
    """
    data = yf.download(ticker, start=from_date, end=to_date, auto_adjust=False, progress=False)
    if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.droplevel(1)

    # --- MATEMÁTICAS INTERNAS DE LOS INDICADORES ---
    # RSI
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_p).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_p).mean()
    data["RSI"] = 100 - (100 / (1 + (gain / loss)))

    # Estocástico (5, 9, 2)
    low_k = data["Low"].rolling(window=est_k).min()
    high_k = data["High"].rolling(window=est_k).max()
    data["%K_rapido"] = 100 * ((data["Close"] - low_k) / (high_k - low_k))
    data["%K"] = data["%K_rapido"].rolling(window=est_s).mean() # Suavizado interno (2)
    data["%D"] = data["%K"].rolling(window=est_d).mean()       # Línea de señal (9)

    # ROC (Rate of Change)
    data["ROC"] = ((data["Close"] - data["Close"].shift(roc_p)) / data["Close"].shift(roc_p)) * 100

    # ATR (Average True Range)
    tr1 = data["High"] - data["Low"]
    tr2 = abs(data["High"] - data["Close"].shift(1))
    tr3 = abs(data["Low"] - data["Close"].shift(1))
    data["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    data["ATR"] = data["TR"].rolling(window=atr_p).mean()

    # ADX
    up_move = data["High"].diff()
    down_move = data["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    tr_smooth = data["TR"].rolling(window=adx_p).sum()
    plus_di = 100 * (pd.Series(plus_dm, index=data.index).rolling(window=adx_p).sum() / tr_smooth)
    minus_di = 100 * (pd.Series(minus_dm, index=data.index).rolling(window=adx_p).sum() / tr_smooth)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    data["ADX"] = dx.rolling(window=adx_p).mean()

    data = data.dropna()

    # --- DISEÑO DE SUBPLOTS VERTICALES ---
    fig, axs = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    
    # 1. RSI
    axs[0].plot(data.index, data["RSI"], color="purple", label=f"RSI ({rsi_p})")
    axs[0].axhline(70, color="red", linestyle="--", alpha=0.5)
    axs[0].axhline(30, color="green", linestyle="--", alpha=0.5)
    axs[0].set_title(f"Panel de Osciladores y Fuerza para: {ticker}")
    
    # 2. Estocástico
    axs[1].plot(data.index, data["%K"], color="blue", label=f"%K ({est_k},{est_s})")
    axs[1].plot(data.index, data["%D"], color="orange", linestyle="--", label=f"%D ({est_d})")
    axs[1].axhline(80, color="gray", linestyle=":")
    axs[1].axhline(20, color="gray", linestyle=":")

    # 3. ADX
    axs[2].plot(data.index, data["ADX"], color="black", lw=2, label="ADX (Fuerza de Tendencia)")
    axs[2].axhline(25, color="red", linestyle="-.", alpha=0.6, label="Umbral de Fuerza (>25)")

    # 4. ROC
    axs[3].plot(data.index, data["ROC"], color="cyan", label=f"ROC ({roc_p}) Momento")
    axs[3].axhline(0, color="gray", linestyle="-")

    # 5. ATR
    axs[4].plot(data.index, data["ATR"], color="magenta", label=f"ATR ({atr_p}) Volatilidad")

    for ax in axs:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")
    
    plt.xlabel("Fecha")
    plt.tight_layout()
    plt.show()
    

import matplotlib.patches as patches

def plot_equivolume_candles(ticker, from_date="2026-01-01", to_date=None, price_color="blue"):
    """
    Descarga datos de Yahoo Finance y grafica barras Equivolume:
    - Alto y Bajo (eje Y) definidos por High y Low del precio.
    - El ANCHO de cada barra (eje X) está definido de forma proporcional al Volumen.
    - El color del borde/relleno cambia según el cierre (Verde = Alcista, Rojo = Bajista).
    """
    # 1. Descarga y Limpieza de Datos de Yahoo Finance
    data = yf.download(
        tickers=ticker,
        start=from_date,
        end=to_date,
        auto_adjust=False,
        progress=False
    )
    
    if data.empty:
        return print(f"No se encontraron datos para {ticker}")
        
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
        
    # Quedarnos solo con las filas necesarias para que el gráfico no se sature visualmente
    df = data.copy().dropna()
    
    # --------------------------------------------------------
    # 2. CÁLCULO DE ANCHOS DINÁMICOS DE BARRA (Mecánica Equivolume)
    # --------------------------------------------------------
    # Calculamos el volumen relativo para que sirva de escala en el eje X
    total_volume = df['Volume'].sum()
    
    # El ancho de cada barra es el porcentaje de volumen diario respecto al promedio o total
    # Multiplicamos por un factor de escala para que las coordenadas de X sean legibles
    df['Width'] = (df['Volume'] / df['Volume'].mean()) * 0.8
    
    # Para poder graficar una barra al lado de la otra con anchos variables,
    # el eje X ya no puede ser una fecha fija, sino una coordenada acumulativa:
    x_centers = []
    current_x = 0.0
    
    for width in df['Width']:
        # El centro de la barra actual estará a la mitad de su propio ancho
        x_centers.append(current_x + width / 2)
        # La siguiente barra comenzará inmediatamente donde termina esta
        current_x += width + 0.1 # 0.1 de separación mínima para que no se encimen por completo

    df['X_Center'] = x_centers

    # --------------------------------------------------------
    # 3. RENDERIZADO DEL GRÁFICO (Matplotlib puro para control total)
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Dibujar cada caja Equivolume
    for idx, row in df.iterrows():
        # Determinar si el día fue alcista o bajista para el color
        color = "green" if row['Close'] >= row['Open'] else "red"
        
        # Calcular esquinas para el rectángulo de Matplotlib
        # Alto: de Low a High del precio
        bottom = row['Low']
        height = row['High'] - row['Low']
        width = row['Width']
        left = row['X_Center'] - width / 2
        
        # Crear la caja de la barra
        rect = patches.Rectangle(
            (left, bottom), 
            width, 
            height, 
            linewidth=1.2,
            edgecolor=color,
            facecolor=color,
            alpha=0.4 # Transparencia para ver la cuadrícula de fondo
        )
        ax.add_patch(rect)
        
        # Dibujar una línea interna para el precio de Cierre (Close) para referencia visual
        ax.plot([left, left + width], [row['Close'], row['Close']], color=color, lw=2)

    # --------------------------------------------------------
    # 4. FORMATEO DE EJES DE TIEMPO (Eje X dinámico)
    # --------------------------------------------------------
    # Como el eje X ahora son números acumulados, seleccionamos algunos puntos 
    # espaciados para poner las fechas reales de Yahoo Finance como etiquetas
    num_etiquetas = 8
    indices_muestreo = np.linspace(0, len(df) - 1, num_etiquetas, dtype=int)
    
    ticks_x = df['X_Center'].iloc[indices_muestreo].values
    labels_x = df.index[indices_muestreo].strftime('%Y-%m-%d')
    
    ax.set_xticks(ticks_x)
    ax.set_xticklabels(labels_x, rotation=15)
    
    # Límites del gráfico para que no se corte
    ax.set_xlim(-0.5, current_x + 0.5)
    ax.set_ylim(df['Low'].min() * 0.98, df['High'].max() * 1.02)
    
    # Configuración de diseño y textos
    ax.set_title(f"Gráfico Equivolume (Precio integrado con Volumen): {ticker}", fontsize=14, fontweight='bold')
    ax.set_ylabel("Escala de Precios")
    ax.set_xlabel("Línea de Tiempo (Espaciada por Presión de Volumen)")
    ax.grid(True, linestyle="--", alpha=0.3)
    
    # Simbología manual explicativa
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=4, alpha=0.6, label='Barra Alcista (Ancho = Más Volumen)'),
        Line2D([0], [0], color='red', lw=4, alpha=0.6, label='Barra Bajista (Ancho = Más Volumen)'),
    ]
    ax.legend(handles=legend_elements, loc="upper left")
    
    plt.tight_layout()
    plt.show()
    
    return df



import base64
import os
from openai import OpenAI 
# SE NECESITA PAGAR PARA PODER ACCEDER A LOS TOCKENS DE OPENAI
def analizar_grafica_con_openai(ticker, from_date="2025-01-01"): 
    """
    Descarga datos, genera la gráfica en el disco y utiliza la API de OpenAI
    (ChatGPT) con capacidades de visión para evaluar soportes, resistencias y Wyckoff.
    """
    # --------------------------------------------------------
    # 1. GENERAR Y GUARDAR LA GRÁFICA (Igual que antes)
    # --------------------------------------------------------
    data = yf.download(ticker, start=from_date, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
        
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data["Close"], color="black", lw=1.5, label=f"Precio {ticker}")
    plt.title(f"Historial de Precios - {ticker}")
    plt.xlabel("Fecha")
    plt.ylabel("Precio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    filename = "temp_market_chart.png"
    plt.savefig(filename, bbox_inches='tight', dpi=150)
    plt.close()

    # --------------------------------------------------------
    # 2. CODIFICAR LA IMAGEN A BASE64
    # --------------------------------------------------------
    with open(filename, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
        
    if os.path.exists(filename):
        os.remove(filename)

    # --------------------------------------------------------
    # 3. ENVIAR LA IMAGEN A OPENAI (CHATGPT)
    # --------------------------------------------------------
    # Introduce aquí tu API Key de OpenAI (Empieza con "sk-proj-...")
    client = OpenAI(api_key="sk-proj-TU-CLAVE-DE-OPENAI-AQUI")

    prompt = f"""
    Eres un experto analista cuantitativo y especialista en la metodología de Richard Wyckoff.
    Analiza visualmente la gráfica adjunta del activo {ticker} y responde de forma estructurada:
    
    1. Soportes y Resistencias Críticos: Identifica visualmente las 2 zonas de precio más importantes.
    2. Líneas de Tendencia: Determina si hay una dirección dominante y describe los puntos de origen.
    3. Ciclo de Wyckoff: Determina si el activo está en Acumulación, Markup (Subida), Distribución o Markdown (Bajada).
    """

    print(f"Enviando gráfica de {ticker} a ChatGPT para análisis de visión...")
    
    # Usamos gpt-4o, el estándar con mejor capacidad de comprensión de imágenes y gráficas
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1000,
        temperature=0.2,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
    )

    print("\n--- ANÁLISIS DE MERCADO INTELIGENTE (OPENAI) ---")
    veredicto = response.choices[0].message.content
    print(veredicto)
    
    return veredicto



class model:
    def __init__(self, security_x, security_y, decimals = 5, model_type = 'statistics'):
        self.security_x = security_x
        self.security_y = security_y
        self.decimals = decimals
        self.model_type = model_type
        self.x_type_column = None
        self.y_type_column = None
        self.timeseries = None
        self.x = None
        self.y = None
        self.std_x = None
        self.std_y = None
        self.beta = None
        self.alpha = None
        self.p_value = None
        self.correlation = None
        self.r_squared = None
        self.hypothesis_null = None
        self.predictor_linreg = None
        self.residuals = None
        self.n = None
        self.y_mean = None
        self.x_mean = None
        self.sxx = None
        self.sxy = None
        self.syy = None
        self.MCO = None
        self.mco_model_variance = None
        self.b0_variance = None
        self.b1_variance = None
        self.b0_interval = None
        self.b1_interval = None
        self.df = None
        self.suma_cuadrados_totales = None
        self.suma_cuadrados_regresion = None
        self.suma_cuadrados_errores = None
        self.cuadrados_medios_regresion = None
        self.cuadrados_medios_errores = None
        self.cuadrados_medios_totales = None
        self.F_test = None
        
    def synchronise_timeseries(self, from_date='aaaa-mm-dd', to_date='aaaa-mm-dd',
                                   log_returns=False):
        """
        Equivalente a synchronise_timeseries() pero descarga datos desde
        Yahoo Finance en lugar de leer CSVs locales.

        Detecta automáticamente si el ticker es de precio o tasa con
        target_column_yf(), luego llama a synchronise_timseries_yf() y
        aplica las mismas transformaciones (estandarización en 'statistics',
        solo estadísticos en 'macro').

        Elimina huecos (días festivos / mercado cerrado) vía inner join
        + dropna() dentro de synchronise_timseries_yf.
        """
        self.x_type_column = target_column_yf(self.security_x)
        self.y_type_column = target_column_yf(self.security_y)

        self.timeseries = synchronise_timeseries(
            self.security_x, self.security_y,
            columna1    = self.x_type_column,
            columna2    = self.y_type_column,
            from_date   = from_date,
            to_date     = to_date,
            log_returns = log_returns,
            model_type  = self.model_type
        )

        if self.model_type == 'statistics':
            mean_x = self.timeseries[self.security_x].mean()
            self.std_x = self.timeseries[self.security_x].std()
            self.timeseries[self.security_x] = (
                self.timeseries[self.security_x] - mean_x
            ) / self.std_x

            mean_y = self.timeseries[self.security_y].mean()
            self.std_y = self.timeseries[self.security_y].std()
            self.timeseries[self.security_y] = (
                self.timeseries[self.security_y] - mean_y
            ) / self.std_y

        elif self.model_type in ['macro', 'statistics']:
            self.std_x = self.timeseries[self.security_x].std()
            self.std_y = self.timeseries[self.security_y].std()

        self.n  = len(self.timeseries)
        self.df = self.n - 2

        if self.timeseries.empty:
            print(f'No hay datos para {self.security_x} y {self.security_y}')

    def plot_timeseries(self, secondary_y = True):
        plot_timeseries(self.timeseries, self.security_x, self.security_y, secondary_y=secondary_y)
        
    def compute_linear_reg(self):
        self.x = self.timeseries[self.security_x].values
        self.y = self.timeseries[self.security_y].values
        self.y_mean = np.mean(self.y)
        self.x_mean = np.mean(self.x)
        self.sxx = np.sum((self.x - self.x_mean)**2)
        self.sxy = np.sum((self.x - self.x_mean)*(self.y - self.y_mean))
        self.syy = np.sum((self.y - self.y_mean)**2)
        
        
        # Lineal Regression 
        slope_beta, intercept_alpha, correl_r, p_value, standard_error = st.linregress(x=self.x, y=self.y)
        self.beta = np.round(slope_beta, self.decimals)
        self.alpha = np.round(intercept_alpha, self.decimals)
        self.p_value = np.round(p_value, self.decimals)
        self.correlation = np.round(correl_r, self.decimals)
        self.r_squared = np.round(correl_r**2, self.decimals)
        self.hypothesis_null = p_value > 0.05
        self.predictor_linreg = intercept_alpha + slope_beta * self.x
        
        self.residuals = self.y - self.predictor_linreg
        self.MCO = np.sum(self.residuals**2)
        self.mco_model_variance = np.sum((self.residuals)**2)/(self.n-2)
        self.b0_variance = ((1/self.n) + (self.x_mean**2)/self.sxx)*self.mco_model_variance
        self.b1_variance = self.mco_model_variance/self.sxx
        
        # STANDARD RESIDUALS
        self.h = (1/self.n) + ((self.x - self.x_mean)**2) / self.sxx # leverage h_i
        self.standardized_residuals = self.residuals / np.sqrt(self.mco_model_variance * (1 - self.h))
        # STUDENTIZED RESIDUALS
        numerator = (self.n - 2) * self.mco_model_variance - (self.residuals**2) / (1 - self.h)
        sigma2_i = numerator / (self.n - 3)
        self.studentized_residuals = self.residuals / np.sqrt(sigma2_i * (1 - self.h))
        
    def plot_linear_reg(self, ax=None):
        created_ax = ax is None
        if created_ax:
            fig, ax = plt.subplots()
        
        str_self = 'Linear regression | security_x ' + self.security_x \
            + ' | security_y ' + self.security_y + '\n' \
            + 'alpha ' + str(self.alpha) \
            + ' | beta (slope) ' + str(self.beta)  + '\n' \
            + 'p-value ' + str(self.p_value) \
            + ' | null-hypothesis ' + str(self.hypothesis_null) + '\n' \
            + 'correl (r-value) ' + str(self.correlation) \
            + ' | r-squared ' + str(self.r_squared)
            
        str_title = 'Scatterplot of returns ' + '\n' + str_self
        str_compacto = self.security_x + ' vs ' + self.security_y  + ' corr: ' + f"{self.correlation:.3f}" + \
        ' | R²: ' + f"{self.r_squared:.3f}"
        
        ## plt.figure(figsize=(10,10))
        #plt.title(str_title)
        #plt.scatter(self.x, self.y)
        #plt.plot(self.x, self.predictor_linreg, color='green' )
        #plt.ylabel(self.security) 
        #plt.xlabel(self.benchmark) 
        #plt.grid()
        #plt.show()
        
        if created_ax:
            ax.set_title(str_title)
        else:
            ax.set_title(str_compacto)
        
        ax.scatter(self.x, self.y)
        ax.plot(self.x, self.predictor_linreg, color='green')
        
        ax.set_ylabel(self.security_y) 
        ax.set_xlabel(self.security_x) 
        ax.grid()
        
        if ax is None:
            plt.show()
        
        
    def model_confidence_intervals(self, significance = 0.05):
        t_crit = t.ppf(1 - significance/2, self.df)
        # Intervalo para b_0
        self.b0_interval = (self.alpha + t_crit * np.sqrt(self.b0_variance), self.alpha - t_crit * np.sqrt(self.b0_variance))
        # Intervalo para b1, diferente punto crítico, pero dado que estamos usando t student el cual es simetrico y por como se definen las zonas en python usamos el mismo pto crítico definido.
        self.b1_interval = (self.beta + t_crit * np.sqrt(self.b1_variance), self.beta - t_crit * np.sqrt(self.b1_variance))
        
    def pred_and_real_intervals(self, x_value = float, significance = 0.05):
        prediction = self.alpha + self.beta * x_value
        print('¿Cuál es la media esperada de Y (la media de Y es el valor ajustado) cuando x = ', x_value,'?: ', prediction)
        # Calculo del intervalo de la media esperada de Y, es decir del valor ajustado, de la predicción:
        t_crit = t.ppf(1 - significance/2, self.df)
        prediction_variance = self.mco_model_variance * (1/self.n + ((x_value - self.x_mean)**2) / self.sxx)
        prediction_interval = (prediction + t_crit * np.sqrt(prediction_variance), prediction - t_crit * np.sqrt(prediction_variance))
        print('Intervalo donde estará mi media esperada/predicción de y:', prediction_interval)
        # Recordemos que prediction + error de estimación = Y_x (real)
        variance_for_real_y = self.mco_model_variance * (1 + 1/self.n + ((x_value - self.x_mean)**2) / self.sxx)
        real_y_interval = ( prediction - t_crit * np.sqrt(variance_for_real_y), prediction + t_crit * np.sqrt(variance_for_real_y))
        print('Intervalo para Y_x - ¿Dónde caerá un valor real Y_x dado x =', x_value,'?:', real_y_interval, 'con', 100-significance*100, '% de confianza')
        # Y_x (real) - INCLUYE ruido real (epsilon) por eso es más ancho
        
    # LOS SIGUIENTES TEST PUEDEN SERVIR PARA PROBAR LO MISMO bajo un diferente enfoque, hacer t test para b1 tal que el valor de prueba sea 0 es equivalente a realizar simplemente la prueba F. La prueba F tienen un único propósito de ver que los x_i valgan diferente de 0, mientras que la prueba t se puede extender a probar con más valores.
        
    def estimator_tests(self, estimator = 'b0', value_test = 0, significance = 0.05):
        if estimator == 'b0':
            t_test = (self.alpha - value_test) / np.sqrt(self.b0_variance) # (1/n + x_mean**2/sxx) * model_variance 
            print('Predicted', estimator, 'by the model:', self.alpha)
            
        elif estimator == 'b1':
            t_test = (self.beta - value_test) / np.sqrt(self.b1_variance) # (1/sxx) * model_variance 
            print('Predicted', estimator, 'by the model:', self.beta)
            
        print('T test - t* de', estimator, ':', t_test)
            
        # Región de rechazo
        if abs(t_test) > t.ppf(1 - significance/2, self.df):
            print('Se rechaza H0 (', estimator,' = ',value_test ,') tq', estimator, 'es diferente de ', value_test, 'ya que', abs(t_test), '>',  t.ppf(1 - significance/2, self.df))
        else:
            print('Se acepta H0 (', estimator,' = ',value_test, ')')
        if t_test > t.ppf(1 - significance, self.df):
            print('Se rechaza H0 (', estimator,' <= ',value_test ,') tq', estimator, '>', value_test, 'ya que', t_test, '>',  t.ppf(1 - significance, self.df))
        elif t_test < t.ppf(1 - significance, self.df):
            print('Se rechaza H0 (', estimator,' >= ',value_test ,') tq', estimator, '<', value_test, 'ya que', t_test, '<',  t.ppf(1 - significance, self.df))
    
    def anova_test(self): # Prueba de que los features sí aporten al modelo y entonces, sean sus variables b1, ..., bn diferentes de 0
        # ANOVA
        self.suma_cuadrados_totales = np.sum((self.y - self.y_mean)**2) # n-1, ya que n-2+1
        self.suma_cuadrados_regresion = np.sum((self.predictor_linreg - self.y_mean)**2) # 1 grado DADO QUE SOLO HAY UN X EN EL MODELO DE REGESION
        # sc_reg_eq = self.beta * self.sxx # Demostrado
        self.suma_cuadrados_errores = np.sum((self.y - self.predictor_linreg)**2) # n-2 grado DADO QUE SE CONSIDERA A Y PREDICHO EL CUAL CONTIENE B0 Y B1 
        # sc_error_eq = sc_total - b1*sxx - dado que total = reg + error entonces error = total - reg
        
        # Suma de Cuadrados Medios... dividir entre sus grados de libertad a la suma de cuadrados 
        self.cuadrados_medios_regresion = self.suma_cuadrados_regresion/1
        self.cuadrados_medios_errores = self.suma_cuadrados_errores/(self.n-2)
        self.cuadrados_medios_totales = self.suma_cuadrados_totales/(self.n-1)
        
        # CON EL CM CALCULAMOS LA F PRUEBA
        # La prueba nos dice qué tanto explica el modelo (señal) vs qué tanto queda como ruido
        self.F_test = self.cuadrados_medios_regresion/self.cuadrados_medios_errores # = CMR/CME = CMR/var_model_mco
        print('Prueba F con valor: ', self.F_test)

        # CON EL SIMPLE VALOR DE F NO PUEDES COMPARAR MODELOS, NECESITAS R2 TAMBIEN Y TENER EN CUENTA:

        print('Hipótesis: \n H0 - todos los coeficientes (excepto intercepto) = 0 \n H1: al menos uno de los features (x_i) es diferente de 0 \n - p-value mas cercano a 0 mas ridículo H0')
        k=1 # Número de variables explicativas, menos intercepto
        df1 = k 
        df2 = self.n - k - 1  # grados de libertad del error... n-2, dinámicamente correcto
        p_value = 1 - stats.f.cdf(self.F_test, df1, df2)
        print(p_value)
    
    def linealidad(self, corregir = True):
        '''
        Si no hay linealidad... el modelo está mal especificado y la relación X - Y no es una recta
        Para solucionarlo debemos de transformar variables (log, diff), agregar términos (polinomios) o usar otro modelo
        EN EL MÚLTIPLE Y_HAT VS Residuals (e) ES MÁS INFORMATIVO.
        
        
        '''
        plt.figure(figsize=(7,5))
        
        # y_hat vs residuos
        plt.scatter(self.predictor_linreg, self.residuals, 
                    color='blue', alpha=0.6, label='Residuals vs Fitted')
        
        # x vs residuos
        plt.scatter(self.x, self.residuals, 
                    color='red', alpha=0.6, label='Residuals vs X')
        
        plt.axhline(0, color='black', linestyle='--')
        
        plt.xlabel("X / Fitted values")
        plt.ylabel("Residuals")
        plt.title("Residuals comparison")
        plt.legend()
        plt.grid()
        
        plt.show()
        
        print("LINEALIDAD:")
        print("- OK: nube aleatoria sin patrón -> varianza no constante") # DEBIDO A QUE LA VARIANZA DE LOS ERRORES e_i DEPENDEN DEL VALOR x_i OBSERVADO, haciendo de su varianza no constante.
        print("- MAL: forma curva, tendencia o estructura visible\n")
        
        
    def homocedasticidad(self, significance = 0.05, corregir = True, plot=False):
        '''
        Si hay heterocedasticidad (varianza no constante)... el “ruido” cambia con el nivel de X o ŷ, los coeficientes siguen siendo válidos pero p-values y errores estándar están mal
        Este NO rompe el modelo, rompe la inferencia.
        usar errores robustos (HC1, HC3) - sm.OLS(y, X).fit(cov_type='HC1')
        modelar volatilidad (GARCH)
        
        Si el parámetro corregir es igual a True realiza...
        
        '''
        # Construimos matriz X manualmente (intercepto + x)
        X = np.column_stack((np.ones(len(self.x)), self.x))
    
        bp_test = het_breuschpagan(self.residuals, X)
        bp_pvalue = bp_test[1]
    
        print("HOMOCEDASTICIDAD (Breusch-Pagan):")
        print(f"p-value: {bp_pvalue:.5f}")
    
        if bp_pvalue < significance:
            print("→ Se RECHAZA H0: hay heterocedasticidad (varianza NO constante)\n")
        else:
            print("→ NO se rechaza H0: varianza constante\n")
            
        # Para evaluar homocedasticidad correctamente, se usan residuos estandarizados o studentizados (más robusto)')
        
        if plot:
            # ERRORES ESTANDARIZADOS, con un fin más robusto de probar la homeosticidad
            plt.figure(figsize=(7,5))
        
            plt.scatter(self.predictor_linreg, self.studentized_residuals,
                        color='purple', alpha=0.6, label='Studentized Residuals')
        
            plt.axhline(0, color='black', linestyle='--')
        
            plt.xlabel("Fitted values")
            plt.ylabel("Studentized Residuals")
            plt.title("Homoscedasticity Check")
            plt.legend()
            plt.grid()
        
            plt.show()
        
        np.where(np.abs(self.studentized_residuals) > 2)
        
    def normalidad(self, significance = 0.05):
        '''
        Si no hay normalidad en los residuos... y además la muestra es pequeña los coeficientes siguen bien pero...
        * tests (t, F) pueden ser menos precisos
        * p values pueden estar mal calibrados
        * intervalos de confianza erroneos.
        Afecta la inferencia, haciendola inválida 
        Es un hecho que en práctica casi nunca se cumple en datos reales y no es crítico si tenemos muchos datos (CLT)
        
        '''
        shapiro_test = shapiro(self.residuals)
        shapiro_pvalue = shapiro_test.pvalue
    
        print("NORMALIDAD (Shapiro-Wilk):")
        print(f"p-value: {shapiro_pvalue:.5f}")
    
        if shapiro_pvalue < significance:
            print("→ Se RECHAZA H0: residuos NO normales\n")
        else:
            print("→ NO se rechaza H0: residuos normales\n")
            
    def independencia(self):
        '''
        Si hay autocorrelación significa que los errores están correlacionados en el tiempo
        En consecuencia la inferencia es inválida y es un modelo mal especificado dinámicamente
        Este sí es grave en series de tiempo.
        Para solucionarlo podemos agregar rezagos (lags) o usar modelos tipo ARIMA
        '''
        dw_stat = durbin_watson(self.residuals)
    
        print(" INDEPENDENCIA (Durbin-Watson):")
        print(f"DW statistic: {dw_stat:.5f}")
    
        if dw_stat < 1.5:
            print("→ Posible autocorrelación positiva\n")
        elif dw_stat > 2.5:
            print("→ Posible autocorrelación negativa\n")
        else:
            print("→ Sin evidencia de autocorrelación\n")
        
    
    
    #Interpretación general:
    #   - p-value < alpha → evidencia contra H0 (se viola el supuesto)
    #   - p-value >= alpha → no hay evidencia para rechazar H0
        
        
    # SE DETECTÓ EVIDENCIA EN CONTRA DE LA HOMOCEDASTICIDAD.
    # MÉTODO 1:
    def yeo_johnson_transform(self, use_standardize=True, plot=False):
        """
        Aplica transformación Yeo-Johnson a Y (permite valores negativos).
        
        Parámetros:
        - use_standardize: si True, devuelve datos con media 0 y varianza 1
        - plot: hist antes/después
        
        Guarda:
        - self.y_transformed
        - self.lambda_yj
        - self.yj_model (para aplicar en nuevos datos)
        
        SE RECOMIENDA NO ESTANDARIZAR al cargar la información/elegir el enfoque
        Es decir, funciona mejor con el enfoque macro.
        """
        
        y = self.y.reshape(-1, 1)
    
        pt = PowerTransformer(method='yeo-johnson', standardize=use_standardize)
        y_t = pt.fit_transform(y)
    
        self.y_transformed = y_t.flatten()
        self.lambda_yj = pt.lambdas_[0]
        self.yj_model = pt  # para usar después en predicciones nuevas
    
        print(f"Lambda Yeo-Johnson: {self.lambda_yj:.4f}")
        
        # Interpretación rápida
        if abs(self.lambda_yj - 1) < 0.1:
            print("→ No es necesaria transformación")
        elif abs(self.lambda_yj) < 0.1:
            print("→ Aproximadamente log-like")
        elif self.lambda_yj < 0:
            print("→ Compresión fuerte (colas pesadas)")
        else:
            print("→ Transformación moderada")
    
        if plot:
            plt.figure(figsize=(10,4))
    
            plt.subplot(1,2,1)
            plt.hist(self.y, bins=30)
            plt.title("Y original")
    
            plt.subplot(1,2,2)
            plt.hist(self.y_transformed, bins=30)
            plt.title("Y transformada (Yeo-Johnson)")
    
            plt.show()
    
        return self.y_transformed
    
    # MÉTODO 2
    # PARA HETEROCEDASTICIDAD EVALUO PRIMERO LA SIGUIENTE FUNCIÓN PARA DECIDIR QUE MÉTODO USAR EN LA REGRESIÓN PONDERADA... GROUPING O UNA REGRESIÓN AUXILIAR 
    def plot_squared_residuals(self):
        """
        Interpretation:
        - Linear relation -> use squared
        - Exponential -> use log
        - Nonlinear pattern(ruido raro) -> g(x) is nonlinear -> use the grouping method
        
        The target is get a random cloude
        - Random cloud → homoscedasticity
        
        Posible observations:
        - Funnel shape (cone) → heteroskedasticity
        - Increasing pattern → variance grows with level
        """
    
        e2 = self.residuals ** 2
        y_hat = self.predictor_linreg
    
        plt.figure(figsize=(7,5))
        plt.scatter(y_hat, e2, alpha=0.6)
        plt.xlabel("Fitted values (ŷ)")
        plt.ylabel("Squared residuals (e²)")
        plt.title("e² vs Fitted Values")
        plt.grid()
        plt.show()
    
    def consolidate_wls_model(self, overwrite=True):
        """
        Convierte el modelo WLS en el modelo activo de la clase.
        
        overwrite=True:
            Reemplaza alpha, beta, residuos, varianza, etc.
        
        overwrite=False:
            Mantiene ambos (OLS y WLS separados)
        """
    
        if not hasattr(self, 'alpha_wls'):
            raise ValueError("Run run_wls_pipeline() first")
    
        # =========================
        # Guardar OLS (por seguridad)
        # =========================
        self.alpha_ols = self.alpha
        self.beta_ols = self.beta
        self.residuals_ols = self.residuals
        self.predictor_ols = self.predictor_linreg
        self.mco_model_variance_ols = self.mco_model_variance
    
        # =========================
        # Calcular varianza WLS
        # =========================
        self.mco_model_variance_wls = np.sum(self.residuals_wls**2) / (self.n - 2)
    
        # =========================
        # Si quieres que WLS sea el modelo principal
        # =========================
        if overwrite:
    
            self.alpha = self.alpha_wls
            self.beta = self.beta_wls
    
            self.predictor_linreg = self.predictor_wls
            self.residuals = self.residuals_wls
    
            self.mco_model_variance = self.mco_model_variance_wls
    
            # Recalcular leverage (h) con mismo X
            self.h = (1/self.n) + ((self.x - self.x_mean)**2) / self.sxx
    
            # Residuales estandarizados
            self.standardized_residuals = self.residuals / np.sqrt(
                self.mco_model_variance * (1 - self.h)
            )
    
            # Studentizados
            numerator = (self.n - 2) * self.mco_model_variance - (self.residuals**2) / (1 - self.h)
            sigma2_i = numerator / (self.n - 3)
    
            self.studentized_residuals = self.residuals / np.sqrt(
                sigma2_i * (1 - self.h)
            )
    
            print("\n--- WLS MODEL IS NOW ACTIVE ---")
            print("Alpha:", self.alpha)
            print("Beta:", self.beta)
        
    
    def run_wls_pipeline(self, method='reg_aux', aux_type='log', use_fitted=True, groups=5):
        """
        Full pipeline:
        1. Estimate g(x)
        2. Fit WLS
        3. Re-test assumptions
        4. Plot comparison
        """
    
        print("\n--- ESTIMATING VARIANCE FUNCTION g(x) ---")
        # Notes
        # - g(x) is an approximation, not exact
        # - weights stabilize variance → better inference
        if self.residuals is None:
            raise ValueError("Run compute_linear_reg() first")
    
        # Choose regressor
        x_var = self.predictor_linreg if use_fitted else self.x
    
        e = self.residuals
    
        # =========================
        # METHOD 1: AUX REGRESSION
        # =========================
        if method == 'reg_aux':
    
            if aux_type == 'log':
                y_aux = np.log(e**2 + 1e-8)  # avoid log(0)
    
            elif aux_type == 'abs':
                y_aux = np.abs(e)
            elif aux_type == 'squared':
                y_aux = e**2
            else:
                raise ValueError("aux_type must be 'log' or 'abs'")
    
            # Regression: y_aux ~ x_var
            X = np.column_stack((np.ones(len(x_var)), x_var))
            beta_aux = np.linalg.inv(X.T @ X) @ (X.T @ y_aux)
            y_hat_aux = X @ beta_aux
    
            # Recover g(x)
            if aux_type == 'log':
                gx = np.exp(y_hat_aux)
    
            elif aux_type == 'abs':
                gx = y_hat_aux**2
            elif aux_type == 'squared':
                gx = np.maximum(y_hat_aux, 1e-8)  # evitar ceros o negativos
            
    
            self.gx = gx
            self.weights = 1 / gx
    
        # =========================
        # METHOD 2: GROUPING
        # =========================
        
        elif method == 'grouping':
    
            # sort by x
            sorted_idx = np.argsort(x_var)
            x_sorted = x_var[sorted_idx]
            e_sorted = e[sorted_idx]
    
            n = len(e)
            group_size = n // groups
    
            gx = np.zeros(n)
    
            for i in range(groups):
                start = i * group_size
                end = (i + 1) * group_size if i < groups - 1 else n
    
                idx = sorted_idx[start:end]
    
                var_group = np.var(e[idx], ddof=1)
    
                gx[idx] = var_group
    
            self.gx = gx
            self.weights = 1 / gx
    
        else:
            raise ValueError("method must be 'reg_aux' or 'grouping'")
    
        print("\n--- FITTING WLS ---")
        
        if self.weights is None:
            raise ValueError("Run weighted_least_squares() first")
    
        W = np.diag(self.weights)
        X = np.column_stack((np.ones(len(self.x)), self.x))
        y = self.y
    
        beta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)
    
        self.alpha_wls = beta[0]
        self.beta_wls = beta[1]
    
        # NUEVAS predicciones WLS
        self.predictor_wls = self.alpha_wls + self.beta_wls * self.x
        self.residuals_wls = self.y - self.predictor_wls
        
        self.consolidate_wls_model(overwrite=True) #
        
    
        print("\n--- WLS ASSUMPTIONS ---")
    
        # reemplazamos temporalmente para reutilizar función
        old_residuals = self.residuals
        old_pred = self.predictor_linreg
    
        self.residuals = self.residuals_wls
        self.predictor_linreg = self.predictor_wls
    
        self.homocedasticidad()
    
        # restaurar
        self.residuals = old_residuals
        self.predictor_linreg = old_pred
    
        # =====================
        # COMPARISON PLOT
        # =====================
        plt.figure(figsize=(7,5))
        plt.scatter(self.x, self.y, alpha=0.5)
    
        plt.plot(self.x, old_pred, label='OLS', color='red')
        plt.plot(self.x, self.predictor_wls, label='WLS', color='green')
    
        plt.legend()
        plt.title("OLS vs WLS")
        plt.grid()
        plt.show()
    
    def pred_and_real_intervals_yeo(self, x_value=float, significance=0.05, inverse=True):

        prediction_t = self.alpha + self.beta * x_value
    
        t_crit = t.ppf(1 - significance/2, self.df)
    
        # Intervalo media (transformado)
        var_mean = self.mco_model_variance * (1/self.n + ((x_value - self.x_mean)**2) / self.sxx)
    
        lower_mean_t = prediction_t - t_crit * np.sqrt(var_mean)
        upper_mean_t = prediction_t + t_crit * np.sqrt(var_mean)
    
        # Intervalo real (transformado)
        var_real = self.mco_model_variance * (1 + 1/self.n + ((x_value - self.x_mean)**2) / self.sxx)
    
        lower_real_t = prediction_t - t_crit * np.sqrt(var_real)
        upper_real_t = prediction_t + t_crit * np.sqrt(var_real)
    
        # 🔥 SI usaste Yeo-Johnson
        if inverse and hasattr(self, 'yj_model'):
    
            pred = self.yj_model.inverse_transform([[prediction_t]])[0,0]
    
            lower_mean = self.yj_model.inverse_transform([[lower_mean_t]])[0,0]
            upper_mean = self.yj_model.inverse_transform([[upper_mean_t]])[0,0]
    
            lower_real = self.yj_model.inverse_transform([[lower_real_t]])[0,0]
            upper_real = self.yj_model.inverse_transform([[upper_real_t]])[0,0]
    
        else:
            pred = prediction_t
            lower_mean, upper_mean = lower_mean_t, upper_mean_t
            lower_real, upper_real = lower_real_t, upper_real_t
    
        print("Predicción:", pred)
        print("Intervalo media:", (lower_mean, upper_mean))
        print("Intervalo real:", (lower_real, upper_real))
    
    def vasicek_beta(self, b_prior, s2_prior, verbose=True):
        """
        Estimador Bayesiano de Beta — Vasicek (1973), Ecuaciones 15 y 16.
        
        Combina la beta estimada por OLS (muestra actual) con una beta
        prior (régimen anterior u otra referencia) usando sus precisiones
        como pesos.
        
        Parámetros:
        ───────────
        b_prior   : float
            Beta prior (b' en el paper).
            Opciones prácticas:
              - Beta del régimen pre-quiebre (estimada con datos previos)
              - Beta histórica promedio del par (si tienes varios períodos)
              - 0.0 si no tienes referencia (prior no informativa centrada en 0)
        
        s2_prior  : float  
            Varianza de la prior (s'_b² en el paper).
            Opciones prácticas:
              - Varianza de la beta estimada en el período previo
              - Dispersión histórica de betas del par benchmark-security
              - Valor grande (ej: 1.0 o 10.0) si la prior es débil/difusa
        
        Retorna:
        ────────
        dict con b_vasicek, s2_posterior, peso_prior, peso_muestra y diagnóstico.
        
        NOTA: Requiere haber corrido compute_linear_reg() antes.
        """
        if self.beta is None or self.b1_variance is None:
            raise ValueError("Corre compute_linear_reg() primero.")
        
        # ── Ingredientes del paper ────────────────────────────────
        b_mco   = self.beta           # beta OLS de la muestra
        s2_mco  = self.b1_variance    # s_b² = s²/Sxx  (Ec. 5 del paper)
        # Esta es exactamente la varianza de b que Vasicek llama s_b²
        # = mco_model_variance / sxx  ← ya lo calculas en compute_linear_reg
        
        # ── Precisiones (inversos de varianzas) ──────────────────
        # Interpretación: mayor precisión = más peso en la posterior
        h_mco   = 1.0 / s2_mco    # precisión de la muestra    (h en paper)
        h_prior = 1.0 / s2_prior  # precisión de la prior      (h' en paper)
        
        # ── Ecuación 15: media posterior (beta Vasicek) ──────────
        b_vasicek = (h_prior * b_prior + h_mco * b_mco) / (h_prior + h_mco)
        
        # ── Ecuación 16: varianza posterior ──────────────────────
        s2_posterior = 1.0 / (h_prior + h_mco)
        s_posterior  = s2_posterior ** 0.5
        
        # ── Pesos relativos (para diagnóstico) ───────────────────
        peso_prior   = h_prior / (h_prior + h_mco)   # fracción que aporta la prior
        peso_muestra = h_mco   / (h_prior + h_mco)   # fracción que aportan los datos
        
        # ── Ajuste relativo a OLS puro ───────────────────────────
        ajuste_abs  = b_vasicek - b_mco   # cuánto se movió respecto a OLS
        ajuste_dir  = "← hacia prior" if (b_prior > b_mco and ajuste_abs > 0) \
                      or (b_prior < b_mco and ajuste_abs < 0) else "→ OLS domina"
        
        resultado = {
            'benchmark':      self.security_x,
            'security':       self.security_y,
            'n_obs':          self.n,
            'b_mco':          round(b_mco, 6),
            'b_prior':        round(b_prior, 6),
            'b_vasicek':      round(b_vasicek, 6),
            's2_mco':         round(s2_mco, 8),
            's2_prior':       round(s2_prior, 8),
            's2_posterior':   round(s2_posterior, 8),
            's_posterior':    round(s_posterior, 6),
            'h_mco':          round(h_mco, 4),
            'h_prior':        round(h_prior, 4),
            'peso_prior_pct': round(peso_prior * 100, 2),
            'peso_mco_pct':   round(peso_muestra * 100, 2),
            'ajuste_abs':     round(ajuste_abs, 6),
        }
        
        if verbose:
            print("\n" + "═"*65)
            print(f"  ESTIMADOR BAYESIANO DE VASICEK (1973)")
            print(f"  {self.security_x} → {self.security_y}  |  N = {self.n} obs")
            print("═"*65)
            
            print(f"\n  INPUTS")
            
            label_b    = "Beta OLS (muestra):"
            label_s2   = "Varianza OLS (s²_b):"
            label_bpr  = "Beta prior (b'):"
            label_s2pr = "Varianza prior (s'²_b):"
            label_hm   = "Precisión muestra (h):"
            label_hp   = "Precisión prior (h'):"
            label_bv   = "Beta Vasicek (b''):"
            label_sv   = "Varianza posterior:"
            label_sd   = "Desv. std posterior:"
            
            print(f"  {label_b:<30} b   = {b_mco:>10.6f}")
            print(f"  {label_s2:<30} s²  = {s2_mco:>10.8f}")
            print(f"  {label_bpr:<30} b'  = {b_prior:>10.6f}")
            print(f"  {label_s2pr:<30} s'² = {s2_prior:>10.8f}")
            
            print(f"\n  PRECISIONES  (h = 1/s²  — mayor = más peso)")
            print(f"  {label_hm:<30}     {h_mco:>10.4f}")
            print(f"  {label_hp:<30}     {h_prior:>10.4f}")
            
            print(f"\n  PESOS EN LA POSTERIOR")
            barra_m = '█' * int(peso_muestra * 40)
            barra_p = '█' * int(peso_prior   * 40)
            print(f"  Muestra : {barra_m:<40} {peso_muestra*100:5.1f}%")
            print(f"  Prior   : {barra_p:<40} {peso_prior*100:5.1f}%")
            
            print(f"\n  RESULTADO — Ecuaciones 15 y 16 de Vasicek (1973)")
            print(f"  {label_bv:<30} b'' = {b_vasicek:>10.6f}")
            print(f"  {label_sv:<30} s''²= {s2_posterior:>10.8f}")
            print(f"  {label_sd:<30} s'' = {s_posterior:>10.6f}")
            
            print(f"\n  INTERPRETACIÓN")
            
            if peso_muestra > 0.75:
                dom  = "📊 MUESTRA DOMINA — los datos tienen alta precisión"
                det  = f"     Con {self.n} obs y s²_b={s2_mco:.6f}, los datos son confiables."
                det2 = "     b'' ≈ b_OLS. La prior apenas ajusta."
            elif peso_prior > 0.75:
                dom  = "📚 PRIOR DOMINA — muestra pequeña o muy ruidosa"
                det  = f"     Con {self.n} obs y s²_b={s2_mco:.6f}, hay poca precisión."
                det2 = f"     b'' se aleja de b_OLS hacia b'={b_prior}."
            else:
                dom  = "⚖️  BALANCE — muestra y prior tienen peso similar"
                det  = "     Ninguna fuente domina. b'' es promedio ponderado."
                det2 = "     Aumentar N acercaría b'' a b_OLS."
            
            print(f"  {dom}")
            print(f"  {det}")
            print(f"  {det2}")
            
            if s2_posterior < s2_mco and s2_posterior < s2_prior:
                print(f"\n  ✅ s''² ({s2_posterior:.8f}) < s²_b ({s2_mco:.8f})")
                print(f"     La posterior es MÁS precisa que cada fuente por separado.")
                print(f"     (La información se acumula — Ec. 16 del paper)")
            
            ajuste_dir = ("← hacia prior" 
                          if (b_prior > b_mco and ajuste_abs > 0) 
                          or (b_prior < b_mco and ajuste_abs < 0) 
                          else "→ OLS domina")
            
            print(f"\n  RESUMEN NUMÉRICO")
            print(f"  b_OLS = {b_mco:.6f}  →  b_Vasicek = {b_vasicek:.6f}  "
                  f"(ajuste: {ajuste_abs:+.6f}  {ajuste_dir})")
            print("═"*65)
            
            # Comparación de varianzas
            if s2_posterior < s2_mco and s2_posterior < s2_prior:
                print(f"\n  ✅ s''² ({s2_posterior:.8f}) < s²_b ({s2_mco:.8f})")
                print(f"     La posterior es MÁS precisa que cada fuente por separado.")
                print(f"     (La información se acumula — Ec. 16 del paper)")
            
            print(f"\n  RESUMEN NUMÉRICO")
            print(f"  b_OLS = {b_mco:.6f}  →  b_Vasicek = {b_vasicek:.6f}  "
                  f"(ajuste: {ajuste_abs:+.6f}  {ajuste_dir})")
            print("═"*65)
        
        return resultado
            
'''
def boxcox_transform(self, plot = False):
    """
    Aplica transformación Box-Cox a la variable dependiente (y).
    Intenta hacer residuos más normales y varianza más consistente.
    
    Es más útil en variables macro (niveles: inflación, GDP, etc.)

    Requisitos:
    - y debe ser positiva
    
    Si falla normalidad o heterocedasticidad:
    model.boxcox_transform()
    Y volver a ajustar
    """

    if np.any(self.y <= 0):
        raise ValueError("Box-Cox requiere valores positivos en y")

    self.y_transformed, self.lambda_boxcox = boxcox(self.y)

    print(f"Lambda óptimo: {self.lambda_boxcox:.4f}")

    # Interpretación rápida
    if abs(self.lambda_boxcox - 1) < 0.1:
        print("→ No es necesaria transformación")
    elif abs(self.lambda_boxcox) < 0.1:
        print("→ Usar log(y)")
    elif self.lambda_boxcox < 0:
        print("→ Transformación inversa (1/y o similar)")
    else:
        print("→ Transformación potencia moderada")
        
    if plot:
        lambdas = np.linspace(-2, 2, 100)
        llf = [boxcox_llf(l, self.y) for l in lambdas]
        
        plt.plot(lambdas, llf)
        plt.axvline(self.lambda_boxcox, color='red', linestyle='--')
        plt.title("Box-Cox Log-Likelihood")
        plt.xlabel("Lambda")
        plt.ylabel("Log-Likelihood")
        plt.grid()
        plt.show()

    return self.y_transformed


def filtrar_por_rango(df, fecha_inicio, fecha_fin):
    """
    Año-Mes-Dia
    Esta es una segunda forma de hacer el filtrado de un dataset, obtener un subset.
    Otra forma es:
        return df[df['Date'] -condition-]
    Hay muchas más.
    """
    mask = (df['Fecha'] >= fecha_inicio) & (df['Fecha'] <= fecha_fin)
    return df.loc[mask].reset_index(drop=True) # loc[mask] selecciona solo las filas que cumplen

------


def primer_dia_de_cada_mes(df):
    """
    ¿Qué está pasando matemáticamente?
    to_period('M') agrupa por mes-año
    groupby(...).first() toma la primera fecha cronológica del mes
    No importa si el primer dato no es exactamente día 1 (festivos, fines de semana, etc.)
    """
    df = df.copy()
    df['YearMonth'] = df['Fecha'].dt.to_period('M')
    df_mes = df.groupby('YearMonth').first()
    df_mes = df_mes.reset_index(drop=True)
    df_mes['Var_x'] = df_mes['Tasa_x'] / df_mes['Tasa_x'].shift(1) - 1
    df_mes['Var_y'] = df_mes['Tasa_y'] / df_mes['Tasa_y'].shift(1) - 1
    df_mes['Dif'] = df_mes['Tasa_y'] - df_mes['Tasa_x']
    return df_mes


# info_mensual.loc[info_mensual['Fecha'] == '2021-01-02']
# Me retorna un df filtrado bajo la condicion previa de Mask

-----

def promedio_mensual(df):
    """
    Calcula el promedio mensual (mes-año) de las tasas y variaciones.
    Retorna un DataFrame con frecuencia mensual.
    """
    df = df.copy()
    
    # Crear identificador Año-Mes
    df['YearMonth'] = df['Fecha'].dt.to_period('M')
    
    # Agrupar por mes y calcular promedios
    df_mensual = (
        df
        .groupby('YearMonth')
        .agg({
            'Tasa_x': 'mean',
            'Tasa_y': 'mean',
            'Var_x': 'mean',
            'Var_y': 'mean'
        })
        .reset_index()
    )
    
    # Convertir YearMonth a fecha (primer día del mes)
    df_mensual['Fecha'] = df_mensual['YearMonth'].dt.to_timestamp()
    
    # Diferencial promedio mensual (opcional, pero útil)
    df_mensual['Dif'] = df_mensual['Tasa_y'] - df_mensual['Tasa_x']
    
    # Orden final
    df_mensual = df_mensual.sort_values('Fecha').reset_index(drop=True)
    df_mensual.drop('Fecha', axis=1, inplace=True)
    
    return df_mensual

-----

def plot_timeseries(df):
    plt.figure(figsize=(12,5))
    plt.title('Timeseries of Rates')
    plt.xlabel( 'Time')
    plt.ylabel( 'Rates')
    ax = plt.gca()
    ax1 = df.plot(kind='line', x='Fecha', y='Tasa_x', ax=ax, grid=True, color='blue', label=info2)
    ax2 = df.plot(kind='line', x='Fecha', y='Tasa_y' , color='red', secondary_y=False, ax=ax, grid=True, label=info1)
    ax3 = df.plot(kind='line', x='Fecha', y='Dif' , color='green', secondary_y=False, ax=ax, grid=True, label='Dif')
    ax1.legend(loc=2)
    ax2.legend(loc=1)
    plt.show()
    
'''

def get_data(x = list, y = list, from_date = 'aaaa-mm-dd', to_date= 'aaaa-mm-dd'):
    results = []
    models_to_plot = []
    for sec in y:
        for bench in x:
            if sec == bench:
                continue
            
            m = model(bench, sec, 6)
            m.synchronise_timeseries(from_date, to_date)
            m.compute_linear_reg()
            
            results.append({
                "security": sec,
                "benchmark": bench,
                "beta": m.beta,
                "correlation": m.correlation,
                "r2": m.r_squared
            })
            if abs(m.correlation) >= 0.3:
                models_to_plot.append(m)
    print('*'*10, 'Resultados desde', m.timeseries['Fecha'].min().strftime('%Y-%m-%d'), 'hasta', m.timeseries['Fecha'].max().strftime('%Y-%m-%d'), '*'*10)
    n = len(models_to_plot)

    if n == 0:
        print("No hay modelos que cumplan el criterio para graficar")
    else:
        cols = 2
        rows = math.ceil(n / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
        axes = np.atleast_1d(axes).flatten()

        for i, m in enumerate(models_to_plot):
            m.plot_linear_reg(ax=axes[i])

        # eliminar ejes sobrantes
        for ax in axes[n:]:
            fig.delaxes(ax)

        plt.tight_layout(pad=2.0)
        plt.show()

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='correlation', ascending=False)
    
    return df_results


def buscar_mejor_r2(x, y, start_date, end_date, min_obs=60, min_corr = 0.3, freq = 'MS', order = 'score'):
    ''' 
    freq='QS' - Quarterly Start - 2018-01-01, 2018-04-01, 2018-07-01, 2018-10-01...

    freq='Y' - Fin de año - 2018-12-31, 2019-12-31, 2020-12-31...
    freq='A-DEC' - Annual, cierre en diciembre
    freq='AS-JAN' - Annual, inicio en enero
    freq='A-JUN' - Fin de año fiscal distinto

    freq='W-FRI' - Para cierres de viernes (muy común en mercados)
    Para weekly el ajuste de min_obs podría ser min_obs = 12 equiv. aprox. a 3 meses
    o min_obs = 26 equiv. aprox.a 6 meses

    freq='D' - Para daily - Cuidado con las combinaciones posibles que genera, lo podría hacer inviable.
    Para daily el ajuste de min_obs podría ser min_obs = 60 equiv. aprox. a 3 meses
    o min_obs = 90 equiv. aprox.a 6 meses
    '''
    
    fechas = pd.date_range(start=start_date, end=end_date, freq=freq)  # Monthly Start, esto crea bloques de fechas mensuales... 2018-01-01, 2018-02-01, 2018-03-01, ... etc
    resultados = []

    for sec in y:
        for bench in x:
            if sec == bench:
                continue
            
            for i in range(len(fechas)):
                for j in range(i+1, len(fechas)): # Luego hace combinaciones de todas las ventanas posibles mensuales. 
                    # De un año son 66! n(n-1)/2... enero a febero,..., enero a diciembre y luego de febero a marzo,..., febrero a diciemnbre y asi hasta noviembre a diciembre
                    
                    from_d = fechas[i]
                    to_d = fechas[j]
                    
                    m = model(bench, sec, 6, model_type='statistics')
                    m.synchronise_timeseries(from_d, to_d)
                    
                    # filtro de tamaño mínimo
                    if len(m.timeseries) < min_obs: # Si los días hábiles son menores a x dia desechalos para evitar ventanas “perfectas” pero espurias
                        continue
                    
                    m.compute_linear_reg()
                    
                    resultados.append({
                        "security": sec,
                        "benchmark": bench,
                        "r2": m.r_squared,
                        "correlation": m.correlation,
                        "beta": m.beta,
                        "from": m.timeseries['Fecha'].min().strftime('%Y-%m-%d'),
                        "to": m.timeseries['Fecha'].max().strftime('%Y-%m-%d'),
                        "n_obs": len(m.timeseries)
                    })

    df = pd.DataFrame(resultados)

    # filtro de calidad
    df = df[
        (df['n_obs'] >= min_obs) & 
        (abs(df['correlation']) > min_corr)]
    
    # score robusto (penaliza ventanas pequeñas)
    df['score'] = df['r2'] * np.log(df['n_obs'])
    
    # ordenar por mejor ajuste
    # df = df.sort_values(by='r2', ascending=False)
    df = df.sort_values(by=order, ascending=False).reset_index(drop=True)
    
    return df


def analizar_securities(df, metric='r2'):
    """
    Analiza el benchmark más frecuente por security con métricas de representatividad.
    Si hay empate en frecuencia, desempata por la métrica elegida (r2 o correlation).
    """
    # 1. Identificar el security principal
    top_security = df['security'].value_counts().idxmax()
    
    # 2. Calcular el total de filas por cada security (para el denominador)
    totales_por_sec = df['security'].value_counts().to_dict()
    
    # 3. Preparar desempate dinámico
    df_temp = df.copy()
    if metric == 'correlation':
        df_temp['metric_abs'] = df_temp[metric].abs()
    else:
        df_temp['metric_abs'] = df_temp[metric]

    # 4. Agrupar por par security-benchmark
    agrupado = df_temp.groupby(['security', 'benchmark']).agg(
        repeticiones=('benchmark', 'count'),
        max_metrica=('metric_abs', 'max')
    ).reset_index()

    # 5. Ordenar y seleccionar el benchmark ganador
    resumen_base = (agrupado.sort_values(['security', 'repeticiones', 'max_metrica'], 
                                        ascending=[True, False, False])
                    .groupby('security').head(1)).copy()

    # TOTALES Y PORCENTAJES 
    resumen_base['total_obs'] = resumen_base['security'].map(totales_por_sec)
    resumen_base['pct_presencia'] = (resumen_base['repeticiones'] / resumen_base['total_obs']) * 100

    # 6. Función para extraer rangos de fecha y r2
    def get_stats(row):
        mask = (df['security'] == row['security']) & (df['benchmark'] == row['benchmark'])
        sub_df = df[mask]
        idx_min = sub_df['r2'].idxmin()
        idx_max = sub_df['r2'].idxmax()
        
        return pd.Series({
            'min_r2': sub_df.loc[idx_min, 'r2'],
            'corr_min': sub_df.loc[idx_min, 'correlation'],
            'score_min': sub_df.loc[idx_min, 'score'],
            'from_date_min': sub_df.loc[idx_min, 'from'],
            'to_date_min': sub_df.loc[idx_min, 'to'],
            'max_r2': sub_df.loc[idx_max, 'r2'],
            'corr_max': sub_df.loc[idx_max, 'correlation'],
            'score_max': sub_df.loc[idx_max, 'score'],
            'from_date_max': sub_df.loc[idx_max, 'from'],
            'to_date_max': sub_df.loc[idx_max, 'to']
        })

    detalles = resumen_base.apply(get_stats, axis=1)
    
    # Reordenamos columnas para que lo nuevo aparezca al principio junto a repeticiones
    cols_base = ['security', 'benchmark', 'repeticiones', 'total_obs', 'pct_presencia']
    resumen_final = pd.concat([resumen_base[cols_base], detalles], axis=1)
    
    principal = resumen_final[resumen_final['security'] == top_security].reset_index(drop=True)
    complemento = resumen_final[resumen_final['security'] != top_security].reset_index(drop=True)
    
    
    return principal, complemento

# "   Presencia: {row['repeticiones']}/{row['total_obs']} ({row['pct_presencia']:.2f}%)" 
def imprimir_detalles_analisis(df_resultados, titulo="ANÁLISIS DETALLADO"):
    """
    Imprime de forma estructurada los resultados (sirve para Principal o Complemento).
    Muestra R2, Periodos y los Scores asociados.
    """
    if df_resultados.empty:
        print(f"No hay datos para mostrar en: {titulo}")
        return

    print("\n" + "="*70)
    print(f"{titulo:^70}")
    print("="*70)

    for i, row in df_resultados.iterrows():
        print(f"\n📌 SECURITY: {row['security']} | Benchmark: {row['benchmark']}")
        print(f"   Presencia: {row['repeticiones']} de {row['total_obs']} obs. ({row['pct_presencia']:.2f}%)")
        
        print(f"   " + "-"*65)
        # Bloque Min con R (correlación)
        print(f"   📉 Min R2: {row['min_r2']:.4f} [r: {row['corr_min']:.4f}] (Score: {row['score_min']:.4f})")
        print(f"      Periodo: {row['from_date_min']} a {row['to_date_min']}")
        
        # Bloque Max con R (correlación)
        print(f"   📈 Max R2: {row['max_r2']:.4f} [r: {row['corr_max']:.4f}] (Score: {row['score_max']:.4f})")
        print(f"      Periodo: {row['from_date_max']} a {row['to_date_max']}")
    
    print("\n" + "="*80)
    
def imprimir_resumen_frecuencias(df):
    # Obtener las frecuencias (conteo de cada security)
    counts = df['security'].value_counts()

    print('Conteo de securities:', counts, '\n')

    # Identificar el más frecuente y su valor
    sec_most_frequent = counts.idxmax()
    frequency = counts.max()

    # Calcular el porcentaje vs el total
    total_rows = len(df)
    percentage = (frequency / total_rows) * 100

    print(f"El security (y) más frecuente es: {sec_most_frequent}")
    print(f"Aparece {frequency} veces de un total de {total_rows} ({percentage:.2f}%)")

    # Filtrar y encontrar su benchmark más común
    top_benchmark = df[df['security'] == sec_most_frequent]['benchmark'].value_counts().idxmax()
    conteo_bm = df[df['security'] == sec_most_frequent]['benchmark'].value_counts().max()

    print(f"Del security '{sec_most_frequent}', su benchmark (x) más frecuente es '{top_benchmark}' ({conteo_bm} veces)")

    # Filtramos por las dos variables líderes para tener el bloque específico
    df_most_freq = df[(df['security'] == sec_most_frequent) & (df['benchmark'] == top_benchmark)]


    # Localizar el mejor registro por Score
    row_best_score = df_most_freq.loc[df_most_freq['score'].idxmax()]

    print(f"\nMejor registro por Score:")
    print(f"Sec: {row_best_score['security']} | Bench: {row_best_score['benchmark']} | Score: {row_best_score['score']:.4f} | R2: {row_best_score['r2']:.4f} | Periodo: {row_best_score['from']} a {row_best_score['to']}")


def analizar_temporalidad_lider(df):
    # 1. Identificar el bloque principal (Líderes)
    top_sec = df['security'].value_counts().idxmax()
    top_bench = df[df['security'] == top_sec]['benchmark'].value_counts().idxmax()
    
    # Filtrar el bloque
    bloque_lider = df[(df['security'] == top_sec) & (df['benchmark'] == top_bench)].copy()
    
    # 2. Convertir a datetime y calcular duración
    bloque_lider['from'] = pd.to_datetime(bloque_lider['from'])
    bloque_lider['to'] = pd.to_datetime(bloque_lider['to'])
    bloque_lider['duracion_dias'] = (bloque_lider['to'] - bloque_lider['from']).dt.days
    
    # 3. Score: Calidad (R2) x Persistencia (Duración)
    # Esto penaliza R2s altos en periodos muy cortos
    bloque_lider['score_temporal'] = bloque_lider['r2'] * bloque_lider['duracion_dias']
    
    # 4. Encontrar el mejor registro bajo este criterio
    mejor_registro = bloque_lider.loc[bloque_lider['score_temporal'].idxmax()]
    
    # 5. Análisis de meses más repetidos (frecuencia temporal)
    # Expandimos los rangos para ver qué meses "viven" más en este bloque
    meses_lista = []
    for _, row in bloque_lider.iterrows():
        rango = pd.date_range(start=row['from'], end=row['to'], freq='MS')
        meses_lista.extend(rango)
    
    mes_mas_frecuente = pd.Series(meses_lista).dt.month_name().value_counts().idxmax()

    print(f"--- ANÁLISIS ESTRUCTURAL PARA {top_sec} vs {top_bench} ---")
    print(f"Mejor periodo detectado (R2 ponderado): {mejor_registro['from'].date()} a {mejor_registro['to'].date()}")
    print(f"R2 en este periodo: {mejor_registro['r2']:.4f} ({mejor_registro['duracion_dias']} días)")
    print(f"Mes con mayor recurrencia en el análisis: {mes_mas_frecuente}")
    
    return mejor_registro.to_frame().T

def analizar_todo_el_df(df, metric='r2'):
    """
    Analiza TODOS los benchmarks de cada security, calculando extremos, 
    scores y representatividad para cada uno.
    """
    # 1. Identificar el security principal para mantener la referencia
    top_security_name = df['security'].value_counts().idxmax()
    totales_por_sec = df['security'].value_counts().to_dict()
    
    # 2. Preparar métrica de ordenamiento
    df_temp = df.copy()
    if metric == 'correlation':
        df_temp['metric_abs'] = df_temp[metric].abs()
    else:
        df_temp['metric_abs'] = df_temp[metric]

    # 3. Agrupar por par security-benchmark (TODOS)
    agrupado = df_temp.groupby(['security', 'benchmark']).agg(
        repeticiones=('benchmark', 'count'),
        max_metrica=('metric_abs', 'max')
    ).reset_index()

    # 4. Ordenar para que el más frecuente de cada sec aparezca primero
    resumen_base = agrupado.sort_values(['security', 'repeticiones', 'max_metrica'], 
                                       ascending=[True, False, False])

    resumen_base['total_obs'] = resumen_base['security'].map(totales_por_sec)
    resumen_base['pct_presencia'] = (resumen_base['repeticiones'] / resumen_base['total_obs']) * 100

    # 5. Extraer estadísticas de min/max R2 para cada par
    def get_stats(row):
        mask = (df['security'] == row['security']) & (df['benchmark'] == row['benchmark'])
        sub_df = df[mask]
        idx_min = sub_df['r2'].idxmin()
        idx_max = sub_df['r2'].idxmax()
        
        return pd.Series({
            'min_r2': sub_df.loc[idx_min, 'r2'],
            'corr_min': sub_df.loc[idx_min, 'correlation'],
            'score_min': sub_df.loc[idx_min, 'score'],
            'from_date_min': sub_df.loc[idx_min, 'from'],
            'to_date_min': sub_df.loc[idx_min, 'to'],
            'max_r2': sub_df.loc[idx_max, 'r2'],
            'corr_max': sub_df.loc[idx_max, 'correlation'],
            'score_max': sub_df.loc[idx_max, 'score'],
            'from_date_max': sub_df.loc[idx_max, 'from'],
            'to_date_max': sub_df.loc[idx_max, 'to']
        })

    detalles = resumen_base.apply(get_stats, axis=1)
    
    cols_base = ['security', 'benchmark', 'repeticiones', 'total_obs', 'pct_presencia']
    resumen_final = pd.concat([resumen_base[cols_base], detalles], axis=1)
    
    # Separar Principal (todas sus filas) y Complemento (todas sus filas)
    df_principal = resumen_final[resumen_final['security'] == top_security_name].reset_index(drop=True)
    df_complemento = resumen_final[resumen_final['security'] != top_security_name].reset_index(drop=True)
    
    return df_principal, df_complemento

def imprimir_desglose_total(df_resultados, titulo="DESGLOSE COMPLETO"):
    if df_resultados.empty:
        print(f"No hay datos para: {titulo}")
        return

    print("\n" + "█"*85)
    print(f"{titulo:^85}")
    print("█"*85)

    # Agrupamos por security para imprimir en bloques
    securities = df_resultados['security'].unique()

    for sec in securities:
        df_sec = df_resultados[df_resultados['security'] == sec]
        total_obs = df_sec['total_obs'].iloc[0]
        
        print(f"\n💎 SECURITY: {sec} (Total obs: {total_obs})")
        print(f"   " + "═"*75)

        for _, row in df_sec.iterrows():
            print(f"   🔸 Benchmark: {row['benchmark']}")
            print(f"      Frecuencia: {row['repeticiones']} [{row['pct_presencia']:.2f}% del security]")
            
            # Bloque de datos técnicos
            print(f"      📉 Min R2: {row['min_r2']:.4f} [r: {row['corr_min']:.4f}] (Score: {row['score_min']:.4f})")
            print(f"         Periodo: {row['from_date_min']} a {row['to_date_min']}")
            
            print(f"      📈 Max R2: {row['max_r2']:.4f} [r: {row['corr_max']:.4f}] (Score: {row['score_max']:.4f})")
            print(f"         Periodo: {row['from_date_max']} a {row['to_date_max']}")
            print(f"      " + "·"*60)

    print("\n" + "█"*85)


def imprimir_resumen_frecuencias(df):
    counts = df['security'].value_counts()
    sec_most_frequent = counts.idxmax()
    frequency = counts.max()
    percentage = (frequency / len(df)) * 100

    print(f"\n--- RESUMEN GENERAL DE ACTIVOS ---")
    print(f"El activo más frecuente es: {sec_most_frequent} ({percentage:.2f}% de la base)")

    # Encontrar el benchmark líder absoluto para el reporte rápido
    top_benchmark = df[df['security'] == sec_most_frequent]['benchmark'].value_counts().idxmax()
    
    # Localizar el mejor registro por Score del security líder
    df_lider = df[(df['security'] == sec_most_frequent) & (df['benchmark'] == top_benchmark)]
    row_best = df_lider.loc[df_lider['score'].idxmax()]

    print(f"Mejor ajuste absoluto para el líder:")
    print(f" > Sec: {row_best['security']} | Bench: {row_best['benchmark']}")
    print(f" > Score: {row_best['score']:.4f} | R2: {row_best['r2']:.4f}")
    print(f" > Periodo: {row_best['from']} a {row_best['to']}\n")

def analizar_temporalidad_todos_los_pares(df):
    """
    Analiza cada combinación de Security-Benchmark y retorna la fila con 
    la mejor relación entre calidad (R2) y persistencia (Duración).
    """
    # 1. Copia y preparación de fechas
    df_temp = df.copy()
    df_temp['from'] = pd.to_datetime(df_temp['from'])
    df_temp['to'] = pd.to_datetime(df_temp['to'])
    
    # 2. Calcular duración y Score Temporal
    df_temp['duracion_dias'] = (df_temp['to'] - df_temp['from']).dt.days
    df_temp['score_temporal'] = df_temp['r2'] * df_temp['duracion_dias']
    
    # 3. Agrupar por Security y Benchmark, y tomar el ID del score máximo de cada grupo
    # idxmax() nos da el índice del mejor registro para cada par único
    indices_mejores = df_temp.groupby(['security', 'benchmark'])['score_temporal'].idxmax()
    
    # 4. Filtrar el DataFrame original usando esos índices
    resultado = df_temp.loc[indices_mejores].copy()
    
    # 5. Limpieza y ordenamiento (Opcional: del más persistente al menos)
    resultado = resultado.sort_values(by=['security', 'score_temporal'], ascending=[True, False]).reset_index(drop=True)
    
    # Reorganizar columnas para que sea fácil de leer
    cols = ['security', 'benchmark', 'r2', 'duracion_dias', 'score_temporal', 'from', 'to']
    return resultado[cols]

# --- Función para imprimir este análisis de forma legible ---
def imprimir_mejores_periodos(df_res):
    print("\n" + "◈"*80)
    print(f"{'MEJORES VENTANAS TEMPORALES (R2 x DURACIÓN)':^80}")
    print("◈"*80)
    
    for sec in df_res['security'].unique():
        print(f"\n📊 SECURITY: {sec}")
        sub = df_res[df_res['security'] == sec]
        for _, row in sub.iterrows():
            print(f"   ▫ Bench: {row['benchmark']:<15} | R2: {row['r2']:.4f} | Días: {row['duracion_dias']:>3} | Periodo: {row['from'].date()} ➜ {row['to'].date()}")
    
    print("\n" + "◈"*80)
    
    
    
# ============================================================
# MÓDULO: KAN-ZHANG DIAGNOSTICS
# Implementa los 4 diagnósticos del paper en tu framework
# ============================================================


# ─────────────────────────────────────────────────────────────
# DIAGNÓSTICO 0 (previo): Covarianza entre factores
# Detecta multicolinealidad entre benchmarks propuestos
# ─────────────────────────────────────────────────────────────
def diagnostico_covarianza_factores(benchmarks_x, from_date, to_date, umbral_corr=0.7):
    """
    Antes de correr cualquier regresión, verifica si los factores
    propuestos están correlacionados entre sí.
    
    Si corr(f_i, f_j) > umbral → multicolinealidad → los coeficientes
    individuales serán inestables en un modelo multifactorial.
    
    Esto NO es exactamente la Prop. 2 de Kan-Zhang (que habla de V,
    la matriz de cov de retornos), pero es el diagnóstico previo
    equivalente para tu setup.
    """
    import itertools
    resultados = []
    
    for f1, f2 in itertools.combinations(benchmarks_x, 2):
        try:
            # Sincronizar los dos factores
            ts = synchronise_timeseries(f1, f2, 
                                          #target_column(f1), 
                                          #target_column(f2),
                                          from_date=from_date, 
                                          to_date=to_date,
                                          model_type='macro')
            if len(ts) < 10:
                continue
            
            corr, p_val = stats.pearsonr(ts[f1].values, ts[f2].values)
            
            alerta = '⚠️  MULTICOLINEALIDAD' if abs(corr) > umbral_corr else '✅ OK'
            
            resultados.append({
                'factor_1': f1,
                'factor_2': f2,
                'correlacion': round(corr, 4),
                'p_value': round(p_val, 4),
                'n_obs': len(ts),
                'alerta': alerta
            })
        except Exception as e:
            print(f"Error en par {f1}-{f2}: {e}")
    
    df_res = pd.DataFrame(resultados).sort_values('correlacion', 
                                                    key=abs, 
                                                    ascending=False)
    print("\n" + "▓"*70)
    print(f"{'DIAGNÓSTICO 0: CORRELACIÓN ENTRE FACTORES':^70}")
    print(f"{'Periodo: ' + str(from_date) + ' a ' + str(to_date):^70}")
    print("▓"*70)
    for _, row in df_res.iterrows():
        print(f"  {row['factor_1']:<15} vs {row['factor_2']:<15} | "
              f"r={row['correlacion']:>7.4f} | p={row['p_value']:.4f} | "
              f"N={row['n_obs']:>3} | {row['alerta']}")
    print("▓"*70)
    
    return df_res


# ─────────────────────────────────────────────────────────────
# DIAGNÓSTICO 1 (Kan-Zhang): ¿Las betas son ≠ 0 en el primer paso?
# Si no → factor potencialmente inútil
# ─────────────────────────────────────────────────────────────
def diagnostico_beta_primer_paso(bench, sec, from_date, to_date, 
                                  significance=0.05, min_obs=10):
    """
    Kan & Zhang Diagnóstico 1:
    Prueba H0: beta = 0 en la regresión de series de tiempo.
    
    Si NO se rechaza H0 → el factor podría ser inútil para ese activo
    en ese período → el R2 del segundo paso es potencialmente espurio.
    
    Retorna dict con resultado del test.
    """
    try:
        m = model(bench, sec, 6, model_type='macro')
        m.synchronise_timeseries(from_date, to_date)
        
        if len(m.timeseries) < min_obs:
            return {'es_util': None, 'motivo': 'Muestra insuficiente', 
                    'n_obs': len(m.timeseries)}
        
        m.compute_linear_reg()
        
        # t-test: H0: beta = 0
        # t* = beta / SE(beta), donde SE = sqrt(var_b1)
        t_stat = m.beta / np.sqrt(m.b1_variance) if m.b1_variance > 0 else 0
        p_valor = 2 * (1 - stats.t.cdf(abs(t_stat), df=m.df))
        
        es_util = p_valor < significance
        
        return {
            'benchmark': bench,
            'security': sec,
            'beta': m.beta,
            't_stat': round(t_stat, 4),
            'p_value': round(p_valor, 4),
            'r2': m.r_squared,
            'n_obs': m.n,
            'es_util': es_util,
            'veredicto': '✅ FACTOR ÚTIL' if es_util else '⚠️  FACTOR POTENCIALMENTE INÚTIL'
        }
    except Exception as e:
        return {'es_util': None, 'motivo': str(e)}


# ─────────────────────────────────────────────────────────────
# DIAGNÓSTICO 1 masivo: para todas las combinaciones de una ventana
# ─────────────────────────────────────────────────────────────
def diagnostico_factores_utiles(benchmarks_x, securities_y, 
                                 from_date, to_date, significance=0.05):
    """
    Aplica el Diagnóstico 1 de Kan-Zhang a todas las combinaciones
    factor-activo en una ventana temporal.
    
    Úsalo ANTES de buscar_mejor_r2 para saber qué factores tienen
    poder explicativo genuino en ese período.
    """
    resultados = []
    
    for sec in securities_y:
        for bench in benchmarks_x:
            if sec == bench:
                continue
            res = diagnostico_beta_primer_paso(bench, sec, from_date, to_date, 
                                               significance=significance)
            if res.get('es_util') is not None:
                resultados.append(res)
    
    df = pd.DataFrame(resultados).sort_values('p_value')
    
    print("\n" + "═"*75)
    print(f"{'DIAGNÓSTICO 1 (KAN-ZHANG): ¿FACTORES ÚTILES?':^75}")
    print(f"{'Período: ' + str(from_date) + ' → ' + str(to_date):^75}")
    print("═"*75)
    print(f"{'Benchmark':<15} {'Security':<10} {'Beta':>8} "
          f"{'t-stat':>8} {'p-value':>8} {'R2':>6} {'N':>4}  Veredicto")
    print("-"*75)
    for _, row in df.iterrows():
        print(f"{row['benchmark']:<15} {row['security']:<10} "
              f"{row['beta']:>8.4f} {row['t_stat']:>8.4f} "
              f"{row['p_value']:>8.4f} {row['r2']:>6.4f} "
              f"{row['n_obs']:>4}  {row['veredicto']}")
    print("═"*75)
    
    factores_utiles = df[df['es_util'] == True]['benchmark'].unique().tolist()
    factores_inutiles = df[df['es_util'] == False]['benchmark'].unique().tolist()
    
    print(f"\n✅ Factores con poder explicativo genuino: {factores_utiles}")
    print(f"⚠️  Factores potencialmente inútiles:       {factores_inutiles}")
    
    return df, factores_utiles, factores_inutiles


# NOTA RESPECTO A N = 2
def nota_n_pequeno(N, r2_ols, r2_gls=None):
    """
    Advertencia formal cuando N (número de activos) es pequeño.
    
    Kan-Zhang Proposición 3: R2_GLS ~ Beta(1/2, (N-2)/2)
    Con N=2: Beta(0.5, 0) → degenerada, no aplicable
    Con N=3: Beta(0.5, 0.5) → media = 0.5 (!!)
    Con N=5: Beta(0.5, 1.5) → media = 0.25
    Con N=10: Beta(0.5, 4) → media = 0.11
    
    Bajo N pequeño, cualquier R2 alto es sospechoso por construcción.
    """
    from scipy.stats import beta as beta_dist
    
    print(f"\n⚠️  ADVERTENCIA N PEQUEÑO (N={N} activos)")
    
    if N <= 2:
        print("   Con N=2 la prueba Beta de Kan-Zhang no aplica.")
        print("   El segundo paso cross-seccional está exactamente identificado.")
        print("   Cualquier R2 del segundo paso = 1 por construcción.")
        print("   → Usa solo el primer paso (series de tiempo) para inferencia.")
        return
    
    a_param = 0.5
    b_param = (N - 2) / 2
    
    media_beta   = a_param / (a_param + b_param)
    p90_beta     = beta_dist.ppf(0.90, a_param, b_param)
    
    print(f"   Bajo factor inútil: R2_GLS ~ Beta({a_param}, {b_param:.1f})")
    print(f"   Media esperada si factor inútil: {media_beta:.3f}")
    print(f"   Percentil 90 si factor inútil:  {p90_beta:.3f}")
    
    if r2_gls is not None:
        p_valor = 1 - beta_dist.cdf(r2_gls, a_param, b_param)
        print(f"   Tu R2_GLS observado: {r2_gls:.4f}")
        print(f"   p-valor (prob de obtener R2 ≥ {r2_gls:.4f} si factor inútil): "
              f"{p_valor:.4f}")
        if p_valor < 0.05:
            print("   ✅ R2_GLS significativo — factor probablemente genuino")
        else:
            print("   ⚠️  R2_GLS no significativo — factor podría ser inútil")
    
    if r2_ols is not None and r2_gls is not None:
        if r2_ols > 2 * r2_gls:
            print(f"   🚨 R2_OLS ({r2_ols:.4f}) >> R2_GLS ({r2_gls:.4f})")
            print("      Inflación de R2_OLS confirmada. No uses R2_OLS para inferencia.")


# ─────────────────────────────────────────────────────────────
# DIAGNÓSTICO 2 (Kan-Zhang): R2_GLS vs R2_OLS
# R2_GLS ~ Beta(1/2, (N-2)/2) bajo factor inútil
# ─────────────────────────────────────────────────────────────
def diagnostico_r2_gls(bench, sec, from_date, to_date, significance=0.05):
    """
    Kan & Zhang Diagnóstico 2:
    
    Calcula R2_OLS y R2_GLS y compara con la distribución teórica
    bajo factor inútil: R2_GLS ~ Beta(1/2, (N-2)/2).
    
    En tu caso con N=2 activos la Beta está degenerada.
    Para N activos de divisas (DXY, USDMXN, EURUSD, ...) funciona mejor.
    
    Con N pequeño: úsalo para comparar R2_OLS vs R2_GLS.
    Si R2_OLS >> R2_GLS → sospecha de inflación espuria.
    """
    try:
        m = model(bench, sec, 6, model_type='macro')
        m.synchronise_timeseries(from_date, to_date)
        
        if len(m.timeseries) < 5:
            return None
        
        m.compute_linear_reg()
        r2_ols = m.r_squared
        n = m.n
        
        # WLS como aproximación de GLS
        # (GLS exacto requiere V conocida)
        m.run_wls_pipeline(method='reg_aux', aux_type='log', use_fitted=True)
        
        # R2 del modelo WLS
        ss_res_wls = np.sum(m.residuals_wls**2) if hasattr(m, 'residuals_wls') else None
        ss_tot = np.sum((m.y - m.y_mean)**2)
        r2_wls = 1 - ss_res_wls/ss_tot if ss_res_wls is not None else None
        
        ratio = r2_ols / r2_wls if (r2_wls and r2_wls > 0) else None
        
        alerta = ''
        if ratio and ratio > 2:
            alerta = '⚠️  R2_OLS inflado (posible factor inútil o heterocedasticidad)'
        elif ratio:
            alerta = '✅ R2_OLS y R2_WLS consistentes'
        
        return {
            'benchmark': bench,
            'security': sec,
            'r2_ols': round(r2_ols, 4),
            'r2_wls': round(r2_wls, 4) if r2_wls else None,
            'ratio_ols_wls': round(ratio, 3) if ratio else None,
            'n_obs': n,
            'alerta': alerta
        }
    except Exception as e:
        return {'error': str(e)}
    
# MAYOR FORMALIDAD SE CREA UNA VERSION 2 DE LA FUNCION ANTERIOR USANDO GLS AR1
def run_gls_ar1(self, significance=0.05):
    """
    GLS para series de tiempo con errores AR(1).
    
    Corrige autocorrelación en los errores mediante la
    transformación de Cochrane-Orcutt:
    
    1. Estima rho (autocorrelación de residuos) con OLS
    2. Transforma: y* = y_t - rho*y_{t-1}
                   x* = x_t - rho*x_{t-1}
    3. Corre OLS sobre variables transformadas
    
    Más apropiado que WLS cuando Durbin-Watson < 1.5 o > 2.5
    """
    if self.residuals is None:
        raise ValueError("Corre compute_linear_reg() primero")
    
    # Paso 1: estimar rho
    e = self.residuals
    rho = np.sum(e[1:] * e[:-1]) / np.sum(e[:-1]**2)
    
    print(f"ρ estimado (AR1): {rho:.4f}")
    if abs(rho) < 0.1:
        print("→ Autocorrelación baja. GLS-AR1 puede no ser necesario.")
    
    # Paso 2: transformación Cochrane-Orcutt
    y = self.y
    x = self.x
    
    y_star = y[1:] - rho * y[:-1]
    x_star = x[1:] - rho * x[:-1]
    
    # Paso 3: OLS sobre transformadas
    slope, intercept, r, p, se = stats.linregress(x_star, y_star)
    
    self.beta_gls  = round(slope, self.decimals)
    self.alpha_gls = round(intercept, self.decimals)
    self.r2_gls    = round(r**2, self.decimals)
    self.rho_ar1   = round(rho, 4)
    
    pred_gls   = intercept + slope * x_star
    resid_gls  = y_star - pred_gls
    
    # Test DW sobre residuos GLS
    from statsmodels.stats.stattools import durbin_watson
    dw_gls = durbin_watson(resid_gls)
    
    print(f"GLS-AR1 → Beta: {self.beta_gls} | Alpha: {self.alpha_gls}")
    print(f"R2_GLS: {self.r2_gls} vs R2_OLS: {self.r_squared}")
    print(f"DW post-GLS: {dw_gls:.4f} "
          f"({'✅ autocorr corregida' if 1.5 < dw_gls < 2.5 else '⚠️ revisar'})")
    
    return {
        'beta_gls':  self.beta_gls,
        'alpha_gls': self.alpha_gls,
        'r2_gls':    self.r2_gls,
        'rho':       self.rho_ar1,
        'dw_gls':    round(dw_gls, 4)
    }

    
def diagnostico_r2_gls_v2(bench, sec, from_date, to_date):
    """
    Versión mejorada del Diagnóstico 2.
    Usa GLS-AR1 en lugar de WLS para la comparación de R2.
    """
    try:
        m = model(bench, sec, 6, model_type='macro')
        m.synchronise_timeseries(from_date, to_date)
        
        if len(m.timeseries) < 10:
            return None
        
        m.compute_linear_reg()
        r2_ols = m.r_squared
        
        res_gls = m.run_gls_ar1()
        r2_gls  = res_gls['r2_gls']
        
        ratio = r2_ols / r2_gls if r2_gls > 0 else None
        
        alerta = (
            '⚠️  R2_OLS inflado (posible factor inútil)' if ratio and ratio > 2
            else '✅ R2_OLS y R2_GLS consistentes'
        )
        
        return {
            'benchmark': bench, 'security': sec,
            'r2_ols': r2_ols, 'r2_gls': r2_gls,
            'ratio': round(ratio, 3) if ratio else None,
            'rho_ar1': res_gls['rho'],
            'alerta': alerta, 'n_obs': m.n
        }
    except Exception as e:
        return {'error': str(e)}


# ─────────────────────────────────────────────────────────────
# DIAGNÓSTICO 3 (Kan-Zhang): EIV-adjusted t-ratio (Shanken 1992)
# Corrige el t-ratio por error de medición en betas
# ─────────────────────────────────────────────────────────────
def eiv_adjusted_t_ratio(bench, sec, from_date, to_date):
    """
    Kan & Zhang Diagnóstico 3 — Shanken (1992) EIV adjustment.
    
    El t-ratio estándar de OLS ignora que beta fue estimada con error.
    Shanken propone ajustar el denominador del t-ratio para corregir esto.
    
    Fórmula (Ec. 33 del paper):
    
    t*_OLS = γ̂₁ / sqrt[ s²(γ̂₁)/T + (γ̂₁/ŝ_g)² * (s²(γ̂₁)/T - ŝ_g²/T) ]
    
    En tu framework de series de tiempo (no two-pass exacto), el análogo es:
    ajustar el error estándar de beta por la varianza del factor.
    
    Si |t_EIV| > |t_OLS| * 0.7 → el ajuste es moderado (factor genuino)
    Si |t_EIV| << |t_OLS|     → el ajuste es fuerte (factor ruidoso)
    """
    try:
        m = model(bench, sec, 6, model_type='macro')
        m.synchronise_timeseries(from_date, to_date)
        
        if len(m.timeseries) < 10:
            return None
        
        m.compute_linear_reg()
        
        # Varianza del factor (s_g^2 en el paper)
        x_vals = m.timeseries[bench].values
        s_g_squared = np.var(x_vals, ddof=1)
        
        # t-ratio estándar
        t_ols = m.beta / np.sqrt(m.b1_variance)
        
        # Varianza de gamma_hat (equivalente en time series)
        s2_gamma = m.mco_model_variance / m.sxx  # = b1_variance
        
        # EIV adjustment (Shanken 1992, Eq. 33 adaptado)
        # El ajuste escala el denominador por (1 + γ²/s_g²)
        if s_g_squared > 0 and m.beta != 0:
            shanken_factor = 1 + (m.beta**2 / s_g_squared)
            var_eiv = s2_gamma * shanken_factor
            t_eiv = m.beta / np.sqrt(var_eiv)
        else:
            t_eiv = t_ols
            shanken_factor = 1.0
        
        p_ols = 2 * (1 - stats.t.cdf(abs(t_ols), df=m.df))
        p_eiv = 2 * (1 - stats.t.cdf(abs(t_eiv), df=m.df))
        
        return {
            'benchmark': bench,
            'security': sec,
            'beta': round(m.beta, 4),
            't_ols': round(t_ols, 4),
            't_eiv': round(t_eiv, 4),
            'p_ols': round(p_ols, 4),
            'p_eiv': round(p_eiv, 4),
            'shanken_factor': round(shanken_factor, 4),
            'r2': m.r_squared,
            'n_obs': m.n,
            'conclusion': (
                '✅ Factor genuino (EIV no cambia conclusión)' 
                if (p_ols < 0.05 and p_eiv < 0.05)
                else '⚠️  EIV cambia significancia — factor ruidoso'
                if (p_ols < 0.05 and p_eiv >= 0.05)
                else '❌ No significativo en ambos'
            )
        }
    except Exception as e:
        return {'error': str(e)}


# ─────────────────────────────────────────────────────────────
# DIAGNÓSTICO 4 (Kan-Zhang): Test conjunto de subperíodos
# ─────────────────────────────────────────────────────────────
def diagnostico_subperiodos(bench, sec, from_date, to_date, 
                             significance=0.1, min_obs_por_mitad=5):
    """
    Kan & Zhang Diagnóstico 4:
    
    Divide el período en dos mitades y verifica si:
    1. Ambas mitades rechazan H0: beta = 0
    2. En la MISMA DIRECCIÓN (mismo signo de correlación)
    
    Un factor genuino → relación consistente en ambas mitades.
    Un factor inútil → signos aleatorios, inconsistentes.
    
    Este diagnóstico resuelve tu problema de N pequeño en 2026:
    si la relación es real, debe aparecer en ambas mitades.
    """
    from_dt = pd.to_datetime(from_date)
    to_dt = pd.to_datetime(to_date)
    mid = from_dt + (to_dt - from_dt) / 2
    
    try:
        # Primera mitad
        m1 = model(bench, sec, 6, model_type='macro')
        m1.synchronise_timeseries(from_dt.strftime('%Y-%m-%d'), 
                                   mid.strftime('%Y-%m-%d'))
        
        # Segunda mitad
        m2 = model(bench, sec, 6, model_type='macro')
        m2.synchronise_timeseries(mid.strftime('%Y-%m-%d'), 
                                   to_dt.strftime('%Y-%m-%d'))
        
        n1 = len(m1.timeseries)
        n2 = len(m2.timeseries)
        
        if n1 < min_obs_por_mitad or n2 < min_obs_por_mitad:
            return {
                'veredicto': 'INDETERMINADO',
                'motivo': f'Muestra insuficiente: mitad1={n1}, mitad2={n2}'
            }
        
        m1.compute_linear_reg()
        m2.compute_linear_reg()
        
        # p-values de cada mitad
        t1 = m1.beta / np.sqrt(m1.b1_variance) if m1.b1_variance > 0 else 0
        t2 = m2.beta / np.sqrt(m2.b1_variance) if m2.b1_variance > 0 else 0
        
        p1 = 2 * (1 - stats.t.cdf(abs(t1), df=m1.df))
        p2 = 2 * (1 - stats.t.cdf(abs(t2), df=m2.df))
        
        sig1 = p1 < significance
        sig2 = p2 < significance
        mismo_signo = (m1.correlation * m2.correlation) > 0
        ambas_sig = sig1 and sig2
        
        if ambas_sig and mismo_signo:
            veredicto = '✅ FACTOR CONSISTENTE (ambas mitades significativas, mismo signo)'
        elif mismo_signo and not ambas_sig:
            veredicto = '🟡 SEÑAL DÉBIL (mismo signo pero no ambas sig.)'
        elif not mismo_signo:
            veredicto = '⚠️  FACTOR INESTABLE (signo cambia entre mitades)'
        else:
            veredicto = '❌ NO SIGNIFICATIVO'
        
        return {
            'benchmark': bench,
            'security': sec,
            'r2_mitad1': round(m1.r_squared, 4),
            'r2_mitad2': round(m2.r_squared, 4),
            'corr_mitad1': round(m1.correlation, 4),
            'corr_mitad2': round(m2.correlation, 4),
            'p_mitad1': round(p1, 4),
            'p_mitad2': round(p2, 4),
            'mismo_signo': mismo_signo,
            'ambas_sig': ambas_sig,
            'n1': n1,
            'n2': n2,
            'mid_date': mid.strftime('%Y-%m-%d'),
            'veredicto': veredicto
        }
    except Exception as e:
        return {'veredicto': 'ERROR', 'motivo': str(e)}


# ─────────────────────────────────────────────────────────────
# FUNCIÓN MAESTRA: Pipeline completo Kan-Zhang
# Corre los 4 diagnósticos en secuencia y filtra factores
# ─────────────────────────────────────────────────────────────
def pipeline_kan_zhang(benchmarks_x, securities_y, from_date, to_date,
                        significance=0.05, umbral_corr_factores=0.7,
                        min_obs=20):
    """
    Pipeline completo inspirado en Kan & Zhang (1999).
    
    FLUJO:
    ─────
    0. Correlación entre factores (multicolinealidad)
    1. ¿Betas ≠ 0 en el primer paso? (factor útil vs inútil)
    2. R2_OLS vs R2_WLS (¿inflación espuria?)
    3. EIV-adjusted t-ratio (corrección Shanken)
    4. Test de subperíodos (consistencia de la señal)
    
    OUTPUT:
    ───────
    Retorna lista de factores que PASAN todos los filtros,
    listos para usar en buscar_mejor_r2 con más confianza.
    """
    
    print("\n" + "█"*70)
    print(f"{'PIPELINE KAN-ZHANG (1999) — DETECCIÓN DE FACTORES ÚTILES':^70}")
    print(f"{'Período: ' + str(from_date) + ' → ' + str(to_date):^70}")
    print("█"*70)
    
    # ── PASO 0: Correlación entre factores ──────────────────
    print("\n⟹  PASO 0: Multicolinealidad entre factores")
    df_cov = diagnostico_covarianza_factores(benchmarks_x, from_date, to_date,
                                              umbral_corr=umbral_corr_factores)
    pares_problematicos = df_cov[df_cov['alerta'].str.contains('MULTI')][
        ['factor_1', 'factor_2', 'correlacion']].values.tolist()
    
    # ── PASO 1: Diagnóstico de betas ────────────────────────
    print("\n⟹  PASO 1: Test de betas (Kan-Zhang Diag. 1)")
    df_diag1, utiles, inutiles = diagnostico_factores_utiles(
        benchmarks_x, securities_y, from_date, to_date, significance)
    
    # ── PASOS 2, 3, 4: Por par ──────────────────────────────
    print("\n⟹  PASOS 2-4: Análisis detallado por par factor-activo")
    
    resumen_final = []
    
    for sec in securities_y:
        for bench in benchmarks_x:
            if sec == bench:
                continue
            
            print(f"\n  📊 {bench} → {sec}")
            
            # Diag 2
            d2 = diagnostico_r2_gls(bench, sec, from_date, to_date)
            if d2 and 'error' not in d2:
                print(f"     [Diag 2] R2_OLS={d2.get('r2_ols')} | "
                      f"R2_WLS={d2.get('r2_wls')} | {d2.get('alerta','')}")
            
            # Diag 3
            d3 = eiv_adjusted_t_ratio(bench, sec, from_date, to_date)
            if d3 and 'error' not in d3:
                print(f"     [Diag 3] t_OLS={d3.get('t_ols')} | "
                      f"t_EIV={d3.get('t_eiv')} | {d3.get('conclusion','')}")
            
            # Diag 4
            d4 = diagnostico_subperiodos(bench, sec, from_date, to_date,
                                          significance=significance,
                                          min_obs_por_mitad=max(5, min_obs//4))
            if d4:
                print(f"     [Diag 4] {d4.get('veredicto','')}")
            
            # Puntaje compuesto
            puntaje = 0
            if bench in utiles:                            puntaje += 2
            if d2 and 'OK' in str(d2.get('alerta','')):   puntaje += 1
            if d3 and 'genuino' in str(d3.get('conclusion','')): puntaje += 1
            if d4 and '✅' in str(d4.get('veredicto','')): puntaje += 2
            
            resumen_final.append({
                'security': sec,
                'benchmark': bench,
                'puntaje_kz': puntaje,
                'max_puntaje': 6,
                'recomendacion': (
                    '🟢 USAR' if puntaje >= 4 
                    else '🟡 CON CAUTELA' if puntaje >= 2 
                    else '🔴 DESCARTAR'
                ),
                'diag1_util': bench in utiles,
                'diag4_consistente': d4.get('ambas_sig', False) if d4 else False,
                'r2_ols': d2.get('r2_ols') if d2 else None,
                't_eiv': d3.get('t_eiv') if d3 else None
            })
    
    df_resumen = pd.DataFrame(resumen_final).sort_values('puntaje_kz', ascending=False)
    
    print("\n" + "═"*70)
    print(f"{'RESUMEN FINAL — RECOMENDACIONES':^70}")
    print("═"*70)
    for _, row in df_resumen.iterrows():
        print(f"  {row['benchmark']:<15} → {row['security']:<10} | "
              f"Puntaje: {row['puntaje_kz']}/6 | {row['recomendacion']}")
    print("═"*70)
    
    # Factores recomendados para usar
    usar = df_resumen[df_resumen['puntaje_kz'] >= 4]['benchmark'].unique().tolist()
    cautela = df_resumen[
        (df_resumen['puntaje_kz'] >= 2) & 
        (df_resumen['puntaje_kz'] < 4)
    ]['benchmark'].unique().tolist()
    descartar = df_resumen[df_resumen['puntaje_kz'] < 2]['benchmark'].unique().tolist()
    
    print(f"\n🟢 Factores recomendados: {usar}")
    print(f"🟡 Usar con cautela:      {cautela}")
    print(f"🔴 Descartar:             {descartar}")
    
    if pares_problematicos:
        print(f"\n⚠️  Pares con multicolinealidad: {pares_problematicos}")
    
    return df_resumen, usar, cautela, descartar





def chow_test(bench, sec, fecha_quiebre, from_date=None, to_date=None, 
              significance=0.05):
    """
    Prueba de Chow (1960): detecta si hubo cambio estructural 
    en la relación bench → sec alrededor de fecha_quiebre.
    
    H0: los coeficientes (alpha, beta) son iguales antes y después
    H1: al menos uno de los coeficientes cambió
    
    Estadístico F:
        F = [(RSS_total - RSS_pre - RSS_post) / k] / 
            [(RSS_pre + RSS_post) / (n_pre + n_post - 2k)]
    
    donde k = número de parámetros (2: alpha y beta)
    
    Bajo H0: F ~ F(k, n_pre + n_post - 2k)
    """
    from scipy import stats as st_
    
    fecha_q = pd.to_datetime(fecha_quiebre)
    
    # Modelo completo (toda la muestra)
    m_full = model(bench, sec, 6, model_type='macro')
    m_full.synchronise_timeseries(from_date or '2026-01-01', 
                                   to_date or '2026-04-30')
    
    # Submuestras
    m_pre = model(bench, sec, 6, model_type='macro')
    m_pre.synchronise_timeseries(from_date or '2026-01-01',
                                  fecha_q.strftime('%Y-%m-%d'))
    
    m_post = model(bench, sec, 6, model_type='macro')
    m_post.synchronise_timeseries(fecha_q.strftime('%Y-%m-%d'),
                                   to_date or '2026-04-30')
    
    n_pre  = len(m_pre.timeseries)
    n_post = len(m_post.timeseries)
    n_full = len(m_full.timeseries)
    
    if n_pre < 5 or n_post < 5:
        return {'error': f'Muestra insuficiente: pre={n_pre}, post={n_post}'}
    
    m_full.compute_linear_reg()
    m_pre.compute_linear_reg()
    m_post.compute_linear_reg()
    
    # Sumas de cuadrados de residuos
    rss_full = m_full.MCO          # RSS modelo restringido (sin quiebre)
    rss_pre  = m_pre.MCO
    rss_post = m_post.MCO
    rss_unrestricted = rss_pre + rss_post  # RSS modelo no restringido
    
    k = 2  # alpha y beta
    
    # Estadístico F de Chow
    numerador   = (rss_full - rss_unrestricted) / k
    denominador = rss_unrestricted / (n_pre + n_post - 2 * k)
    
    if denominador <= 0:
        return {'error': 'Denominador no positivo'}
    
    F_chow = numerador / denominador
    p_value = 1 - st_.f.cdf(F_chow, dfn=k, dfd=(n_pre + n_post - 2 * k))
    
    hay_quiebre = p_value < significance
    
    resultado = {
        'benchmark':      bench,
        'security':       sec,
        'fecha_quiebre':  fecha_quiebre,
        'F_chow':         round(F_chow, 4),
        'p_value':        round(p_value, 4),
        'n_pre':          n_pre,
        'n_post':         n_post,
        'beta_pre':       round(m_pre.beta, 4),
        'beta_post':      round(m_post.beta, 4),
        'r2_pre':         m_pre.r_squared,
        'r2_post':        m_post.r_squared,
        'corr_pre':       round(m_pre.correlation, 4),
        'corr_post':      round(m_post.correlation, 4),
        'cambio_beta':    round(m_post.beta - m_pre.beta, 4),
        'hay_quiebre':    hay_quiebre,
        'veredicto':      (
            '🔴 CAMBIO ESTRUCTURAL CONFIRMADO' if hay_quiebre
            else '✅ Sin evidencia de cambio estructural'
        )
    }
    
    print(f"\n{'─'*60}")
    print(f"CHOW TEST: {bench} → {sec}")
    print(f"Fecha de quiebre: {fecha_quiebre}")
    print(f"{'─'*60}")
    print(f"  F = {F_chow:.4f} | p-value = {p_value:.4f}")
    print(f"  Beta PRE:  {m_pre.beta:.4f}  (r={m_pre.correlation:.4f}, R2={m_pre.r_squared:.4f})")
    print(f"  Beta POST: {m_post.beta:.4f}  (r={m_post.correlation:.4f}, R2={m_post.r_squared:.4f})")
    print(f"  {resultado['veredicto']}")
    
    return resultado


def chow_multiplos_factores(benchmarks_x, securities_y, fechas_quiebre,
                             from_date=None, to_date=None, significance=0.05):
    """
    Aplica Chow test a todas las combinaciones factor-activo
    para múltiples fechas de quiebre candidatas.
    
    Útil para confirmar cuál fecha marca el cambio estructural
    más significativo estadísticamente.
    """
    resultados = []
    
    for sec in securities_y:
        for bench in benchmarks_x:
            if sec == bench:
                continue
            for fecha in fechas_quiebre:
                res = chow_test(bench, sec, fecha, 
                                from_date=from_date, 
                                to_date=to_date,
                                significance=significance)
                if 'error' not in res:
                    resultados.append(res)
    
    df = pd.DataFrame(resultados)
    
    # Ordenar por p-value (los quiebres más significativos primero)
    df = df.sort_values('p_value')
    
    print("\n" + "█"*70)
    print(f"{'RESUMEN CHOW TEST — MÚLTIPLES FECHAS':^70}")
    print("█"*70)
    print(f"{'Benchmark':<15} {'Security':<10} {'Fecha':<12} "
          f"{'F':>8} {'p-val':>7} {'ΔBeta':>8}  Veredicto")
    print("─"*70)
    for _, r in df.iterrows():
        print(f"{r['benchmark']:<15} {r['security']:<10} "
              f"{r['fecha_quiebre']:<12} {r['F_chow']:>8.3f} "
              f"{r['p_value']:>7.4f} {r['cambio_beta']:>8.4f}  "
              f"{'🔴' if r['hay_quiebre'] else '✅'}")
    
    return df



def run_gls_ar1(self, significance=0.05):
    """
    GLS para series de tiempo con errores AR(1).
    
    Corrige autocorrelación en los errores mediante la
    transformación de Cochrane-Orcutt:
    
    1. Estima rho (autocorrelación de residuos) con OLS
    2. Transforma: y* = y_t - rho*y_{t-1}
                   x* = x_t - rho*x_{t-1}
    3. Corre OLS sobre variables transformadas
    
    Más apropiado que WLS cuando Durbin-Watson < 1.5 o > 2.5
    """
    if self.residuals is None:
        raise ValueError("Corre compute_linear_reg() primero")
    
    # Paso 1: estimar rho
    e = self.residuals
    rho = np.sum(e[1:] * e[:-1]) / np.sum(e[:-1]**2)
    
    print(f"ρ estimado (AR1): {rho:.4f}")
    if abs(rho) < 0.1:
        print("→ Autocorrelación baja. GLS-AR1 puede no ser necesario.")
    
    # Paso 2: transformación Cochrane-Orcutt
    y = self.y
    x = self.x
    
    y_star = y[1:] - rho * y[:-1]
    x_star = x[1:] - rho * x[:-1]
    
    # Paso 3: OLS sobre transformadas
    slope, intercept, r, p, se = stats.linregress(x_star, y_star)
    
    self.beta_gls  = round(slope, self.decimals)
    self.alpha_gls = round(intercept, self.decimals)
    self.r2_gls    = round(r**2, self.decimals)
    self.rho_ar1   = round(rho, 4)
    
    pred_gls   = intercept + slope * x_star
    resid_gls  = y_star - pred_gls
    
    # Test DW sobre residuos GLS
    from statsmodels.stats.stattools import durbin_watson
    dw_gls = durbin_watson(resid_gls)
    
    print(f"GLS-AR1 → Beta: {self.beta_gls} | Alpha: {self.alpha_gls}")
    print(f"R2_GLS: {self.r2_gls} vs R2_OLS: {self.r_squared}")
    print(f"DW post-GLS: {dw_gls:.4f} "
          f"({'✅ autocorr corregida' if 1.5 < dw_gls < 2.5 else '⚠️ revisar'})")
    
    return {
        'beta_gls':  self.beta_gls,
        'alpha_gls': self.alpha_gls,
        'r2_gls':    self.r2_gls,
        'rho':       self.rho_ar1,
        'dw_gls':    round(dw_gls, 4)
    }

### ESTIMACION DE BETAS POR OLDRICH 1973
# ─────────────────────────────────────────────────────────────
# ESTIMADOR BAYESIANO DE VASICEK (1973)
# Agrega este método a tu clase `model`
# ─────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────
# FUNCIÓN AUXILIAR: estimar b_prior y s2_prior desde datos históricos
# ─────────────────────────────────────────────────────────────

def estimar_prior_desde_periodo(bench, sec, from_date_prior, to_date_prior,
                                 verbose=True):
    """
    Estima la beta prior (b') y su varianza (s'²_b) corriendo OLS
    sobre un período histórico diferente al de análisis.
    
    En el contexto de tu proyecto:
    - El período prior es el PRE-quiebre (antes del 28-feb-2026)
    - El período de análisis es el POST-quiebre
    
    Así la prior captura el régimen anterior y la posterior
    incorpora qué tan bien ese régimen aún aplica hoy.
    
    Retorna: (b_prior, s2_prior, r2_prior, n_prior)
    """
    m_prior = model(bench, sec, 6, model_type='macro')
    m_prior.synchronise_timeseries(from_date_prior, to_date_prior)
    
    if len(m_prior.timeseries) < 5:
        print(f"⚠️  Muestra prior insuficiente: {len(m_prior.timeseries)} obs")
        return None, None, None, None
    
    m_prior.compute_linear_reg()
    
    b_prior  = m_prior.beta
    s2_prior = m_prior.b1_variance   # varianza de la beta en el período prior
    
    if verbose:
        print(f"\n  Prior estimada de {from_date_prior} a {to_date_prior}")
        print(f"  b' = {b_prior:.6f} | s'²_b = {s2_prior:.8f} | "
              f"R² = {m_prior.r_squared:.4f} | N = {m_prior.n}")
    
    return b_prior, s2_prior, m_prior.r_squared, m_prior.n


# ─────────────────────────────────────────────────────────────
# FUNCIÓN MAESTRA: Vasicek completo con prior estimada del régimen previo
# ─────────────────────────────────────────────────────────────

def vasicek_pipeline(benchmarks_recomendados, securities_y,
                     fecha_quiebre,
                     ventana_prior_dias=40,
                     from_date_post=None,
                     to_date_post=None,
                     s2_prior_difusa=1.0):
    """
    Pipeline completo de estimación Bayesiana de Vasicek (1973).
    
    FLUJO:
    ──────
    1. Estima la prior (b', s'²) usando datos PRE-quiebre
    2. Estima la muestra (b, s²) usando datos POST-quiebre  
    3. Combina vía Ecuaciones 15-16 de Vasicek
    4. Diagnóstica qué fuente domina y cuánto se ajusta la beta
    
    Parámetros:
    ───────────
    benchmarks_recomendados : list
        Factores que pasaron el pipeline Kan-Zhang (🟢 y 🟡)
    
    securities_y : list
        Activos a modelar (DXY, USDMXN, etc.)
    
    fecha_quiebre : str  'YYYY-MM-DD'
        Fecha del cambio de régimen (28-feb-2026 en tu caso)
    
    ventana_prior_dias : int
        Cuántos días antes del quiebre usar para estimar la prior.
        Default: 40 días hábiles ≈ 2 meses
    
    from_date_post, to_date_post : str
        Rango del período POST-quiebre para la muestra actual.
        Si None: usa desde fecha_quiebre hasta hoy.
    
    s2_prior_difusa : float
        Varianza a usar cuando la prior tiene muy pocos datos.
        Valor grande = prior difusa = más peso a la muestra.
        Default: 1.0
    """
    from datetime import datetime, timedelta
    import pandas as pd
    
    fecha_q    = pd.to_datetime(fecha_quiebre)
    fecha_pre  = fecha_q - pd.Timedelta(days=ventana_prior_dias)
    
    from_post  = from_date_post or fecha_quiebre
    to_post    = to_date_post   or datetime.today().strftime('%Y-%m-%d')
    
    print("\n" + "█"*70)
    print(f"{'PIPELINE VASICEK (1973) — BETAS BAYESIANAS':^70}")
    print(f"{'Quiebre: ' + fecha_quiebre:^70}")
    print(f"{'Prior: ' + fecha_pre.strftime('%Y-%m-%d') + ' → ' + fecha_quiebre:^70}")
    print(f"{'Post:  ' + from_post + ' → ' + to_post:^70}")
    print("█"*70)
    
    todos_resultados = []
    
    for sec in securities_y:
        for bench in benchmarks_recomendados:
            if sec == bench:
                continue
            
            print(f"\n{'─'*60}")
            print(f"  PAR: {bench} → {sec}")
            print(f"{'─'*60}")
            
            # ── PASO 1: Estimar prior desde período pre-quiebre ──
            print("\n  [PRIOR — régimen pre-quiebre]")
            b_prior, s2_prior, r2_prior, n_prior = estimar_prior_desde_periodo(
                bench, sec,
                fecha_pre.strftime('%Y-%m-%d'),
                fecha_quiebre
            )
            
            # Si la prior tiene muy pocos datos, usar prior difusa
            if b_prior is None or n_prior < 5:
                print(f"  ⚠️  Prior insuficiente → usando prior difusa "
                      f"(b'=0, s'²={s2_prior_difusa})")
                b_prior  = 0.0
                s2_prior = s2_prior_difusa
            elif r2_prior < 0.05:
                print(f"  ⚠️  R²_prior bajo ({r2_prior:.4f}) → prior poco informativa")
                # Inflamos la varianza prior para darle menos peso
                s2_prior = s2_prior * 3.0
                print(f"     s'²_prior ajustada a {s2_prior:.8f} (x3)")
            
            # ── PASO 2: Estimar muestra en período post-quiebre ──
            print("\n  [MUESTRA — período de análisis]")
            m_post = model(bench, sec, 6, model_type='macro')
            m_post.synchronise_timeseries(from_post, to_post)
            
            if len(m_post.timeseries) < 5:
                print(f"  ❌ Muestra post insuficiente "
                      f"({len(m_post.timeseries)} obs). Saltando.")
                continue
            
            m_post.compute_linear_reg()
            
            print(f"  b_OLS = {m_post.beta:.6f} | "
                  f"s²_b = {m_post.b1_variance:.8f} | "
                  f"R² = {m_post.r_squared:.4f} | "
                  f"N = {m_post.n}")
            
            # ── PASO 3: Estimador Vasicek ─────────────────────────
            print("\n  [VASICEK — combinación Bayesiana]")
            res = m_post.vasicek_beta(b_prior=b_prior, s2_prior=s2_prior)
            
            # ── PASO 4: ¿Cuánto cambió respecto a OLS? ───────────
            cambio_pct = abs(res['b_vasicek'] - res['b_mco']) / \
                         (abs(res['b_mco']) + 1e-10) * 100
            
            res.update({
                'b_prior_usado':   b_prior,
                's2_prior_usado':  s2_prior,
                'r2_prior':        r2_prior,
                'n_prior':         n_prior,
                'r2_post':         m_post.r_squared,
                'n_post':          m_post.n,
                'cambio_vs_ols_pct': round(cambio_pct, 2)
            })
            
            todos_resultados.append(res)
    
    # ── Tabla resumen final ───────────────────────────────────
    if todos_resultados:
        df_res = pd.DataFrame(todos_resultados)
        
        cols_resumen = ['benchmark', 'security', 'n_post',
                        'b_mco', 'b_prior_usado', 'b_vasicek',
                        's2_mco', 's2_posterior',
                        'peso_mco_pct', 'peso_prior_pct',
                        'cambio_vs_ols_pct', 'r2_post']
        
        print("\n\n" + "═"*70)
        print(f"{'RESUMEN FINAL — BETAS VASICEK':^70}")
        print("═"*70)
        
        for _, r in df_res.iterrows():
            domina = ("📊 datos" if r['peso_mco_pct'] > 60 
                      else "📚 prior" if r['peso_prior_pct'] > 60 
                      else "⚖️  balance")
            
            print(f"\n  {r['benchmark']:<15} → {r['security']:<8} "
                  f"(N={r['n_post']}, R²={r['r2_post']:.3f})")
            print(f"  b_OLS={r['b_mco']:>9.5f}  |  "
                  f"b_prior={r['b_prior_usado']:>9.5f}  |  "
                  f"b_Vasicek={r['b_vasicek']:>9.5f}")
            print(f"  Peso datos={r['peso_mco_pct']:5.1f}%  |  "
                  f"Peso prior={r['peso_prior_pct']:5.1f}%  |  "
                  f"Ajuste={r['cambio_vs_ols_pct']:5.1f}%  |  "
                  f"Domina: {domina}")
        
        print("\n" + "═"*70)
        print("  GUÍA DE LECTURA:")
        print("  • b_Vasicek ≈ b_OLS   → muestra suficiente, prior irrelevante")
        print("  • b_Vasicek ≈ b_prior → N pequeño, prior domina (cautela)")
        print("  • Ajuste alto         → el régimen cambió ó N muy pequeño")
        print("  • s²_posterior < ambas varianzas individuales → Ec.16 cumplida")
        print("═"*70)
        
        return df_res[cols_resumen]
    
    return None