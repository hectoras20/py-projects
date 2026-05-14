import pandas as pd
import matplotlib.pyplot as plt
import importlib
import csv
from pathlib import Path
import numpy as np
import math
import scipy.stats as st

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




# Quiero simplemente cargar mi información, pasandole el nombre del archivo cargado previamente en el directorio de  trabajo
def load_timeseries(info, columna=str, tipo_var=None, from_date=None, to_date=None, log_returns = False):
    '''
    info str - Nombre del archivo (sin .csv)
    columna str - nombre de la columna a trabajar
    tipo_var str - tipo de variación:
        None     - sin variación
        'lag'    - variación t vs t-1
        'dec_base' - diciembre año anterior como base
        'yoy'    - mismo mes vs año anterior
    '''

    # path = info + '.csv'
    base_path = Path(__file__).parent  # carpeta donde está market_data.py
    data_path = base_path / "macro_data"
    path = data_path / f"{info}.csv"
    # print("Buscando en:", path)
    # print("Existe?", path.exists())
    clean_data = pd.read_csv(path)
    
    # clean_data = pd.read_csv(path, sep=None, engine='python') 
    # sep=None - pandas intenta detectar automáticamente el separador
    # engine='python' - necesario para que esa autodetección funcione bien
    clean_data.columns = clean_data.columns.str.strip()

    # Ajuste si viene de Investing
    clean_data = corrector(clean_data, columna)


    t = pd.DataFrame()
    t['Fecha'] = pd.to_datetime(clean_data['Fecha'], format='mixed', dayfirst=True)
    t[columna] = clean_data[columna]

    t = t.sort_values(by='Fecha', ascending=True)

    # Extraer componentes de fecha para poder trabajar las variaciones...
    # t['year'] = t['Fecha'].dt.year
    # t['month'] = t['Fecha'].dt.month

    # 1. VARIACIÓN t vs t-1
    if tipo_var == 'lag':
        t['Var%'] = (t[columna] / t[columna].shift(1) - 1) * 100

    # 2. VARIACIÓN ANUAL (ENERO vs DICIEMBRE)
    elif tipo_var == 'dec_base':
        t = t.sort_values('Fecha')
        
        t['year'] = t['Fecha'].dt.year
        t['month'] = t['Fecha'].dt.month

        # Obtener valor base por año (diciembre del año anterior)
        t['base'] = t.groupby('year')[columna].transform('first')  # placeholder

        # Crear serie con diciembre por año
        dec_values = t[t['month'] == 12].set_index('year')[columna]

        # Mapear diciembre del año anterior
        t['base'] = t['year'].map(lambda y: dec_values.get(y - 1))

        # Caso especial: si no existe diciembre previo → usar primer valor del año
        t['base'] = t['base'].fillna(
            t.groupby('year')[columna].transform('first')
        )

        # Calcular variación
        t['Var%'] = (t[columna] / t['base'] - 1) * 100


    # 3. VARIACIÓN YoY (MISMO MES AÑO ANTERIOR)
    elif tipo_var == 'yoy':
        t['Var%'] = (t[columna] / t[columna].shift(12) - 1)*100

    # Limpieza final
    t = t.dropna()
    t = t.reset_index(drop=True)
 
    
    # Convertir fechas si existen
    if from_date is not None:
        from_date = pd.to_datetime(from_date)
    
    if to_date is not None:
        to_date = pd.to_datetime(to_date)
    
    # FILTERING DATES
    if from_date is not None and to_date is not None:
        t = t[(t['Fecha'] >= from_date) & (t['Fecha'] <= to_date)]
    
    elif to_date is not None:
        t = t[t['Fecha'] <= to_date]
    
    elif from_date is not None:
        t = t[t['Fecha'] >= from_date]
    

    if columna in ['Cierre', 'Close']:
        name_close = 'Cierre' if columna == 'Cierre' else 'Close'
        # COMPUTE RETURNS
        t['close_previous'] = t[name_close].shift(1) 
        if log_returns == False:
            t['return_' + info] = t[name_close] / t['close_previous'] - 1
        elif log_returns == True:
            t['return_' + info] = np.log(t[name_close] / t['close_previous'])
    
    
    # COMPUTE DIFFERENCES FOR RATES 
    if columna in ['Tasa', 'Diferencia']:
        t['rate_previous'] = t[columna].shift(1)
        t['dif_bps_' + info] = (t[columna] - t['rate_previous']) * 100
    
    t = t.dropna()
    t = t.reset_index(drop=True)
    return t

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

# Quiero sincronizar por fecha la información de dos fuentes/consultas. Ejemplo, fecha de bono mx 10 años con bono usa 10 años
def synchronise_timseries_df(info1, info2, columna1=str, columna2=str, name1 = None, name2 = None, dif = False, suma = False, from_date = 'aaaa-mm-dd', to_date = 'aaaa-mm-dd', log_returns = False, model_type = 'macro'):
    if isinstance(info1, str):
        timeseries_x = load_timeseries(info1, columna=columna1, log_returns=log_returns)
        name1 = name1 or info1
    else:
        timeseries_x = info1
        name1 = name1 or columna1

    if isinstance(info2, str):
        timeseries_y = load_timeseries(info2, columna=columna2, log_returns=log_returns)
        name2 = name2 or info2
    else:
        timeseries_y = info2
        name2 = name2 or columna2

    # Los siguientes pasos son necesarios para tener mismas dimensiones en ambos activos, ya que no podemos manipularlas para la reg. lin. si son de diferente tamaño.
    timestamps_x = timeseries_x['Fecha'].values
    timestamps_y = timeseries_y['Fecha'].values
    # Lo que quiero hacer es la intersección de ambas stamstapms, para ello AMBAS LISTAS CREADAS DEBEN DE SER CONJUNTOS, lo hacemos con la funcións set
    timestamps = list(set(timestamps_x) & set(timestamps_y)) # Tal que el resultado de la intersección lo quiero como una lista y no como un conjunto 

    # Ahora hacemos el filtrado para obtener mismas dimensiones, esto lo logramos con Pandas
    timeseries_x = timeseries_x[timeseries_x['Fecha'].isin(timestamps)]
    timeseries_y = timeseries_y[timeseries_y['Fecha'].isin(timestamps)]

    # Re ordenamos ambos subconjuntos 
    timeseries_x = timeseries_x.sort_values(by='Fecha', ascending = True)
    timeseries_y = timeseries_y.sort_values(by='Fecha', ascending = True)

    # Re organizamos el índice de ambos subsets
    timeseries_x = timeseries_x.reset_index(drop = True)
    timeseries_y = timeseries_y.reset_index(drop = True)

    # AHORA DEBEMOS DE CREAR UN DATAFRAME QUE CONTENGA UNICAMENTE LA FECHA, EL CLOSE DE  X, EL CLOSE DE Y y EL RETURN DE AMBOS (rendimiento)
    # Pero la mejor prática es encapsular todo lo anterior dentro de una función! Para después hacer el plot.

    timeseries = pd.DataFrame()
    timeseries['Fecha'] = timeseries_x['Fecha']
    
    
    if model_type in ['macro', 'statistics']:
        timeseries[name1] = timeseries_x['dif_bps_' + info1] if columna1 in ['Tasa', 'Diferencia'] else timeseries_x['return_' + info1]
        timeseries[name2] = timeseries_y['dif_bps_' + info2] if columna2 in ['Tasa', 'Diferencia'] else timeseries_y['return_' + info2]
    elif model_type == 'learning':
        # precios y tasas crudos, sin transformar
        timeseries[name1] = timeseries_x[columna1]
        timeseries[name2] = timeseries_y[columna2]
    
    if dif == True:
        timeseries['Diferencia'] = timeseries_x[columna1] - timeseries_y[columna2]
    if suma == True:
        timeseries['Suma'] = timeseries[name1] + timeseries[name2]
        
    # FILTERING DATES
    timeseries = timeseries.set_index('Fecha')
    timeseries = timeseries.reset_index()
    if from_date != 'aaaa-mm-dd' and to_date != 'aaaa-mm-dd':
        subsetting = (timeseries['Fecha'] >= from_date) & (timeseries['Fecha'] <= to_date)
        timeseries = timeseries.loc[subsetting].reset_index(drop=True)
    elif to_date != 'aaaa-mm-dd':
        subsetting = timeseries['Fecha'] <= to_date
        timeseries = timeseries.loc[subsetting].reset_index(drop=True)
    elif from_date != 'aaaa-mm-dd':
        subsetting = timeseries['Fecha'] >= from_date
        timeseries = timeseries.loc[subsetting].reset_index(drop=True)
    # else, does not make a subsetting and therefore takes all the dates loaded in the database.
    

    return timeseries



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


def plot_normalized_timeseries(rics, from_date=None, to_date=None, base_value=100, title = 'Normalized Price Series'):
    """
    Plot normalized price time series for a group of assets.

    Prices are first filtered by the selected date window and then
    normalized so that the first observation within the window equals
    the chosen base_value.

    Price_norm = (Price_t / Price_from) * base_value
    """

    plt.figure(figsize=(12,6))

    for ric in rics:
        col = target_column(ric)
        t = load_timeseries(ric, col, from_date=from_date, to_date=to_date)

        if t.empty:
            print("No data available for", ric)
            continue

        base = t[col].iloc[0]
        normalized = t[col] / base * base_value
        
        
        plt.plot(t['Fecha'], normalized, label=ric)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(f"Price Index (Base = {base_value})")
    plt.grid(True)
    plt.legend()
    plt.show()


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
        
    def synchronise_timeseries(self, from_date = 'aaaa-mm-dd', to_date = 'aaaa-mm-dd', log_returns = False):
        self.x_type_column = target_column(self.security_x)
        self.y_type_column = target_column(self.security_y)
        self.timeseries = synchronise_timseries_df(self.security_x, self.security_y, self.x_type_column, self.y_type_column, from_date = from_date, to_date = to_date, log_returns = log_returns, model_type=self.model_type)
        
        if self.model_type == 'statistics':
            # estandarizar X
            mean_x = self.timeseries[self.security_x].mean()
            self.std_x  = self.timeseries[self.security_x].std()
            self.timeseries[self.security_x] = (
                self.timeseries[self.security_x] - mean_x
            ) / self.std_x
    
            # estandarizar Y
            mean_y = self.timeseries[self.security_y].mean()
            self.std_y  = self.timeseries[self.security_y].std()
            self.timeseries[self.security_y] = (
                self.timeseries[self.security_y] - mean_y
            ) / self.std_y
    
        elif self.model_type in ['macro', 'statistics']:
            # solo guardar estadísticos, sin transformar
            self.std_x  = self.timeseries[self.security_x].std()
            self.std_y  = self.timeseries[self.security_y].std()
        
        self.n = len(self.timeseries)
        self.df = self.n - 2
        if self.timeseries.empty: 
            print('There is a problem with ', self.security, ' and ', self.benchmark, '. There is not information to match')
        
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
            ts = synchronise_timseries_df(f1, f2, 
                                          target_column(f1), 
                                          target_column(f2),
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