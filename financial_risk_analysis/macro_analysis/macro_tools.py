import pandas as pd
import matplotlib.pyplot as plt
import importlib
import csv
from pathlib import Path
import numpy as np
import scipy.stats as st


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
    

    if columna == 'Cierre':
        # COMPUTE RETURNS
        t['close_previous'] = t['Cierre'].shift(1) 
        if log_returns == True:
            t['return_' + info] = np.log(t['Cierre'] / t['close_previous'])
        else:
            t['return_' + info] = t['Cierre'] / t['close_previous'] - 1
        
        mean_cx = t['return_' + info].mean()
        std_cx = t['return_'+ info].std()
        t['standard_'+ info] = (t['return_'+ info] - mean_cx) / std_cx
                    
    
    
    # CIMPUTE DIFFERENCES FOR RATES 
    if columna in ['Tasa', 'Diferencia']:
        t['rate_previous'] = t[columna].shift(1)
        t['dif_bps_' + info] = (t['rate_previous'] - t[columna]) * 100
        
        mean_cx = t['dif_bps_'+info].mean()
        std_cx = t['dif_bps_'+info].std()
        t['standard_'+ info] = (t['dif_bps_'+info] - mean_cx) / std_cx
    
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
def synchronise_timseries_df(info1, info2, columna1=str, columna2=str, name1 = None, name2 = None, dif = False, suma = False, from_date = 'aaaa-mm-dd', to_date = 'aaaa-mm-dd', log_returns = False):
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
    
    if columna1 != columna2:
        timeseries[name1] = timeseries_x['standard_'+info1]
        timeseries[name2] = timeseries_y['standard_'+info2]
    if columna1 == 'Cierre' and columna2 == 'Cierre':
        timeseries[name1] = timeseries_x['return_' + info1]
        timeseries[name2] = timeseries_y['return_' + info2]
    elif columna1 in ['Tasa', 'Diferencia'] and columna2 in ['Tasa', 'Diferencia']:
        timeseries[name1] = timeseries_x['dif_bps_' + info1]
        timeseries[name2] = timeseries_y['dif_bps_' + info2]
    
        
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

def plot_normalized_timeseries(rics, columns = list, from_date='aaaa-mm-dd', to_date='aaaa-mm-dd', base_value=100):
    """
    Plot normalized price time series for a group of assets.

    Prices are first filtered by the selected date window and then
    normalized so that the first observation within the window equals
    the chosen base_value.

    Price_norm = (Price_t / Price_from) * base_value
    
    length(rics) must be equal to the length of columnas
    """

    plt.figure(figsize=(12,6))

    for a, b in zip(rics, columns):
        t = load_timeseries(a, b)

        # Date filtering
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
    
        plt.plot(t['Fecha'], normalized, label=a)

    plt.title(f"Normalized Price Series (Base = {base_value})")
    plt.xlabel("Time")
    plt.ylabel(f"Price Index (Base = {base_value})")
    plt.grid(True)
    plt.legend()
    plt.show()

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

def target_column(name_file = str):
    base_path = Path(__file__).parent
    data_path = base_path / "macro_data"
    path = data_path / f"{name_file}.csv"
    df = pd.read_csv(path)
    columns = df.columns
    targets = ['diferencia', 'cierre', 'close', 'tasa', 'rate']
    for col in columns:
        if col.lower() in targets:
            return col
    

class model:
    def __init__(self, security_x, security_y, decimals = 5):
        self.security_x = security_x
        self.security_y = security_y
        self.decimals = decimals
        self.x_type_column = None
        self.y_type_column = None
        self.timeseries = None
        self.x = None
        self.y = None
        self.beta = None
        self.alpha = None
        self.p_value = None
        self.correlation = None
        self.r_squared = None
        self.hypothesis_null = None
        self.predictor_linreg = None
        
    def synchronise_timeseries(self, from_date = 'aaaa-mm-dd', to_date = 'aaaa-mm-dd', log_returns = False):
        self.x_type_column = target_column(self.security_x)
        self.y_type_column = target_column(self.security_y)
        self.timeseries = synchronise_timseries_df(self.security_x, self.security_y, self.x_type_column, self.y_type_column, from_date = from_date, to_date = to_date, log_returns = log_returns)
        if self.timeseries.empty: 
            print('There is a problem with ', self.security, ' and ', self.benchmark, '. There is not information to match')
        
    def plot_timeseries(self, secondary_y = True):
        plot_timeseries(self.timeseries, self.security_x, self.security_y, secondary_y=secondary_y)
        
    def compute_linear_reg(self):
        self.x = self.timeseries[self.security_x].values
        self.y = self.timeseries[self.security_y].values
        # Lineal Regression 
        slope_beta, intercept_alpha, correl_r, p_value, standard_error = st.linregress(x=self.x, y=self.y)
        self.beta = np.round(slope_beta, self.decimals)
        self.alpha = np.round(intercept_alpha, self.decimals)
        self.p_value = np.round(p_value, self.decimals)
        self.correlation = np.round(correl_r, self.decimals)
        self.r_squared = np.round(correl_r**2, self.decimals)
        self.hypothesis_null = p_value > 0.5
        self.predictor_linreg = intercept_alpha + slope_beta * self.x
        
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
'''

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