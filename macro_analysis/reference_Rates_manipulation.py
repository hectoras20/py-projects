import pandas as pd
import matplotlib.pyplot as plt
import importlib
import csv



def load_timeseries(info):
    # directory = '/Users/hectorastudillo/py-proyects/Actuary_Science/projects/quantitative_finance/market_data_c/'
    # path = directory + ric + '.csv'
    path = 'macro/' + info + '.csv'
    raw_data = pd.read_csv(path)
    # Si usamos información de Investing usamos la siguiente linea
    # raw_data = corrector(raw_data)
    t = pd.DataFrame()
    t['Fecha'] = pd.to_datetime(raw_data['Fecha'],format='mixed', dayfirst=True)
    t['Tasa'] = raw_data['Tasa']
    t = t.sort_values(by='Fecha', ascending=True)
    t['Tasa_previa'] = t['Tasa'].shift(1) # This function shift "recorrer" one cell. 
    t['Var'] = t['Tasa'] / t['Tasa_previa'] - 1
    t = t.dropna()
    t = t.reset_index(drop=True)
    return t


def synchronise_timseries_df(info1, info2):
    timeseries_x = load_timeseries(info2)
    timeseries_y = load_timeseries(info1)

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
    timeseries['Tasa_x'] = timeseries_x['Tasa']
    timeseries['Tasa_y'] = timeseries_y['Tasa']
    timeseries['Var_x'] = timeseries_x['Var']
    timeseries['Var_y'] = timeseries_y['Var']
    timeseries['Dif'] = timeseries['Tasa_y'] - timeseries['Tasa_x']
    return timeseries



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
    
    
def cargar_archivo(df, archivo = "reference_rate.csv"):
    with open(archivo, "w", encoding="UTF8", newline="") as file:
        writer = csv.writer(file)
    
        # encabezado
        writer.writerow(df.columns.tolist())
    
        # múltiples líneas
        writer.writerows(df.to_numpy().tolist())
    
info1 = 'refRateMX'
info2 = 'refRateUSA'

info = synchronise_timseries_df(info1, info2)

info_filtrado = filtrar_por_rango(
    info,
    fecha_inicio='2024-01-01',
    fecha_fin='2025-06-30'
)

# info_mensual = primer_dia_de_cada_mes(info)

info_mensual_prom = promedio_mensual(info)

plot_timeseries(info_filtrado)

cargar_archivo(info_mensual_prom)
