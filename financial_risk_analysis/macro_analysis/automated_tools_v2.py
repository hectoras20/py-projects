# =============================================================================
# automated_tools_v2.py
# =============================================================================
# Evolución de automated_tools.py con:
#   1. Detección explícita de tipo de serie → evita ambigüedad ['Close']/['Tasa']
#   2. Fuentes de datos:
#       · Yahoo Finance  → activos (precios de cierre)
#       · FRED (FED)     → tasas EUA (DGS5, DGS10, DGS2, DFEDTARU, T10YIE, etc.)
#       · Banxico SIE    → CETES y tasas México (SF60633, SF60648, SF43936, etc.)
#   3. La clase `model` unificada: un solo synchronise_timeseries() que acepta
#      cualquier fuente y aplica la transformación correcta según asset_type.
# =============================================================================

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import math
import scipy.stats as st

from pathlib import Path
from scipy.stats import shapiro, t
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from sklearn.preprocessing import PowerTransformer

try:
    import yfinance as yf
except ImportError:
    yf = None  # opcional si solo se usan APIs macro
    
# BASES DE DATOS MANUALES: 
def load_csv(nombre: str, asset_type: str,
             columna: str = None,
             from_date: str = None, to_date: str = None,
             log_returns: bool = False,
             data_dir: str = None) -> pd.DataFrame:
    """
    Lee un CSV desde macro_data/ (o el directorio que especifiques)
    y devuelve el mismo formato que load_yf / load_fred / load_banxico.

    nombre     : str  — nombre del archivo sin .csv (ej. 'BBVA', 'DGS5')
    asset_type : str  — 'price' | 'rate' | 'pct'
    columna    : str  — nombre de la columna de valor. Si None, la detecta
                        automáticamente buscando 'Cierre','Close','Tasa','Rate'
    """
    if data_dir:
        path = Path(data_dir) / f"{nombre}.csv"
    else:
        base_path = Path(__file__).parent
        path = base_path / "macro_data" / f"{nombre}.csv"

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # Detección automática de columna si no se especifica
    if columna is None:
        candidatas = ['Cierre', 'Close', 'Tasa', 'Rate', 'Precio', 'Price']
        for c in candidatas:
            if c in df.columns:
                columna = c
                break
        if columna is None:
            raise ValueError(f"No se encontró columna de valor en {nombre}.csv. "
                             f"Columnas disponibles: {list(df.columns)}")

    # Normalizar fecha
    df['Fecha'] = pd.to_datetime(
        df['Fecha'].astype(str).str.replace('/', '.', regex=False),
        format='mixed', dayfirst=True
    )

    # Renombrar a nombre interno estándar
    col_interna = 'Cierre' if asset_type == ASSET_TYPE_PRICE else 'Tasa'
    df = df.rename(columns={columna: col_interna})[['Fecha', col_interna]].copy()
    df[col_interna] = (
        df[col_interna].astype(str)
        .str.replace(',', '', regex=False)
        .replace('N/E', None)
        .astype(float)
    )
    df = df.dropna().sort_values('Fecha').reset_index(drop=True)

    # Filtrado de fechas
    if from_date:
        df = df[df['Fecha'] >= pd.to_datetime(from_date)]
    if to_date:
        df = df[df['Fecha'] <= pd.to_datetime(to_date)]
    df = df.reset_index(drop=True)

    # Transformaciones (mismo pipeline que las otras fuentes)
    df = _apply_transformations(df, col_interna, nombre, asset_type, log_returns)
    return df


# =============================================================================
# SECCIÓN 1 — DETECCIÓN DE TIPO DE SERIE
# =============================================================================
#
# PROBLEMA QUE RESUELVE:
#   En el flujo anterior, la columna ['Close'] o ['Tasa'] era la única señal
#   para decidir si calcular returns o diferencias en bps. Esto era frágil:
#   si un CSV tenía un nombre de columna diferente o si un dato macro venía
#   como porcentaje (ej. CPI, PMI) sin llamarse 'Tasa', el modelo lo trataría
#   erróneamente como precio y calcularía returns en lugar de bps.
#
# SOLUCIÓN ADOPTADA:
#   Introducir el concepto de `asset_type` como parámetro explícito:
#       'price'  → calcular log/simple returns       (activos, índices, FX)
#       'rate'   → calcular diferencias en bps       (CETES, DGS5, T-Bills, spreads)
#       'pct'    → igual que 'rate' (% macro como CPI, inflación, PMI en %)
#
#   Esto desacopla la lógica del nombre de la columna. Las columnas internas
#   se siguen llamando 'Cierre' (prices) o 'Tasa' (rates/pcts) por compatibilidad
#   con el código existente, pero el asset_type es quien manda en los cálculos.
#
# REGLA DE ORO:
#   · Precio de mercado que sube/baja → asset_type='price'  → return_%
#   · Tasa o % macro ya expresado en % anual → asset_type='rate'  → dif_bps
#   · Indicador en % pero no tasa (ej. Var% YoY de CPI) → asset_type='pct' → dif_bps

ASSET_TYPE_PRICE = 'price'   # activos: Yahoo Finance, precios de cierre
ASSET_TYPE_RATE  = 'rate'    # tasas: CETES, DGS5, DGS10, T-Bills
ASSET_TYPE_PCT   = 'pct'     # porcentajes macro: inflación, CPI, PMI en %


def infer_asset_type(ticker: str) -> str:
    """
    Inferencia heurística del asset_type a partir del ticker.
    Siempre preferible pasar asset_type explícito; esto es fallback.

    Yahoo Finance tickers de tasas reconocidos:
        ^IRX  = T-Bill 13 semanas
        ^FVX  = T-Note 5 años
        ^TNX  = T-Note 10 años
        ^TYX  = T-Bond 30 años
        ^TWO  = T-Note 2 años

    Cualquier otro ticker de Yahoo → se asume precio.
    Series FRED y Banxico siempre se pasan con asset_type explícito.
    """
    tasas_yf = {'^IRX', '^FVX', '^TNX', '^TYX', '^TWO'}
    if ticker.upper() in tasas_yf:
        return ASSET_TYPE_RATE
    return ASSET_TYPE_PRICE


# =============================================================================
# SECCIÓN 2 — FUENTE: YAHOO FINANCE (activos)
# =============================================================================

def load_yf(ticker: str, asset_type: str = None,
            from_date: str = None, to_date: str = None,
            log_returns: bool = False) -> pd.DataFrame:
    """
    Descarga datos desde Yahoo Finance y devuelve un DataFrame normalizado.

    Columnas de salida:
        Fecha      : datetime normalizado (sin hora)
        Cierre     : precio de cierre ajustado  (asset_type='price')
        Tasa       : tasa en % anual             (asset_type='rate')
        return_<ticker>  : rendimiento simple o log  (si price)
        dif_bps_<ticker> : diferencia en bps         (si rate)

    Parámetros
    ----------
    ticker     : str  — ticker de Yahoo Finance
    asset_type : str  — 'price' | 'rate' | None (autodetecta)
    from_date  : str  — 'YYYY-MM-DD'
    to_date    : str  — 'YYYY-MM-DD'
    log_returns: bool — True = log-returns, False = returns simples
    """
    if yf is None:
        raise ImportError("yfinance no está instalado. pip install yfinance")

    if asset_type is None:
        asset_type = infer_asset_type(ticker)

    raw = yf.download(
        tickers=ticker,
        start=from_date,
        end=to_date,
        auto_adjust=True,
        progress=False
    )

    if raw.empty:
        raise ValueError(f"Yahoo Finance no devolvió datos para '{ticker}'")

    # Manejar MultiIndex defensivamente
    if isinstance(raw.columns, pd.MultiIndex):
        cierre_raw = raw['Close'][ticker]
    else:
        cierre_raw = raw['Close']

    col_nombre = 'Cierre' if asset_type == ASSET_TYPE_PRICE else 'Tasa'

    t = pd.DataFrame()
    t['Fecha']    = pd.to_datetime(cierre_raw.index).normalize()
    t[col_nombre] = cierre_raw.values
    t = t.dropna(subset=[col_nombre]).sort_values('Fecha').reset_index(drop=True)

    # Filtrado de fechas
    if from_date:
        t = t[t['Fecha'] >= pd.to_datetime(from_date)]
    if to_date:
        t = t[t['Fecha'] <= pd.to_datetime(to_date)]

    # Transformaciones según asset_type
    t = _apply_transformations(t, col_nombre, ticker, asset_type, log_returns)

    return t


# =============================================================================
# SECCIÓN 3 — FUENTE: FRED / FED (tasas EUA)
# =============================================================================
#
# Series útiles de FRED:
#   DGS2      = Treasury 2 años (rendimiento constante, % anual)
#   DGS5      = Treasury 5 años
#   DGS10     = Treasury 10 años
#   DGS30     = Treasury 30 años
#   DFEDTARU  = Fed Funds Rate (límite superior, target)
#   T10YIE    = Breakeven inflación 10 años (TIPS spread)
#   T5YIE     = Breakeven inflación 5 años
#   BAMLH0A0HYM2  = High Yield spread OAS
#
# Formato de la API:
#   https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS5&vintage_date=...
#   No requiere API key para el endpoint CSV público. Para el endpoint JSON
#   (api.stlouisfed.org) sí se necesita key, pero el CSV es suficiente.

FRED_BASE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def load_fred(serie_id: str, asset_type: str = ASSET_TYPE_RATE,
              from_date: str = None, to_date: str = None) -> pd.DataFrame:
    """
    Descarga una serie del FRED (FED) sin API key.

    La API CSV pública de FRED devuelve datos diarios (días hábiles).
    Los valores son tasas en % anual (ej. 4.25 = 4.25% anual).
    Los puntos con '.' (sin dato) se eliminan automáticamente.

    Columnas de salida:
        Fecha          : datetime
        Tasa           : tasa en % anual
        dif_bps_<id>   : diferencia en bps período a período

    Parámetros
    ----------
    serie_id   : str  — ID de la serie FRED (ej. 'DGS5', 'DGS10', 'T10YIE')
    asset_type : str  — casi siempre 'rate'; usar 'pct' para breakevens o spreads
    from_date  : str  — 'YYYY-MM-DD'
    to_date    : str  — 'YYYY-MM-DD'

    Ejemplo
    -------
    >>> t5 = load_fred('DGS5', from_date='2020-01-01', to_date='2024-12-31')
    >>> t10 = load_fred('DGS10')
    """
    params = {'id': serie_id}
    response = requests.get(FRED_BASE_URL, params=params, timeout=30)
    response.raise_for_status()

    from io import StringIO
    raw = pd.read_csv(StringIO(response.text))

    # FRED usa 'DATE' y el nombre de la serie como columnas
    # Renombrar consistentemente
    raw.columns = ['Fecha', 'Tasa']

    # Eliminar puntos (días sin dato en FRED)
    raw = raw[raw['Tasa'] != '.'].copy()
    raw['Fecha'] = pd.to_datetime(raw['Fecha'])
    raw['Tasa']  = raw['Tasa'].astype(float)
    raw = raw.sort_values('Fecha').reset_index(drop=True)

    # Filtrado de fechas
    if from_date:
        raw = raw[raw['Fecha'] >= pd.to_datetime(from_date)]
    if to_date:
        raw = raw[raw['Fecha'] <= pd.to_datetime(to_date)]

    raw = raw.reset_index(drop=True)

    # Transformaciones
    raw = _apply_transformations(raw, 'Tasa', serie_id, asset_type, log_returns=False)

    return raw


# =============================================================================
# SECCIÓN 4 — FUENTE: BANXICO SIE (tasas México)
# =============================================================================
#
# Series útiles del SIE de Banxico:
#   SF60633  = CETES 28 días (tasa de rendimiento anual, %)
#   SF60648  = CETES 91 días
#   SF60649  = CETES 182 días
#   SF60650  = CETES 364 días       ← referencia para RF anual en MXN
#   SF43936  = Tasa de referencia (objetivo de la política monetaria)
#   SF61745  = TIIE 28 días
#   SF282    = Tipo de cambio FIX (USD/MXN) — este es precio, no tasa
#
# Bonos M y Udibonos:
#   NOTA IMPORTANTE: Los rendimientos de Bonos M y Udibonos NO tienen
#   una serie limpia y lista en el SIE de Banxico con el mismo patrón
#   que CETES. Para obtenerlos hay dos caminos:
#
#   (a) Vectores de precios (serie SF...): Banxico publica algunas series
#       de rendimiento por plazo (ej. SF46407 para el Bono M a 10 años),
#       pero la cobertura histórica y frecuencia pueden variar.
#       → Buscar en https://www.banxico.org.mx/SieAPIRest/service/v1/series/
#         y localizar el ID correcto para el plazo que necesitas.
#
#   (b) Curva de rendimientos (Valmer/PiP): La fuente más completa para
#       Bonos M y Udibonos por plazo es el vector de precios de Valmer
#       (ahora parte de Grupo BMV / PiP). Requiere suscripción.
#
#   (c) Yahoo Finance: Algunos ETFs de deuda mexicana (ej. MBONO.MX) están
#       disponibles como proxies, pero NO son el rendimiento puro del bono.
#
#   → Si encuentras el ID de serie en Banxico para el Bono M que te interesa,
#     simplemente úsalo con load_banxico(serie_id=..., TOKEN=...) y el código
#     funciona igual que con CETES. El único cambio es que la frecuencia puede
#     ser semanal o mensual en lugar de diaria.
#
# UDIBONOS:
#   El mismo comentario aplica. Además, el rendimiento de Udibonos está en
#   términos reales (sobre UDIs), por lo que la interpretación del dif_bps
#   es diferente: es el cambio en la tasa real, no nominal.
#   → Tratar igual que CETES: asset_type='rate', dif_bps.

BANXICO_BASE_URL = "https://www.banxico.org.mx/SieAPIRest/service/v1/series"

# CETES 28 días: ID de la serie en el SIE de Banxico
CETES_28D  = "SF60633"
CETES_91D  = "SF60648"
CETES_182D = "SF60649"
CETES_364D = "SF60650"    # Tasa anual de referencia habitual para Rf en MXN


def load_banxico(serie_id: str, token: str,
                 asset_type: str = ASSET_TYPE_RATE,
                 from_date: str = None, to_date: str = None,
                 anual: bool = True) -> pd.DataFrame:
    """
    Descarga una serie del SIE de Banxico vía API REST (requiere token).

    Los datos vienen en % anual tal como los publica Banxico.
    El campo 'dato' puede contener 'N/E' (no especificado) → se elimina.

    Frecuencia típica:
        CETES: diaria (pero en la práctica se publica cada martes en subasta)
        Tasa de referencia: diaria (días en que hubo decisión)
        TIIE: diaria

    Parámetros
    ----------
    serie_id   : str  — ID de serie del SIE (ej. 'SF60633')
    token      : str  — token de autenticación del SIE de Banxico
    asset_type : str  — 'rate' para tasas, 'pct' para indicadores en %
    from_date  : str  — 'YYYY-MM-DD'
    to_date    : str  — 'YYYY-MM-DD'
    anual      : bool — True verifica que los datos ya vienen en % anual
                        (solo documenta, no transforma; Banxico siempre los
                        publica anualizados para CETES y tasas de referencia)

    Ejemplo
    -------
    >>> TOKEN = "tu_token_aqui"
    >>> cetes = load_banxico(CETES_28D, TOKEN, from_date='2020-01-01')
    >>> cetes364 = load_banxico(CETES_364D, TOKEN, from_date='2022-01-01')

    Nota sobre frecuencia semanal / CAPM:
        En el script CAPM_theory.py se filtraron los martes porque CETES
        se subasta los martes, necesario para el merge con precios.
        Aquí NO se fuerza ese filtro: la función devuelve todos los datos
        disponibles y el usuario decide la frecuencia al sincronizar series.
        Para el modelo general de regresión, el merge inner join elimina
        automáticamente los días que no coinciden entre ambas series.
    """
    # Construir rango de fechas para la URL
    if from_date and to_date:
        rango = f"{from_date}/{to_date}"
    elif from_date:
        rango = f"{from_date}/2099-12-31"
    elif to_date:
        rango = f"1990-01-01/{to_date}"
    else:
        rango = "1990-01-01/2099-12-31"

    url = f"{BANXICO_BASE_URL}/{serie_id}/datos/{rango}"
    headers = {"Bmx-Token": token}

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    datos_json = response.json()["bmx"]["series"][0]["datos"]
    raw = pd.DataFrame(datos_json)

    # Limpiar y convertir
    raw = raw[raw["dato"] != "N/E"].copy()
    raw["Fecha"] = pd.to_datetime(raw["fecha"], format="%d/%m/%Y")
    raw["Tasa"]  = pd.to_numeric(raw["dato"], errors="coerce")
    raw = raw[["Fecha", "Tasa"]].dropna().sort_values("Fecha").reset_index(drop=True)

    # Filtrado de fechas (redundante pero defensivo)
    if from_date:
        raw = raw[raw["Fecha"] >= pd.to_datetime(from_date)]
    if to_date:
        raw = raw[raw["Fecha"] <= pd.to_datetime(to_date)]

    raw = raw.reset_index(drop=True)

    # Transformaciones
    raw = _apply_transformations(raw, 'Tasa', serie_id, asset_type, log_returns=False)

    return raw


# =============================================================================
# SECCIÓN 5 — TRANSFORMACIONES CENTRALIZADAS
# =============================================================================
#
# LÓGICA CENTRAL UNIFICADA:
#   · asset_type='price' → return simple o log (columna: return_<name>)
#   · asset_type='rate' o 'pct' → diferencia en bps (columna: dif_bps_<name>)
#
# Esta función es el único lugar donde se toma esa decisión.
# Todas las funciones de carga (load_yf, load_fred, load_banxico) la llaman.
# La clase model también la respeta al sincronizar.

def _apply_transformations(df: pd.DataFrame, col: str, name: str,
                            asset_type: str, log_returns: bool = False) -> pd.DataFrame:
    """
    Aplica la transformación correcta según asset_type y agrega la columna derivada.

    Columna de entrada : col  (nombre en df, ej. 'Cierre' o 'Tasa')
    Columna de salida  :
        'price'        → return_<name>   (en decimal, ej. 0.012 = 1.2%)
        'rate' / 'pct' → dif_bps_<name> (en bps, ej. 25.0 = +25 bps)

    Nota sobre unidades:
        FRED y Banxico publican tasas en % (ej. 4.25 = 4.25% anual).
        La diferencia en bps = (Tasa_t - Tasa_{t-1}) * 100
        Ej: de 4.25% a 4.50% → (4.50 - 4.25)*100 = 25 bps ✓
    """
    df = df.copy()

    if asset_type == ASSET_TYPE_PRICE:
        prev = df[col].shift(1)
        if log_returns:
            df[f'return_{name}'] = np.log(df[col] / prev)
        else:
            df[f'return_{name}'] = df[col] / prev - 1

    elif asset_type in (ASSET_TYPE_RATE, ASSET_TYPE_PCT):
        prev = df[col].shift(1)
        # bps = (nivel_actual - nivel_anterior) * 100
        # Funciona correctamente para tasas en % anual (ej. CETES, DGS5)
        # También funciona para indicadores en % (CPI YoY, PMI en %, etc.)
        df[f'dif_bps_{name}'] = (df[col] - prev) * 100

    df = df.dropna().reset_index(drop=True)
    return df


# =============================================================================
# SECCIÓN 6 — SINCRONIZACIÓN DE DOS SERIES
# =============================================================================

def synchronise(info1, info2,
                asset_type1: str = None, asset_type2: str = None,
                name1: str = None, name2: str = None,
                from_date: str = None, to_date: str = None,
                log_returns: bool = False,
                model_type: str = 'macro',
                dif: bool = False, suma: bool = False) -> pd.DataFrame:
    """
    Sincroniza dos series de tiempo en un solo DataFrame por fecha (inner join).

    Acepta series ya cargadas (DataFrame) o tickers/IDs para descargar en el acto.

    Parámetros
    ----------
    info1, info2 : str | pd.DataFrame
        · str   → ticker de Yahoo Finance (se descarga automáticamente)
        · DataFrame → ya cargado con load_yf / load_fred / load_banxico
    asset_type1/2 : str
        · 'price' | 'rate' | 'pct' | None (autodetecta si str)
    name1, name2 : str
        · Etiquetas para las columnas en el DataFrame sincronizado
    model_type   : str
        · 'macro'      → usa returns o dif_bps según asset_type
        · 'statistics' → ídem + estandariza ambas series (Z-score)
        · 'learning'   → usa la columna raw (Cierre o Tasa) sin transformar

    Ejemplo
    -------
    # Precio vs precio (ambos Yahoo)
    >>> ts = synchronise('^GSPC', 'AAPL', from_date='2022-01-01')

    # Precio vs tasa (Yahoo + FRED)
    >>> spx = load_yf('^GSPC', from_date='2020-01-01')
    >>> dgs5 = load_fred('DGS5', from_date='2020-01-01')
    >>> ts = synchronise(spx, dgs5, name1='SPX', name2='DGS5')

    # Precio vs CETES (Yahoo + Banxico)
    >>> ipc = load_yf('^MXX', from_date='2022-01-01')
    >>> cetes = load_banxico(CETES_28D, TOKEN, from_date='2022-01-01')
    >>> ts = synchronise(ipc, cetes, name1='IPC', name2='CETES28')
    """
    # ── Cargar si son strings ─────────────────────────────────────────────────
    if isinstance(info1, str):
        at1 = asset_type1 or infer_asset_type(info1)
        ts1 = load_yf(info1, asset_type=at1, from_date=from_date,
                      to_date=to_date, log_returns=log_returns)
        name1 = name1 or info1
    else:
        ts1 = info1.copy()
        at1 = asset_type1 or ASSET_TYPE_PRICE
        name1 = name1 or 'serie_x'

    if isinstance(info2, str):
        at2 = asset_type2 or infer_asset_type(info2)
        ts2 = load_yf(info2, asset_type=at2, from_date=from_date,
                      to_date=to_date, log_returns=log_returns)
        name2 = name2 or info2
    else:
        ts2 = info2.copy()
        at2 = asset_type2 or ASSET_TYPE_PRICE
        name2 = name2 or 'serie_y'

    # ── Normalizar fechas ─────────────────────────────────────────────────────
    ts1['Fecha'] = pd.to_datetime(ts1['Fecha']).dt.normalize()
    ts2['Fecha'] = pd.to_datetime(ts2['Fecha']).dt.normalize()

    # ── Seleccionar columna derivada según model_type y asset_type ────────────
    def _pick_derived_col(ts, at, nm, model_type):
        if model_type == 'learning':
            # Devuelve la columna raw
            return 'Cierre' if at == ASSET_TYPE_PRICE else 'Tasa'
        # Para macro y statistics: usa la columna transformada
        if at == ASSET_TYPE_PRICE:
            return f'return_{nm}'
        else:  # rate o pct
            return f'dif_bps_{nm}'

    col1 = _pick_derived_col(ts1, at1, name1, model_type)
    col2 = _pick_derived_col(ts2, at2, name2, model_type)

    # Verificar que las columnas existen
    if col1 not in ts1.columns:
        raise KeyError(f"Columna '{col1}' no encontrada en la serie 1. "
                       f"Columnas disponibles: {list(ts1.columns)}")
    if col2 not in ts2.columns:
        raise KeyError(f"Columna '{col2}' no encontrada en la serie 2. "
                       f"Columnas disponibles: {list(ts2.columns)}")

    # ── Reducir y hacer merge ─────────────────────────────────────────────────
    x_slim = ts1[['Fecha', col1]].rename(columns={col1: name1})
    y_slim = ts2[['Fecha', col2]].rename(columns={col2: name2})

    merged = pd.merge(x_slim, y_slim, on='Fecha', how='inner').dropna()
    merged = merged.sort_values('Fecha').reset_index(drop=True)

    # ── Filtrado adicional de fechas ──────────────────────────────────────────
    if from_date:
        merged = merged[merged['Fecha'] >= pd.to_datetime(from_date)]
    if to_date:
        merged = merged[merged['Fecha'] <= pd.to_datetime(to_date)]
    merged = merged.reset_index(drop=True)

    # ── Columnas opcionales ───────────────────────────────────────────────────
    if dif:
        merged['Diferencia'] = merged[name1] - merged[name2]
    if suma:
        merged['Suma'] = merged[name1] + merged[name2]

    return merged


# =============================================================================
# SECCIÓN 7 — CLASE MODEL UNIFICADA
# =============================================================================

class model:
    """
    Clase de regresión lineal simple con fuentes de datos unificadas.

    Parámetros
    ----------
    security_x   : str | pd.DataFrame
        · str     → ticker Yahoo Finance (se descarga solo)
        · DataFrame → pre-cargado con load_yf / load_fred / load_banxico
    security_y   : str | pd.DataFrame
    asset_type_x : str  — 'price' | 'rate' | 'pct'
    asset_type_y : str  — 'price' | 'rate' | 'pct'
    name_x, name_y : str — etiqueta para el eje / columna
    decimals     : int
    model_type   : str  — 'macro' | 'statistics' | 'learning'

    Ejemplo rápido
    --------------
    # SPX vs Apple (ambos precios)
    >>> m = model('^GSPC', 'AAPL', model_type='macro')
    >>> m.synchronise_timeseries('2022-01-01', '2024-12-31')
    >>> m.compute_linear_reg()
    >>> m.plot_linear_reg()

    # IPC vs CETES (precio vs tasa)
    >>> TOKEN = "tu_token_banxico"
    >>> ipc   = load_yf('^MXX', asset_type='price', from_date='2022-01-01')
    >>> cetes = load_banxico(CETES_28D, TOKEN, from_date='2022-01-01')
    >>> m = model(ipc, cetes, asset_type_x='price', asset_type_y='rate',
    ...           name_x='IPC', name_y='CETES28', model_type='macro')
    >>> m.synchronise_timeseries('2022-01-01', '2024-12-31')
    >>> m.compute_linear_reg()

    # DXY vs DGS5 (ambas tasas/macro)
    >>> dxy  = load_yf('DX-Y.NYB', asset_type='price', from_date='2020-01-01')
    >>> dgs5 = load_fred('DGS5', from_date='2020-01-01')
    >>> m = model(dxy, dgs5, asset_type_x='price', asset_type_y='rate',
    ...           name_x='DXY', name_y='DGS5', model_type='macro')
    """

    def __init__(self, security_x, security_y,
                 asset_type_x: str = None, asset_type_y: str = None,
                 name_x: str = None, name_y: str = None,
                 decimals: int = 5, model_type: str = 'macro'):

        self.security_x   = security_x
        self.security_y   = security_y
        self.asset_type_x = asset_type_x
        self.asset_type_y = asset_type_y
        self.name_x = name_x or (security_x if isinstance(security_x, str) else 'serie_x')
        self.name_y = name_y or (security_y if isinstance(security_y, str) else 'serie_y')
        self.decimals     = decimals
        self.model_type   = model_type

        # Atributos que se llenan después
        self.timeseries        = None
        self.x                 = None
        self.y                 = None
        self.std_x             = None
        self.std_y             = None
        self.beta              = None
        self.alpha             = None
        self.p_value           = None
        self.correlation       = None
        self.r_squared         = None
        self.hypothesis_null   = None
        self.predictor_linreg  = None
        self.residuals         = None
        self.n                 = None
        self.y_mean            = None
        self.x_mean            = None
        self.sxx               = None
        self.sxy               = None
        self.syy               = None
        self.MCO               = None
        self.mco_model_variance= None
        self.b0_variance       = None
        self.b1_variance       = None
        self.b0_interval       = None
        self.b1_interval       = None
        self.df                = None
        self.suma_cuadrados_totales      = None
        self.suma_cuadrados_regresion    = None
        self.suma_cuadrados_errores      = None
        self.cuadrados_medios_regresion  = None
        self.cuadrados_medios_errores    = None
        self.cuadrados_medios_totales    = None
        self.F_test                      = None

    def synchronise_timeseries(self, from_date: str = None, to_date: str = None,
                                log_returns: bool = False):
        """
        Sincroniza las dos series y prepara self.timeseries para la regresión.

        Si security_x / security_y son strings → descarga desde Yahoo Finance.
        Si son DataFrames pre-cargados (con load_fred / load_banxico) → usa directo.

        El asset_type de cada serie determina si se usan returns o dif_bps.
        En model_type='statistics' se estandariza adicionalmente (Z-score).
        """
        # Auto-detectar asset_type si no fue especificado
        if self.asset_type_x is None:
            if isinstance(self.security_x, str):
                self.asset_type_x = infer_asset_type(self.security_x)
            else:
                # Si ya viene como DataFrame, intentar inferir de columnas
                cols = list(self.security_x.columns)
                self.asset_type_x = ASSET_TYPE_RATE if 'Tasa' in cols else ASSET_TYPE_PRICE

        if self.asset_type_y is None:
            if isinstance(self.security_y, str):
                self.asset_type_y = infer_asset_type(self.security_y)
            else:
                cols = list(self.security_y.columns)
                self.asset_type_y = ASSET_TYPE_RATE if 'Tasa' in cols else ASSET_TYPE_PRICE

        self.timeseries = synchronise(
            self.security_x, self.security_y,
            asset_type1  = self.asset_type_x,
            asset_type2  = self.asset_type_y,
            name1        = self.name_x,
            name2        = self.name_y,
            from_date    = from_date,
            to_date      = to_date,
            log_returns  = log_returns,
            model_type   = self.model_type
        )

        # Estandarización para model_type='statistics'
        if self.model_type == 'statistics':
            mean_x = self.timeseries[self.name_x].mean()
            self.std_x = self.timeseries[self.name_x].std()
            self.timeseries[self.name_x] = (
                self.timeseries[self.name_x] - mean_x) / self.std_x

            mean_y = self.timeseries[self.name_y].mean()
            self.std_y = self.timeseries[self.name_y].std()
            self.timeseries[self.name_y] = (
                self.timeseries[self.name_y] - mean_y) / self.std_y
        else:
            self.std_x = self.timeseries[self.name_x].std()
            self.std_y = self.timeseries[self.name_y].std()

        self.n  = len(self.timeseries)
        self.df = self.n - 2

        if self.timeseries.empty:
            print(f'⚠ No hay datos para {self.name_x} y {self.name_y}')

    # ── Regresión lineal ──────────────────────────────────────────────────────

    def compute_linear_reg(self):
        """OLS estándar. Calcular después de synchronise_timeseries."""
        self.x = self.timeseries[self.name_x].values
        self.y = self.timeseries[self.name_y].values
        self.y_mean = np.mean(self.y)
        self.x_mean = np.mean(self.x)
        self.sxx = np.sum((self.x - self.x_mean) ** 2)
        self.sxy = np.sum((self.x - self.x_mean) * (self.y - self.y_mean))
        self.syy = np.sum((self.y - self.y_mean) ** 2)

        slope, intercept, r, p, se = st.linregress(x=self.x, y=self.y)
        self.beta           = np.round(slope, self.decimals)
        self.alpha          = np.round(intercept, self.decimals)
        self.p_value        = np.round(p, self.decimals)
        self.correlation    = np.round(r, self.decimals)
        self.r_squared      = np.round(r ** 2, self.decimals)
        self.hypothesis_null = p > 0.05
        self.predictor_linreg = intercept + slope * self.x
        self.residuals      = self.y - self.predictor_linreg
        self.MCO            = np.sum(self.residuals ** 2)
        self.mco_model_variance = self.MCO / (self.n - 2)
        self.b0_variance    = ((1 / self.n) + (self.x_mean ** 2) / self.sxx) * self.mco_model_variance
        self.b1_variance    = self.mco_model_variance / self.sxx

        self.h = (1 / self.n) + ((self.x - self.x_mean) ** 2) / self.sxx
        self.standardized_residuals = self.residuals / np.sqrt(
            self.mco_model_variance * (1 - self.h))

        num = (self.n - 2) * self.mco_model_variance - (self.residuals ** 2) / (1 - self.h)
        sigma2_i = num / (self.n - 3)
        self.studentized_residuals = self.residuals / np.sqrt(sigma2_i * (1 - self.h))

    # ── Gráficas ──────────────────────────────────────────────────────────────

    def plot_linear_reg(self, ax=None):
        created_ax = ax is None
        if created_ax:
            fig, ax = plt.subplots()
    
        str_self = (
            'Linear regression | x: ' + self.name_x
            + ' | y: ' + self.name_y + '\n'
            + 'alpha ' + str(self.alpha)
            + ' | beta (slope) ' + str(self.beta) + '\n'
            + 'p-value ' + str(self.p_value)
            + ' | null-hypothesis ' + str(self.hypothesis_null) + '\n'
            + 'correl (r-value) ' + str(self.correlation)
            + ' | r-squared ' + str(self.r_squared)
        )
    
        str_compacto = (self.name_x + ' vs ' + self.name_y
                        + ' | corr: ' + f"{self.correlation:.3f}"
                        + ' | R²: ' + f"{self.r_squared:.3f}")
    
        title = ('Scatterplot of returns\n' + str_self) if created_ax else str_compacto
    
        ax.set_title(title)
        ax.scatter(self.x, self.y, alpha=0.6)
        ax.plot(self.x, self.predictor_linreg, color='green', label='OLS')
        ax.set_xlabel(self.name_x)
        ax.set_ylabel(self.name_y)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
        if created_ax:
            plt.tight_layout()
            plt.show()

    # ── Tests de supuestos ────────────────────────────────────────────────────

    def model_confidence_intervals(self, significance: float = 0.05):
        t_crit = t.ppf(1 - significance / 2, self.df)
        self.b0_interval = (self.alpha + t_crit * np.sqrt(self.b0_variance),
                            self.alpha - t_crit * np.sqrt(self.b0_variance))
        self.b1_interval = (self.beta  + t_crit * np.sqrt(self.b1_variance),
                            self.beta  - t_crit * np.sqrt(self.b1_variance))

    def normalidad(self, significance: float = 0.05):
        sw = shapiro(self.residuals)
        print("NORMALIDAD (Shapiro-Wilk):")
        print(f"  p-value: {sw.pvalue:.5f}")
        if sw.pvalue < significance:
            print("  → Se RECHAZA H0: residuos NO normales\n")
        else:
            print("  → NO se rechaza H0: residuos normales\n")

    def homocedasticidad(self, significance: float = 0.05, plot: bool = False):
        X = np.column_stack((np.ones(len(self.x)), self.x))
        bp = het_breuschpagan(self.residuals, X)
        print("HOMOCEDASTICIDAD (Breusch-Pagan):")
        print(f"  p-value: {bp[1]:.5f}")
        if bp[1] < significance:
            print("  → Hay heterocedasticidad (varianza NO constante)\n")
        else:
            print("  → Varianza constante\n")
        if plot:
            plt.figure(figsize=(7, 4))
            plt.scatter(self.predictor_linreg, self.studentized_residuals, alpha=0.6)
            plt.axhline(0, color='black', linestyle='--')
            plt.xlabel("Fitted values")
            plt.ylabel("Studentized Residuals")
            plt.title("Homocedasticidad")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

    def independencia(self):
        dw = durbin_watson(self.residuals)
        print("INDEPENDENCIA (Durbin-Watson):")
        print(f"  DW = {dw:.5f}")
        if dw < 1.5:
            print("  → Posible autocorrelación positiva\n")
        elif dw > 2.5:
            print("  → Posible autocorrelación negativa\n")
        else:
            print("  → Sin evidencia de autocorrelación\n")

    def anova_test(self):
        self.suma_cuadrados_totales    = np.sum((self.y - self.y_mean) ** 2)
        self.suma_cuadrados_regresion  = np.sum((self.predictor_linreg - self.y_mean) ** 2)
        self.suma_cuadrados_errores    = np.sum((self.y - self.predictor_linreg) ** 2)
        self.cuadrados_medios_regresion = self.suma_cuadrados_regresion / 1
        self.cuadrados_medios_errores   = self.suma_cuadrados_errores / (self.n - 2)
        self.F_test = self.cuadrados_medios_regresion / self.cuadrados_medios_errores
        k = 1
        p_value = 1 - stats.f.cdf(self.F_test, k, self.n - k - 1)
        print(f"ANOVA — F test: {self.F_test:.5f}  |  p-value: {p_value:.6f}")
        if p_value < 0.05:
            print("  → Se rechaza H0: al menos un coeficiente es significativo\n")
        else:
            print("  → No se rechaza H0: modelo no significativo\n")

    # ── Estimador Bayesiano de Vasicek (1973) ────────────────────────────────

    def vasicek_beta(self, b_prior: float, s2_prior: float, verbose: bool = True) -> dict:
        """
        Estimador Bayesiano de Beta — Vasicek (1973), Ecuaciones 15 y 16.
        Requiere haber corrido compute_linear_reg() primero.
        """
        if self.beta is None or self.b1_variance is None:
            raise ValueError("Corre compute_linear_reg() primero.")

        b_mco   = self.beta
        s2_mco  = self.b1_variance
        h_mco   = 1.0 / s2_mco
        h_prior = 1.0 / s2_prior

        b_vasicek    = (h_prior * b_prior + h_mco * b_mco) / (h_prior + h_mco)
        s2_posterior = 1.0 / (h_prior + h_mco)
        s_posterior  = s2_posterior ** 0.5
        peso_prior   = h_prior / (h_prior + h_mco)
        peso_muestra = h_mco   / (h_prior + h_mco)
        ajuste_abs   = b_vasicek - b_mco

        resultado = {
            'benchmark':      self.name_x,
            'security':       self.name_y,
            'n_obs':          self.n,
            'b_mco':          round(b_mco, 6),
            'b_prior':        round(b_prior, 6),
            'b_vasicek':      round(b_vasicek, 6),
            's2_mco':         round(s2_mco, 8),
            's2_prior':       round(s2_prior, 8),
            's2_posterior':   round(s2_posterior, 8),
            's_posterior':    round(s_posterior, 6),
            'peso_prior_pct': round(peso_prior * 100, 2),
            'peso_mco_pct':   round(peso_muestra * 100, 2),
            'ajuste_abs':     round(ajuste_abs, 6),
        }

        if verbose:
            print("\n" + "═" * 60)
            print(f"  VASICEK (1973) — {self.name_x} → {self.name_y}  N={self.n}")
            print("═" * 60)
            print(f"  b_OLS     = {b_mco:>10.6f}   s²_b = {s2_mco:.8f}")
            print(f"  b_prior   = {b_prior:>10.6f}   s'²  = {s2_prior:.8f}")
            bm = '█' * int(peso_muestra * 40)
            bp = '█' * int(peso_prior   * 40)
            print(f"  Muestra : {bm:<40} {peso_muestra*100:5.1f}%")
            print(f"  Prior   : {bp:<40} {peso_prior*100:5.1f}%")
            print(f"  b_Vasicek = {b_vasicek:>10.6f}   s''² = {s2_posterior:.8f}")
            print(f"  Ajuste vs OLS: {ajuste_abs:+.6f}")
            print("═" * 60)

        return resultado

    # ── Transformaciones de heterocedasticidad ────────────────────────────────

    def yeo_johnson_transform(self, use_standardize: bool = True, plot: bool = False):
        y = self.y.reshape(-1, 1)
        pt = PowerTransformer(method='yeo-johnson', standardize=use_standardize)
        y_t = pt.fit_transform(y)
        self.y_transformed = y_t.flatten()
        self.lambda_yj = pt.lambdas_[0]
        self.yj_model  = pt
        print(f"Lambda Yeo-Johnson: {self.lambda_yj:.4f}")
        if abs(self.lambda_yj - 1) < 0.1:
            print("→ Transformación innecesaria")
        elif abs(self.lambda_yj) < 0.1:
            print("→ Aproximadamente log-like")
        elif self.lambda_yj < 0:
            print("→ Compresión fuerte (colas pesadas)")
        else:
            print("→ Transformación moderada")
        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].hist(self.y, bins=30)
            axes[0].set_title("Y original")
            axes[1].hist(self.y_transformed, bins=30)
            axes[1].set_title("Y transformada (Yeo-Johnson)")
            plt.tight_layout()
            plt.show()
        return self.y_transformed

    def run_wls_pipeline(self, method: str = 'reg_aux',
                         aux_type: str = 'log',
                         use_fitted: bool = True,
                         groups: int = 5):
        """WLS pipeline completo (reg_aux o grouping). Requiere compute_linear_reg()."""
        if self.residuals is None:
            raise ValueError("Corre compute_linear_reg() primero.")

        x_var = self.predictor_linreg if use_fitted else self.x
        e = self.residuals

        if method == 'reg_aux':
            if aux_type == 'log':
                y_aux = np.log(e ** 2 + 1e-8)
            elif aux_type == 'abs':
                y_aux = np.abs(e)
            elif aux_type == 'squared':
                y_aux = e ** 2
            else:
                raise ValueError("aux_type debe ser 'log', 'abs' o 'squared'")
            X = np.column_stack((np.ones(len(x_var)), x_var))
            beta_aux = np.linalg.inv(X.T @ X) @ (X.T @ y_aux)
            y_hat_aux = X @ beta_aux
            if aux_type == 'log':
                gx = np.exp(y_hat_aux)
            elif aux_type == 'abs':
                gx = y_hat_aux ** 2
            else:
                gx = np.maximum(y_hat_aux, 1e-8)

        elif method == 'grouping':
            sorted_idx = np.argsort(x_var)
            e_sorted = e[sorted_idx]
            n = len(e)
            group_size = n // groups
            gx = np.zeros(n)
            for i in range(groups):
                start = i * group_size
                end   = (i + 1) * group_size if i < groups - 1 else n
                idx   = sorted_idx[start:end]
                gx[idx] = np.var(e[idx], ddof=1)
        else:
            raise ValueError("method debe ser 'reg_aux' o 'grouping'")

        self.gx = gx
        self.weights = 1 / gx

        W = np.diag(self.weights)
        X = np.column_stack((np.ones(len(self.x)), self.x))
        beta_wls = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ self.y)

        self.alpha_wls = beta_wls[0]
        self.beta_wls  = beta_wls[1]
        self.predictor_wls = self.alpha_wls + self.beta_wls * self.x
        self.residuals_wls = self.y - self.predictor_wls

        # Consolidar WLS como modelo activo
        self.alpha = self.alpha_wls
        self.beta  = self.beta_wls
        self.predictor_linreg = self.predictor_wls
        self.residuals = self.residuals_wls
        self.mco_model_variance = np.sum(self.residuals ** 2) / (self.n - 2)
        print("--- WLS MODEL ACTIVO ---")
        print(f"Alpha: {self.alpha:.6f} | Beta: {self.beta:.6f}")


# =============================================================================
# SECCIÓN 8 — HELPERS ADICIONALES
# =============================================================================

def promedio_mensual(df: pd.DataFrame, columna_interes: str) -> pd.DataFrame:
    """Agrega a frecuencia mensual calculando el promedio."""
    df = df.copy()
    df['YearMonth'] = pd.to_datetime(df['Fecha']).dt.to_period('M')
    df_mensual = (
        df.groupby('YearMonth')
        .agg({columna_interes: 'mean'})
        .reset_index()
    )
    df_mensual['Fecha'] = df_mensual['YearMonth'].dt.to_timestamp()
    return df_mensual.sort_values('Fecha').reset_index(drop=True)


def plot_timeseries(df: pd.DataFrame, col1: str, col2: str,
                    secondary_y: bool = True, titulo: str = 'Timeseries'):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(titulo)
    ax.plot(df['Fecha'], df[col1], color='steelblue', label=col1)
    ax.set_ylabel(col1, color='steelblue')
    if secondary_y:
        ax2 = ax.twinx()
        ax2.plot(df['Fecha'], df[col2], color='tomato', label=col2)
        ax2.set_ylabel(col2, color='tomato')
    else:
        ax.plot(df['Fecha'], df[col2], color='tomato', label=col2)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def cargar_archivo(df: pd.DataFrame, archivo: str, output_dir: str = None):
    """Guarda un DataFrame como CSV."""
    if output_dir:
        path = Path(output_dir) / f"{archivo}.csv"
    else:
        base_path = Path(__file__).resolve().parent
        path = base_path / f"{archivo}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8')
    print(f"Guardado en: {path}")


# =============================================================================
# SECCIÓN 9 — PIPELINE VASICEK (función de nivel superior)
# =============================================================================

def estimar_prior_desde_periodo(bench, sec, from_date_prior, to_date_prior,
                                 asset_type_x=None, asset_type_y=None,
                                 verbose=True):
    """
    Estima beta prior y su varianza corriendo OLS en un período histórico.
    Acepta strings (tickers Yahoo) o DataFrames pre-cargados.
    """
    m_prior = model(bench, sec,
                    asset_type_x=asset_type_x,
                    asset_type_y=asset_type_y,
                    model_type='macro')
    m_prior.synchronise_timeseries(from_date_prior, to_date_prior)

    if len(m_prior.timeseries) < 5:
        print(f"⚠  Muestra prior insuficiente: {len(m_prior.timeseries)} obs")
        return None, None, None, None

    m_prior.compute_linear_reg()

    if verbose:
        print(f"  Prior: {from_date_prior} → {to_date_prior} | "
              f"b'={m_prior.beta:.6f} | s'²={m_prior.b1_variance:.8f} | "
              f"R²={m_prior.r_squared:.4f} | N={m_prior.n}")

    return m_prior.beta, m_prior.b1_variance, m_prior.r_squared, m_prior.n


def vasicek_pipeline(benchmarks, securities,
                     fecha_quiebre: str,
                     ventana_prior_dias: int = 40,
                     from_date_post: str = None,
                     to_date_post: str = None,
                     s2_prior_difusa: float = 1.0,
                     asset_type_x: str = None,
                     asset_type_y: str = None) -> pd.DataFrame:
    """
    Pipeline completo de estimación Bayesiana de Vasicek (1973).
    Ahora acepta asset_type explícito para benchmarks y securities.
    """
    from datetime import datetime
    fecha_q   = pd.to_datetime(fecha_quiebre)
    fecha_pre = fecha_q - pd.Timedelta(days=ventana_prior_dias)
    from_post = from_date_post or fecha_quiebre
    to_post   = to_date_post or datetime.today().strftime('%Y-%m-%d')

    print("\n" + "█" * 65)
    print(f"{'PIPELINE VASICEK (1973)':^65}")
    print(f"{'Quiebre: ' + fecha_quiebre:^65}")
    print(f"{'Prior:   ' + fecha_pre.strftime('%Y-%m-%d') + ' → ' + fecha_quiebre:^65}")
    print(f"{'Post:    ' + from_post + ' → ' + to_post:^65}")
    print("█" * 65)

    todos = []

    for sec in securities:
        for bench in benchmarks:
            if isinstance(bench, str) and isinstance(sec, str) and bench == sec:
                continue

            name_b = bench if isinstance(bench, str) else 'benchmark'
            name_s = sec   if isinstance(sec, str)   else 'security'
            print(f"\n{'─'*55}\n  PAR: {name_b} → {name_s}\n{'─'*55}")

            # Prior
            b_prior, s2_prior, r2_prior, n_prior = estimar_prior_desde_periodo(
                bench, sec,
                fecha_pre.strftime('%Y-%m-%d'), fecha_quiebre,
                asset_type_x=asset_type_x, asset_type_y=asset_type_y
            )

            if b_prior is None or n_prior < 5:
                b_prior, s2_prior = 0.0, s2_prior_difusa
            elif r2_prior < 0.05:
                s2_prior *= 3.0

            # Post
            m_post = model(bench, sec,
                           asset_type_x=asset_type_x,
                           asset_type_y=asset_type_y,
                           name_x=name_b, name_y=name_s,
                           model_type='macro')
            m_post.synchronise_timeseries(from_post, to_post)

            if len(m_post.timeseries) < 5:
                print(f"  ❌ Muestra insuficiente ({len(m_post.timeseries)} obs). Saltando.")
                continue

            m_post.compute_linear_reg()
            res = m_post.vasicek_beta(b_prior=b_prior, s2_prior=s2_prior)

            cambio_pct = abs(res['b_vasicek'] - res['b_mco']) / (abs(res['b_mco']) + 1e-10) * 100
            res.update({
                'b_prior_usado': b_prior, 's2_prior_usado': s2_prior,
                'r2_prior': r2_prior, 'n_prior': n_prior,
                'r2_post': m_post.r_squared, 'n_post': m_post.n,
                'cambio_vs_ols_pct': round(cambio_pct, 2)
            })
            todos.append(res)

    if todos:
        df_res = pd.DataFrame(todos)
        print("\n" + "═" * 65)
        print(f"{'RESUMEN VASICEK':^65}")
        print("═" * 65)
        for _, r in df_res.iterrows():
            dom = ("📊 datos" if r['peso_mco_pct'] > 60
                   else "📚 prior" if r['peso_prior_pct'] > 60
                   else "⚖  balance")
            print(f"  {r['benchmark']:<15} → {r['security']:<10} "
                  f"N={r['n_post']} R²={r['r2_post']:.3f}")
            print(f"  b_OLS={r['b_mco']:>9.5f} | b_prior={r['b_prior_usado']:>9.5f} | "
                  f"b_Vasicek={r['b_vasicek']:>9.5f} | {dom}")
        print("═" * 65)
        return df_res

    return None


# =============================================================================
# EJEMPLO DE USO
# =============================================================================

if __name__ == "__main__":

    # ── Ejemplo 1: Precio vs Precio (Yahoo) ──────────────────────────────────
    m1 = model('^GSPC', 'AAPL',
               asset_type_x='price', asset_type_y='price',
               name_x='SP500', name_y='AAPL',
               model_type='macro')
    m1.synchronise_timeseries('2022-01-01', '2024-12-31')
    m1.compute_linear_reg()
    print(f"SP500 vs AAPL | β={m1.beta} | R²={m1.r_squared}")

    # ── Ejemplo 2: IPC vs CETES (precio vs tasa) ─────────────────────────────
    # TOKEN_BANXICO = "tu_token_aqui"
    # cetes = load_banxico(CETES_28D, TOKEN_BANXICO, from_date='2022-01-01')
    # ipc   = load_yf('^MXX', asset_type='price', from_date='2022-01-01')
    # m2 = model(ipc, cetes,
    #            asset_type_x='price', asset_type_y='rate',
    #            name_x='IPC', name_y='CETES28', model_type='macro')
    # m2.synchronise_timeseries()
    # m2.compute_linear_reg()

    # ── Ejemplo 3: SPX vs DGS5 (precio vs tasa FED) ──────────────────────────
    # dgs5 = load_fred('DGS5', from_date='2020-01-01', to_date='2024-12-31')
    # spx  = load_yf('^GSPC', asset_type='price', from_date='2020-01-01')
    # m3 = model(spx, dgs5,
    #            asset_type_x='price', asset_type_y='rate',
    #            name_x='SPX', name_y='DGS5', model_type='macro')
    # m3.synchronise_timeseries()
    # m3.compute_linear_reg()
    # m3.normalidad()
    # m3.homocedasticidad()
    # m3.independencia()
    # m3.plot_linear_reg()
