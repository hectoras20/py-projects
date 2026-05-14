# ============================================================
# CAPM - Security Market Line con IPC y CETES
# ============================================================
# Lo que hace este script:
#   1. Descarga la tasa libre de riesgo (CETES 28 días) desde Banxico
#   2. Descarga precios del IPC y acciones del BMV desde Yahoo Finance
#   3. Alinea todo a frecuencia semanal (martes)
#   4. Calcula betas y rendimientos esperados con el modelo CAPM
#   5. Grafica la Security Market Line (SML) con los activos encima
# ============================================================
 
 
# ------------------------------------------------------------
# 1. LIBRERÍAS
# ------------------------------------------------------------
 
import requests                  # Llamadas HTTP para la API de Banxico (equivale a httr en R)
import yfinance as yf            # Descarga de precios desde Yahoo Finance (equivale a quantmod)
import numpy as np               # Operaciones numéricas (cov, var, mean, log) — equivale a base R
import pandas as pd              # DataFrames y Series indexadas por fecha — equivale a dplyr + xts
import matplotlib.pyplot as plt  # Gráficas (equivale a ggplot2, aunque más imperativo)
import matplotlib.ticker as mticker  # Formateo de ejes (equivale a scales en R)
 
 
# ------------------------------------------------------------
# 2. PARÁMETROS GLOBALES
# ------------------------------------------------------------
 
# Token de autenticación para la API del SIE de Banxico
TOKEN = "ee878065cdc64b207edc54ea9ac5919085e3250705ac0ee569f9f461573cf5ff"
 
fecha_ini = "2018-01-01"   # Inicio del periodo de análisis
fecha_fin = "2025-12-31"   # Fin del periodo de análisis
 
# ID de la serie de CETES 28 días en el SIE de Banxico
idserie = "SF60633"
 
# Ticker del índice de mercado (IPC de la BMV en Yahoo Finance)
ticker_mercado = "^MXX"
 
# Tickers de las acciones individuales a analizar
tickers = [
    "WALMEX.MX",    # Walmart de México
    "FEMSAUBD.MX",  # FEMSA
    "GMEXICOB.MX",  # Grupo México
    "BIMBOA.MX",    # Grupo Bimbo
    "CEMEXCPO.MX",  # CEMEX
    "GFNORTEO.MX",  # Banorte
    "ASURB.MX",     # Grupo Aeroportuario del Sureste
    "KIMBERA.MX"    # Kimberly-Clark de México
]
 
 
# ------------------------------------------------------------
# 3. DESCARGA Y PROCESAMIENTO DE CETES (tasa libre de riesgo)
# ------------------------------------------------------------
 
# Construimos la URL del endpoint del SIE de Banxico para obtener la serie
# El SIE expone una API REST: los datos vienen en JSON
url_banxico = (
    f"https://www.banxico.org.mx/SieAPIRest/service/v1/series/"
    f"{idserie}/datos/{fecha_ini}/{fecha_fin}"
)
 
# Hacemos la petición HTTP GET con el token en los headers de autenticación
# En R esto equivale a getSeriesData() de siebanxicor, que internamente hace lo mismo
respuesta = requests.get(url_banxico, headers={"Bmx-Token": TOKEN})
 
# Extraemos la lista de observaciones desde el JSON anidado
# La estructura es: respuesta → bmx → series → [0] → datos
datos_json = respuesta.json()["bmx"]["series"][0]["datos"]
 
# Convertimos la lista de dicts a un DataFrame de pandas
# Cada elemento tiene: {"fecha": "01/01/2018", "dato": "7.25"}
# En R esto equivale a getSerieDataFrame() que hace la misma conversión
tasas = pd.DataFrame(datos_json)
 
# Transformamos columnas: fecha a datetime, dato a float (tasa anual en decimal)
tasas["date"]     = pd.to_datetime(tasas["fecha"], format="%d/%m/%Y")
tasas["rf_anual"] = pd.to_numeric(tasas["dato"], errors="coerce") / 100
 
# Ordenamos cronológicamente (equivale a dplyr::arrange)
tasas = tasas.sort_values("date").reset_index(drop=True)
 
# Convertimos tasa anual a semanal con capitalización continua: ln(1 + r_anual) / 52
# En R: dplyr::mutate(rf_semanal = log(1 + rf_anual) / 52)
tasas["rf_semanal"] = np.log(1 + tasas["rf_anual"]) / 52
 
# Nos quedamos solo con las columnas necesarias y seteamos date como índice
# En R: dplyr::select(date, rf_semanal)
rf_df = tasas[["date", "rf_semanal"]].set_index("date")
 
 
# ------------------------------------------------------------
# 4. DESCARGA DE PRECIOS (IPC + ACCIONES) DESDE YAHOO FINANCE
# ------------------------------------------------------------
 
# Unimos todos los símbolos en una lista (equivale al vector c() de R)
simbolos = [ticker_mercado] + tickers
 
# yf.download() descarga todos los tickers de una vez y regresa un DataFrame
# con MultiIndex en las columnas: (campo, ticker)
# En R: getSymbols() guarda objetos individuales en el entorno global
datos_yahoo = yf.download(
    tickers   = simbolos,
    start     = fecha_ini,
    end       = fecha_fin,
    auto_adjust = True    # Equivale a usar Ad() en R: precios ajustados por dividendos/splits
)
 
# Extraemos solo la columna "Close" (= precio ajustado porque auto_adjust=True)
# .xs() hace un "cross-section": selecciona un nivel del MultiIndex de columnas
# En R: Ad(get(ticker)) para cada símbolo
precios = datos_yahoo["Close"].copy()
 
# Renombramos el ticker del mercado de "^MXX" a "MERCADO" para mayor claridad
# En R: colnames(precio_mercado) <- "MERCADO"
precios = precios.rename(columns={"^MXX": "MERCADO"})
 
 
# ------------------------------------------------------------
# 5. FILTRAR PRECIOS DE LOS MARTES
# ------------------------------------------------------------
 
# CETES se publica cada martes → filtramos precios del mismo día para poder hacer merge
# .index es el DatetimeIndex del DataFrame; .dayofweek: lunes=0, martes=1, ..., viernes=4
# En R: precios[weekdays(index(precios)) == "Tuesday", ]
precios_martes = precios[precios.index.dayofweek == 1].copy()
 
 
# ------------------------------------------------------------
# 6. CÁLCULO DE RENDIMIENTOS SEMANALES
# ------------------------------------------------------------
 
# .pct_change() calcula rendimiento simple por defecto; usamos np.log para log-rendimientos
# log(P_t / P_{t-1}) = log(P_t) - log(P_{t-1})
# En R: Return.calculate(precios_martes, method = "log")
rend_martes = np.log(precios_martes / precios_martes.shift(1))
 
# Eliminamos la primera fila que queda como NaN (no tiene período anterior)
# En R: na.omit(...)
rend_martes = rend_martes.dropna()
 
 
# ------------------------------------------------------------
# 7. MERGE DE RENDIMIENTOS CON TASA LIBRE DE RIESGO
# ------------------------------------------------------------
 
# Hacemos join "inner" → solo fechas presentes en AMBOS DataFrames
# En R: merge(rend_martes, rf_xts, join = "inner")
base_df = rend_martes.join(rf_df, how="inner")
 
# Desplazamos la tasa libre de riesgo 1 período hacia atrás (usamos la tasa del inicio de la semana)
# .shift(1) en pandas equivale a lag.xts(x, k=1) en R
# Razón: comparamos el rendimiento de la semana con la tasa conocida AL INICIO de esa semana
base_df["rf_lag"] = base_df["rf_semanal"].shift(1)
 
# Eliminamos NAs generados por el shift (primera fila)
base_df = base_df.dropna()
 
 
# ------------------------------------------------------------
# 8. PARÁMETROS DEL MODELO CAPM
# ------------------------------------------------------------
 
R_M = base_df["MERCADO"]   # Rendimiento semanal del mercado (IPC)
r_f = base_df["rf_lag"]    # Tasa libre de riesgo semanal rezagada (CETES)
 
ER_M = R_M.mean()          # Rendimiento promedio del mercado E[R_M]
ER_f = r_f.mean()          # Tasa libre de riesgo promedio E[r_f]
 
# Prima de riesgo del mercado: exceso por encima de la tasa libre de riesgo
prima_mercado = ER_M - ER_f
 
 
# ------------------------------------------------------------
# 9. ESTIMACIÓN DE BETA Y RENDIMIENTOS CAPM POR ACCIÓN
# ------------------------------------------------------------
 
# Construimos la lista de tickers válidos excluyendo columnas auxiliares
# En R: setdiff(colnames(base_xts), c("MERCADO", "rf", "rf_lag"))
tickers_validos = [col for col in base_df.columns if col not in ("MERCADO", "rf_semanal", "rf_lag")]
 
# Lista vacía donde iremos guardando los resultados de cada acción
# En R: lapply() hace esto de forma implícita, aquí lo hacemos explícito con un for
resultados = []
 
for tk in tickers_validos:
 
    R_i = base_df[tk]          # Rendimiento semanal de la acción i
 
    # Excesos sobre la tasa libre de riesgo
    exceso_i = R_i - r_f       # Exceso del activo i sobre r_f
    exceso_m = R_M - r_f       # Exceso del mercado sobre r_f (prima realizada)
 
    # Beta = Cov(R_i - r_f, R_M - r_f) / Var(R_M - r_f)
    # np.cov() regresa una matriz de covarianzas 2x2 → [0,1] es la covarianza cruzada
    # En R: cov(exceso_i, exceso_m) / var(exceso_m)
    cov_matrix = np.cov(exceso_i.values, exceso_m.values)
    beta_i     = cov_matrix[0, 1] / cov_matrix[1, 1]
 
    ER_i_obs  = R_i.mean()                        # Rendimiento observado promedio
    ER_i_capm = ER_f + beta_i * prima_mercado     # Rendimiento TEÓRICO por CAPM: r_f + β*(E[R_M] - r_f)
 
    # Alpha de Jensen (visual): diferencia observado - teórico
    # > 0 → outperformance / < 0 → underperformance respecto al riesgo asumido
    alpha_visual = ER_i_obs - ER_i_capm
 
    # Guardamos los resultados de esta acción como diccionario en la lista
    # En R: data.frame(ticker=tk, beta=beta_i, ...) dentro del lapply
    resultados.append({
        "ticker":                 tk,
        "beta":                   beta_i,
        "rendimiento_observado":  ER_i_obs,
        "rendimiento_capm":       ER_i_capm,
        "alpha_visual":           alpha_visual
    })
 
# Convertimos la lista de dicts a DataFrame (equivale a dplyr::bind_rows en R)
resultados_df = pd.DataFrame(resultados)
print(resultados_df)
 
 
# ------------------------------------------------------------
# 10. CONSTRUCCIÓN DE LA SML (Security Market Line)
# ------------------------------------------------------------
 
# Grilla de betas con margen en ambos extremos
# En R: seq(min - 0.2, max + 0.2, length.out = 200)
beta_grid = np.linspace(
    resultados_df["beta"].min() - 0.2,
    resultados_df["beta"].max() + 0.2,
    200   # 200 puntos para una línea suave
)
 
# Rendimiento teórico de la SML para cada beta de la grilla
rend_sml = ER_f + beta_grid * prima_mercado
 
 
# ------------------------------------------------------------
# 11. GRÁFICA DE LA SML
# ------------------------------------------------------------
 
# En R ggplot trabaja con capas sumadas con +
# En Python matplotlib trabaja de forma imperativa: llamamos métodos sobre el objeto ax
fig, ax = plt.subplots(figsize=(10, 6))
 
# Línea de la SML (teórica)
# Equivale a geom_line(data=sml, aes(x=beta, y=rendimiento))
ax.plot(beta_grid, rend_sml, linewidth=1.5, color="black", label="SML teórica")
 
# Puntos de cada acción (rendimiento OBSERVADO)
# Equivale a geom_point(data=resultados, aes(x=beta, y=rendimiento_observado))
ax.scatter(
    resultados_df["beta"],
    resultados_df["rendimiento_observado"],
    s=60, zorder=5, color="steelblue"
)
 
# Segmentos verticales entre rendimiento CAPM y OBSERVADO (visualiza el alpha)
# Equivale a geom_segment(..., linetype="dashed")
for _, fila in resultados_df.iterrows():
    # iterrows() itera fila por fila del DataFrame — equivale al lapply interno que usaría ggplot
    ax.vlines(
        x       = fila["beta"],
        ymin    = min(fila["rendimiento_capm"], fila["rendimiento_observado"]),
        ymax    = max(fila["rendimiento_capm"], fila["rendimiento_observado"]),
        linestyles = "dashed",
        colors  = "gray",
        linewidth = 0.8
    )
 
# Etiquetas con el nombre del ticker sobre cada punto
# Equivale a geom_text(..., nudge_y=0.0015)
for _, fila in resultados_df.iterrows():
    ax.annotate(
        text    = fila["ticker"],
        xy      = (fila["beta"], fila["rendimiento_observado"]),
        xytext  = (0, 5),             # Desplazamiento en píxeles: 5px arriba (equivale a nudge_y)
        textcoords = "offset points",
        fontsize   = 8
    )
 
# Formatear el eje Y como porcentaje con 2 decimales
# Equivale a scale_y_continuous(labels = percent_format(accuracy = 0.01))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y*100:.2f}%"))
 
# Títulos y etiquetas de ejes
ax.set_title("Security Market Line (SML)", fontsize=13, fontweight="bold")
ax.set_xlabel("β (Beta)", fontsize=11)
ax.set_ylabel("Rendimiento esperado semanal", fontsize=11)
 
# Cuadrícula suave y tema limpio (equivale a theme_minimal en ggplot2)
ax.grid(True, linestyle="--", alpha=0.4)
ax.set_facecolor("white")
 
plt.tight_layout()
plt.show()
 
 
# ------------------------------------------------------------
# 12. TABLA FINAL ORDENADA POR ALPHA
# ------------------------------------------------------------
 
# Ordenar de mayor a menor alpha (equivale a dplyr::arrange(desc(alpha_visual)))
resultados_df = resultados_df.sort_values("alpha_visual", ascending=False).reset_index(drop=True)
print(resultados_df)