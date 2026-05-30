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
 
fecha_ini = "2025-01-28"   # Inicio del periodo de análisis
fecha_fin = "2026-04-29"   # Fin del periodo de análisis
 
# ID de la serie de CETES 28 días en el SIE de Banxico
idserie = "SF60633"
 
# Ticker del índice de mercado (IPC de la BMV en Yahoo Finance)
ticker_mercado = "^SPX"
 
# Tickers de las acciones individuales a analizar
'''tickers = [
    "WALMEX.MX",    # Walmart de México
    "FEMSAUBD.MX",  # FEMSA
    "GMEXICOB.MX",  # Grupo México
    "BIMBOA.MX",    # Grupo Bimbo
    "CEMEXCPO.MX",  # CEMEX
    "GFNORTEO.MX",  # Banorte
    "ASURB.MX",     # Grupo Aeroportuario del Sureste
    "KIMBERA.MX"    # Kimberly-Clark de México
]'''
tickers = ["AAPL.MX"]   
 
 
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
    ###¿Tienes el dato ANUAL y quieres ir a SEMANAS?  ➡️   DIVIDES / 52  (Ej. Ajustar la Rf)
    ###¿Tienes datos SEMANALES y quieres ir a AÑO?    ➡️   MULTIPLICAS * 52 (Ej. Anualizar la acción)
# En R: dplyr::mutate(rf_semanal = log(1 + rf_anual) / 52)
tasas["rf_semanal"] = np.log(1 + tasas["rf_anual"]) / 52

    ### Si usas Rendimientos Logarítmicos (Continuos) para pasar a rend. anuales - Rendimiento Anualizado = Media Diaria * 252
    ### Si usas Rendimientos Simples (Porcentuales) - Rendimiento Anualizado=(1+Media Diaria)^252 - 1
 
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
precios = precios.rename(columns={ticker_mercado: "MERCADO"})
 
 
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
rend_sml = ER_f + beta_grid * prima_mercado # beta_grid tiene los valores predichos que son el rendimiento exigido = y^, tal que si multipliamos por x (la prima de riesgo del mercado) y sumando el intercepto nos dará la linea de mejor ajuste = SML 
 
 
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








# ============================================================
# CAPM - Security Market Line con S&P 500 y T-Bills (EUA)
# ============================================================
# Cambios clave vs versión México:
#   - Rf: ^IRX (T-Bill 13 semanas) desde Yahoo Finance, NO API externa
#   - Benchmark: ^GSPC (S&P 500) por defecto
#   - Todo en Yahoo Finance → una sola fuente, menos fricción
#   - Frecuencia semanal: viernes (mercado USA cierra viernes)
# ============================================================

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def capm_sml_usa(
    fecha_ini:       str,          # "YYYY-MM-DD"
    fecha_fin:       str,          # "YYYY-MM-DD"
    tickers:         list,         # Lista de tickers, ej. ["AAPL", "MSFT"]
    ticker_mercado:  str = "^GSPC" # Benchmark; S&P 500 por defecto
):
    """
    Calcula el modelo CAPM y grafica la SML para activos del mercado de EUA.

    Parámetros
    ----------
    fecha_ini      : Fecha de inicio del análisis (str, formato YYYY-MM-DD)
    fecha_fin      : Fecha de fin del análisis    (str, formato YYYY-MM-DD)
    tickers        : Lista de tickers de Yahoo Finance a analizar
    ticker_mercado : Ticker del índice de mercado (default: ^GSPC = S&P 500)

    Retorna
    -------
    resultados_df  : DataFrame con beta, rendimiento observado, CAPM y alpha
    """

    # ----------------------------------------------------------
    # 1. DESCARGA DE T-BILLS (^IRX) — TASA LIBRE DE RIESGO
    # ----------------------------------------------------------
    # ^IRX = Treasury Bill 13 semanas
    # Yahoo lo reporta como tasa ANUAL de descuento * 100
    # Ej: 5.25 significa 5.25% anual → hay que dividir entre 100
    #
    # DIFERENCIA VS MÉXICO:
    #   México → API REST de Banxico con token de autenticación
    #   EUA    → yf.download("^IRX") directo, sin API key, sin JSON anidado

    raw_irx = yf.download("^IRX", start=fecha_ini, end=fecha_fin, auto_adjust=True)

    # Extraemos el precio de cierre (= tasa publicada ese día)
    irx = raw_irx["Close"].copy()
    irx.columns = ["rf_anual_pct"]          # Renombramos para claridad

    # Convertimos: porcentaje → decimal → semanal (capitalización continua)
    # Misma fórmula que en México: ln(1 + r_anual) / 52
    irx["rf_anual"]   = irx["rf_anual_pct"] / 100
    irx["rf_semanal"] = np.log(1 + irx["rf_anual"]) / 52

    # Nos quedamos solo con rf_semanal indexado por fecha
    rf_df = irx[["rf_semanal"]].copy()


    # ----------------------------------------------------------
    # 2. DESCARGA DE PRECIOS (BENCHMARK + ACCIONES)
    # ----------------------------------------------------------
    simbolos = [ticker_mercado] + tickers

    datos_yahoo = yf.download(
        tickers     = simbolos,
        start       = fecha_ini,
        end         = fecha_fin,
        auto_adjust = True      # Precios ajustados por dividendos y splits
    )

    precios = datos_yahoo["Close"].copy()
    precios = precios.rename(columns={ticker_mercado: "MERCADO"})


    # ----------------------------------------------------------
    # 3. FILTRAR VIERNES
    # ----------------------------------------------------------
    # EUA: usamos VIERNES como frecuencia semanal
    #   → El mercado cierra el viernes; es el día natural de cierre semanal
    #
    # DIFERENCIA VS MÉXICO:
    #   México usaba MARTES porque CETES se subasta los martes (alineación con Banxico)
    #   EUA no tiene esa restricción: ^IRX tiene dato cada día hábil → usamos viernes
    #
    # dayofweek: lunes=0, martes=1, miércoles=2, jueves=3, viernes=4

    precios_viernes = precios[precios.index.dayofweek == 4].copy()
    rf_viernes      = rf_df[rf_df.index.dayofweek == 4].copy()


    # ----------------------------------------------------------
    # 4. RENDIMIENTOS LOG-SEMANALES
    # ----------------------------------------------------------
    rend = np.log(precios_viernes / precios_viernes.shift(1)).dropna()


    # ----------------------------------------------------------
    # 5. MERGE: RENDIMIENTOS + TASA LIBRE DE RIESGO
    # ----------------------------------------------------------
    base_df = rend.join(rf_viernes, how="inner")

    # Rezagamos rf 1 período: usamos la tasa conocida AL INICIO de cada semana
    base_df["rf_lag"] = base_df["rf_semanal"].shift(1)
    base_df = base_df.dropna()


    # ----------------------------------------------------------
    # 6. PARÁMETROS CAPM
    # ----------------------------------------------------------
    R_M = base_df["MERCADO"]
    r_f = base_df["rf_lag"]

    ER_M = R_M.mean()
    ER_f = r_f.mean()

    prima_mercado = ER_M - ER_f   # E[R_M] - r_f


    # ----------------------------------------------------------
    # 7. ESTIMACIÓN DE BETA Y ALPHA POR ACCIÓN
    # ----------------------------------------------------------
    tickers_validos = [
        col for col in base_df.columns
        if col not in ("MERCADO", "rf_semanal", "rf_lag")
    ]

    resultados = []

    for tk in tickers_validos:
        R_i      = base_df[tk]
        exceso_i = R_i - r_f
        exceso_m = R_M - r_f

        # β = Cov(Ri - rf, Rm - rf) / Var(Rm - rf)
        cov_mat = np.cov(exceso_i.values, exceso_m.values)
        beta_i  = cov_mat[0, 1] / cov_mat[1, 1]

        ER_i_obs  = R_i.mean()
        ER_i_capm = ER_f + beta_i * prima_mercado   # CAPM: rf + β*(E[Rm] - rf)
        alpha     = ER_i_obs - ER_i_capm            # Alpha de Jensen

        resultados.append({
            "ticker":                tk,
            "beta":                  beta_i,
            "rendimiento_observado": ER_i_obs,
            "rendimiento_capm":      ER_i_capm,
            "alpha_visual":          alpha
        })

    resultados_df = pd.DataFrame(resultados)


    # ----------------------------------------------------------
    # 8. GRÁFICA DE LA SML
    # ----------------------------------------------------------
    beta_grid = np.linspace(
        resultados_df["beta"].min() - 0.2,
        resultados_df["beta"].max() + 0.2,
        200
    )
    rend_sml = ER_f + beta_grid * prima_mercado

    fig, ax = plt.subplots(figsize=(10, 6))

    # Línea teórica SML
    ax.plot(beta_grid, rend_sml, linewidth=1.5, color="black", label="SML teórica")

    # Puntos observados
    ax.scatter(
        resultados_df["beta"],
        resultados_df["rendimiento_observado"],
        s=70, zorder=5, color="steelblue", label="Activos"
    )

    # Segmentos verticales (alpha visual)
    for _, fila in resultados_df.iterrows():
        ax.vlines(
            x          = fila["beta"],
            ymin       = min(fila["rendimiento_capm"], fila["rendimiento_observado"]),
            ymax       = max(fila["rendimiento_capm"], fila["rendimiento_observado"]),
            linestyles = "dashed",
            colors     = "tomato" if fila["alpha_visual"] > 0 else "gray",
            linewidth  = 1.0
        )

    # Etiquetas de tickers
    for _, fila in resultados_df.iterrows():
        ax.annotate(
            text       = fila["ticker"],
            xy         = (fila["beta"], fila["rendimiento_observado"]),
            xytext     = (0, 6),
            textcoords = "offset points",
            fontsize   = 8
        )

    # Formato eje Y como porcentaje
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y*100:.2f}%"))

    ax.set_title(
        f"Security Market Line — {ticker_mercado}\n"
        f"{fecha_ini}  a  {fecha_fin}",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("β (Beta)", fontsize=11)
    ax.set_ylabel("Rendimiento esperado semanal", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_facecolor("white")

    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------------
    # 9. TABLA FINAL
    # ----------------------------------------------------------
    resultados_df = resultados_df.sort_values("alpha_visual", ascending=False).reset_index(drop=True)
    print("\n=== Resultados CAPM ===")
    print(resultados_df.to_string(index=False))

    return resultados_df


# ============================================================
# EJEMPLO DE USO
# ============================================================

if __name__ == "__main__":

    df = capm_sml_usa(
        fecha_ini      = "2020-01-29",
        fecha_fin      = "2026-04-29",
        tickers        = ["AAPL"],
        ticker_mercado = "^GSPC"   # S&P 500
    )
    
    
# PUEDE LLEVAR A POSIBLES ERRORES PARA INTERPRETAR RESULTADOS CON UNA MUESTRA PEQUEÑA Y GRANDE
# APT SURGE DE ESTO