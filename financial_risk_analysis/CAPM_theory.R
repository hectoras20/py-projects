## ============================================================
## CAPM - Security Market Line con IPC y CETES
## ============================================================
## Lo que hace este script:
##   1. Descarga la tasa libre de riesgo (CETES 28 días) desde Banxico
##   2. Descarga precios del IPC y acciones del BMV desde Yahoo Finance
##   3. Alinea todo a frecuencia semanal (martes)
##   4. Calcula betas y rendimientos esperados con el modelo CAPM
##   5. Grafica la Security Market Line (SML) con los activos encima
## ============================================================


# ------------------------------------------------------------
# 1. LIBRERÍAS
# ------------------------------------------------------------

library(siebanxicor)       # API de Banxico (Series de Información Económica)
library(quantmod)          # Descarga de precios financieros desde Yahoo Finance
library(PerformanceAnalytics) # Cálculo de rendimientos (Return.calculate)
library(xts)               # Series de tiempo indexadas por fecha (eXtensible Time Series)
library(dplyr)             # Manipulación de data frames con verbos legibles
library(ggplot2)           # Gráficas declarativas (grammar of graphics)
library(scales)            # Formato de ejes en ggplot (ej. porcentajes)
library(zoo)               # Herramientas auxiliares para series de tiempo (lag, na.omit, etc.)


# ------------------------------------------------------------
# 2. PARÁMETROS GLOBALES
# ------------------------------------------------------------

# Token de autenticación para la API del SIE de Banxico
setToken("ee878065cdc64b207edc54ea9ac5919085e3250705ac0ee569f9f461573cf5ff")

fecha_ini <- "2018-01-01"   # Inicio del periodo de análisis
fecha_fin  <- "2025-12-31"  # Fin del periodo de análisis

# ID de la serie de CETES 28 días en el SIE de Banxico
# Se busca en: https://www.banxico.org.mx/SieAPIRest/service/v1/series
idserie <- c("SF60633")

# Ticker del índice de mercado (IPC de la BMV en Yahoo Finance)
ticker_mercado <- "^MXX"

# Tickers de las acciones individuales a analizar (sufijo .MX = Bolsa Mexicana)
tickers <- c(
  "WALMEX.MX",   # Walmart de México
  "FEMSAUBD.MX", # FEMSA
  "GMEXICOB.MX", # Grupo México
  "BIMBOA.MX",   # Grupo Bimbo
  "CEMEXCPO.MX", # CEMEX
  "GFNORTEO.MX", # Banorte
  "ASURB.MX",    # Grupo Aeroportuario del Sureste
  "KIMBERA.MX"   # Kimberly-Clark de México
)


# ------------------------------------------------------------
# 3. DESCARGA Y PROCESAMIENTO DE CETES (tasa libre de riesgo)
# ------------------------------------------------------------

# getSeriesData() llama a la API del SIE y regresa una lista con los datos crudos
serie_banxico <- getSeriesData(idserie, fecha_ini, fecha_fin)

# getSerieDataFrame() convierte esa lista a un data frame con columnas: date, value
tasas <- getSerieDataFrame(serie_banxico, idserie)

# Transformamos la tasa anual a tasa semanal continua (log-rendimiento)
# Usamos el pipe nativo de R (|>) para encadenar operaciones sin variables intermedias
rf_df <- tasas |>
  dplyr::transmute(                          # transmute = mutate + select (solo las columnas nuevas)
    date     = as.Date(date),               # Convertir a tipo Date
    rf_anual = as.numeric(value) / 100      # Pasar de porcentaje (6.5) a decimal (0.065)
  ) |>
  dplyr::arrange(date) |>                   # Ordenar cronológicamente
  dplyr::mutate(
    # Convertir tasa anual a semanal usando capitalización continua:
    # rf_semanal = ln(1 + rf_anual) / 52
    rf_semanal = log(1 + rf_anual) / 52
  ) |>
  dplyr::select(date, rf_semanal)           # Quedarnos solo con las columnas necesarias


# ------------------------------------------------------------
# 4. DESCARGA DE PRECIOS (IPC + ACCIONES) DESDE YAHOO FINANCE
# ------------------------------------------------------------

# Unimos el ticker del mercado con los tickers individuales en un solo vector
simbolos <- c(ticker_mercado, tickers)

# getSymbols() descarga cada símbolo y lo guarda como objeto xts en el entorno global
# auto.assign = TRUE significa que crea automáticamente variables como MXX, WALMEX.MX, etc.
getSymbols(
  Symbols     = simbolos,
  src         = "yahoo",
  from        = fecha_ini,
  to          = fecha_fin,
  auto.assign = TRUE
)

# Ad() extrae la columna de "Adjusted Close" (precio ajustado por dividendos y splits)
# get() recupera un objeto del entorno global por nombre (string → objeto)
precio_mercado         <- Ad(get("MXX"))
colnames(precio_mercado) <- "MERCADO"      # Renombramos la columna para claridad

# lapply() itera sobre el vector de tickers y aplica una función a cada elemento
# resultado: lista de xts, uno por ticker
# do.call(merge, lista) equivale a merge(xts1, xts2, xts3, ...) con un número variable de args
precios_acciones <- do.call(
  merge,
  lapply(tickers, function(x) Ad(get(x)))  # función anónima: recibe ticker, regresa precios ajustados
)
colnames(precios_acciones) <- tickers      # Restaurar nombres originales (Ad() los modifica)

# Unir mercado + acciones en un solo xts alineado por fecha
precios <- merge(precio_mercado, precios_acciones)


# ------------------------------------------------------------
# 5. FILTRAR PRECIOS DE LOS MARTES
# ------------------------------------------------------------

# CETES se publica cada martes → filtramos precios del mismo día para poder hacer merge
# weekdays() regresa el nombre del día en el idioma del sistema operativo
# En inglés: "Tuesday" / En español: "martes"
precios_martes <- precios[weekdays(index(precios)) == "Tuesday", ]

# Si el sistema está en español, usar:
# precios_martes <- precios[weekdays(index(precios)) == "martes", ]


# ------------------------------------------------------------
# 6. CÁLCULO DE RENDIMIENTOS SEMANALES
# ------------------------------------------------------------

# Return.calculate() calcula rendimientos entre períodos consecutivos
# method = "log" → log-rendimientos: ln(P_t / P_{t-1})
# na.omit() elimina filas con NA (la primera fila siempre es NA porque no tiene período anterior)
rend_martes <- na.omit(Return.calculate(precios_martes, method = "log"))


# ------------------------------------------------------------
# 7. MERGE DE RENDIMIENTOS CON TASA LIBRE DE RIESGO
# ------------------------------------------------------------

# rf_df es un data.frame pero necesitamos unirlo con un xts → convertir a xts
rf_xts          <- xts(rf_df$rf_semanal, order.by = rf_df$date)
colnames(rf_xts) <- "rf"

# join = "inner" → solo conserva fechas presentes en AMBOS objetos
base_xts <- merge(rend_martes, rf_xts, join = "inner")

# Usamos la tasa del martes ANTERIOR (lag de 1 período)
# Razón: el rendimiento de la semana se compara con la tasa conocida al inicio de esa semana
# lag.xts(x, k=1) desplaza la serie 1 período hacia adelante (trae el valor pasado al presente)
base_xts$rf_lag <- lag.xts(base_xts$rf, k = 1)

# Eliminamos NAs generados por el lag (primera observación)
base_xts <- na.omit(base_xts)


# ------------------------------------------------------------
# 8. PARÁMETROS DEL MODELO CAPM
# ------------------------------------------------------------

R_M <- base_xts$MERCADO   # Rendimiento semanal del mercado (IPC)
r_f <- base_xts$rf_lag    # Tasa libre de riesgo semanal (CETES, rezagada 1 semana)

ER_M <- mean(R_M)         # Rendimiento promedio del mercado E[R_M]
ER_f <- mean(r_f)         # Tasa libre de riesgo promedio E[r_f]

# Prima de riesgo del mercado: lo que el mercado rinde por encima de lo "seguro"
prima_mercado <- ER_M - ER_f


# ------------------------------------------------------------
# 9. ESTIMACIÓN DE BETA Y RENDIMIENTOS CAPM POR ACCIÓN
# ------------------------------------------------------------

# Excluimos del loop las columnas que no son acciones individuales
tickers_validos <- setdiff(colnames(base_xts), c("MERCADO", "rf", "rf_lag"))

# lapply() itera sobre cada ticker y devuelve una lista de data.frames
resultados <- lapply(tickers_validos, function(tk) {
  
  R_i <- base_xts[, tk][, 1]   # Rendimiento de la acción i (seleccionamos columna como vector xts)
  
  # Excesos sobre la tasa libre de riesgo (lo que rinde MÁS ALLÁ de lo garantizado)
  exceso_i <- R_i - r_f         # Exceso del activo i
  exceso_m <- R_M - r_f         # Exceso del mercado (prima realizada)
  
  # Beta = Cov(R_i - r_f, R_M - r_f) / Var(R_M - r_f)
  # Mide la sensibilidad del activo a los movimientos del mercado
  # beta > 1 → más volátil que el mercado / beta < 1 → menos volátil
  beta_i <- cov(as.numeric(exceso_i), as.numeric(exceso_m)) /
    var(as.numeric(exceso_m))
  
  ER_i_obs  <- mean(as.numeric(R_i))              # Rendimiento observado promedio
  ER_i_capm <- ER_f + beta_i * prima_mercado       # Rendimiento TEÓRICO según CAPM: r_f + β*(E[R_M] - r_f)
  
  # Alpha de Jensen (visual): diferencia entre lo observado y lo predicho por CAPM
  # alpha > 0 → acción generó más rendimiento del que el riesgo justificaría (outperformance)
  # alpha < 0 → acción rindió menos de lo esperado por su nivel de riesgo (underperformance)
  data.frame(
    ticker               = tk,
    beta                 = beta_i,
    rendimiento_observado = ER_i_obs,
    rendimiento_capm     = ER_i_capm,
    alpha_visual         = ER_i_obs - ER_i_capm
  )
})

# bind_rows() apila la lista de data.frames en uno solo (equivalente a rbind de toda la lista)
resultados <- dplyr::bind_rows(resultados)
print(resultados)


# ------------------------------------------------------------
# 10. CONSTRUCCIÓN DE LA SML (Security Market Line)
# ------------------------------------------------------------

# La SML es la recta: E[R_i] = r_f + β * (E[R_M] - r_f)
# Creamos una grilla de betas que cubra el rango observado con margen
beta_grid <- seq(
  min(resultados$beta) - 0.2,    # Límite inferior: beta mínimo observado menos margen
  max(resultados$beta) + 0.2,    # Límite superior: beta máximo observado más margen
  length.out = 200               # 200 puntos para una línea suave
)

# Data frame con la línea teórica de la SML
sml <- data.frame(
  beta       = beta_grid,
  rendimiento = ER_f + beta_grid * prima_mercado   # Ecuación del CAPM
)


# ------------------------------------------------------------
# 11. GRÁFICA DE LA SML
# ------------------------------------------------------------

# ggplot usa "capas" (geoms) que se suman con + para construir la visualización
ggplot() +
  
  # Línea de la SML (teórica)
  geom_line(
    data = sml,
    aes(x = beta, y = rendimiento),
    linewidth = 1
  ) +
  
  # Puntos de cada acción (rendimiento OBSERVADO)
  geom_point(
    data = resultados,
    aes(x = beta, y = rendimiento_observado),
    size = 3
  ) +
  
  # Segmentos verticales entre el rendimiento CAPM y el OBSERVADO (visualiza el alpha)
  geom_segment(
    data = resultados,
    aes(
      x    = beta, xend = beta,
      y    = rendimiento_capm, yend = rendimiento_observado
    ),
    linetype = "dashed"   # Línea punteada para indicar la desviación
  ) +
  
  # Etiquetas con el nombre del ticker sobre cada punto
  geom_text(
    data = resultados,
    aes(x = beta, y = rendimiento_observado, label = ticker),
    nudge_y = 0.0015,     # Desplazar etiqueta ligeramente arriba del punto
    size    = 3.2
  ) +
  
  # Formatear el eje Y como porcentaje con 2 decimales
  scale_y_continuous(labels = percent_format(accuracy = 0.01)) +
  
  labs(
    title    = "Security Market Line (SML)",
    subtitle = "CAPM semanal con IPC y CETES",
    x        = expression(beta),                   # Símbolo β en el eje X
    y        = "Rendimiento esperado semanal"
  ) +
  
  theme_minimal(base_size = 12)   # Tema limpio sin fondo gris


# ------------------------------------------------------------
# 12. TABLA FINAL ORDENADA POR ALPHA
# ------------------------------------------------------------

# Ordenar de mayor a menor alpha (los mejores outperformers primero)
resultados <- resultados |>
  dplyr::arrange(desc(alpha_visual))

print(resultados)