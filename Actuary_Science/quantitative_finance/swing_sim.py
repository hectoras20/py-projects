
import importlib

import market_swing
importlib.reload(market_swing)

import capm
importlib.reload(capm)


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

ric = 'USDMXN'
# For a given list (an easier way)
rics = ['^SPX', 'AMZN']

# We can achieve this with the "market_data" class
ric_info = market_swing.distribution_manager(ric)

# Getting the asset information.
# Get its graph
ric_info.load_timeseries()
ric_info.plot_timeseries()
# Get its metrics
ric_info.compute_stats()
ric_info.plot()

"""
Interpretación de métricas financieras para series de retornos

Estas métricas son útiles para swing trading, para evaluar el comportamiento del activo
en ventanas de días a semanas. En cada métrica se explican valores altos, bajos, 
negativos, rangos prácticos y qué significan en la realidad del trader.

----------------------------------------------------------------------
MEAN_ANNUAL (Retorno anualizado)
----------------------------------------------------------------------
Descripción:
    Mide cuánto rinde el activo en promedio durante un año si los 
    retornos diarios se amplificaran proporcionalmente.

Valores típicos y su interpretación:
    < 0%  → Retorno negativo: el activo pierde valor en promedio.
    0% a 5% → Retorno bajo. Común en activos defensivos o periodos laterales.
    5% a 12% → Retorno moderado. Rango típico en acciones estables.
    12% a 20% → Retorno bueno. Indica tendencia positiva sostenida.
    > 20% → Muy alto. Puede reflejar gran momentum o burbuja.

Notas:
    Para swing trading NO predice lo que ganarás en semanas, pero indica 
    si el activo tiende a ser ganador o perdedor.

----------------------------------------------------------------------
VOLATILITY_ANNUAL (Volatilidad anualizada)
----------------------------------------------------------------------
Descripción:
    Mide qué tanto varían los retornos. No indica pérdidas sino dispersión.

Valores típicos:
    < 10% → Muy baja volatilidad. Movimiento estable y lento.
    10% - 20% → Volatilidad moderada. Común en acciones grandes.
    20% - 40% → Alta volatilidad. Común en tecnología o emergentes.
    > 40% → Volatilidad extrema. Cripto, small caps, penny stocks.

Interpretación:
    Más volatilidad = más amplitud de swings, mejor para trading, 
    pero mayor riesgo de movimientos violentos en tu contra.

----------------------------------------------------------------------
SHARPE_RATIO
----------------------------------------------------------------------
Descripción:
    Mide retorno ajustado por riesgo. Relación retorno/volatilidad.

Rangos típicos:
    < 0  → Malo. El riesgo NO vale el retorno.
    0.0 - 0.5 → Débil. Inversión mediocre.
    0.5 - 1.0 → Aceptable. Relación riesgo/retorno razonable.
    1.0 - 1.5 → Buena. Activo eficiente.
    1.5 - 2.0 → Excelente.
    > 2.0 → Excepcional (raro en activos reales).

Interpretación:
    Para swing trading, un Sharpe > 0.7 suele indicar un activo cómodo
    para operar con setups medianos. Valores negativos alertan falta de tendencia.

----------------------------------------------------------------------
VAR_95 (Value-at-Risk al 95%)
----------------------------------------------------------------------
Descripción:
    Estima la pérdida máxima esperada en un día (o periodo elegido) 
    con 95% de confianza.

Interpretación:
    VAR negativo es normal. Ejemplo:
        VaR_95 = -3% significa:
        “En condiciones normales, 95% del tiempo NO perderás más de 3% en un día.”

Rangos:
    -1% a -2% → Riesgo pequeño por día.
    -2% a -4% → Riesgo moderado.
    -4% a -8% → Riesgo alto.
    > -8% → Riesgo extremo.

Notas:
    Útil para dimensionar stops y tamaños de posición.

----------------------------------------------------------------------
SKEWNESS (Asimetría)
----------------------------------------------------------------------
Descripción:
    Mide si los retornos tienen colas largas hacia derecha o izquierda.

Rangos típicos:
    ≈ 0 → Distribución simétrica.
    Negativo (< 0) → Riesgo de caídas bruscas. Colas más peligrosas.
    Positivo (> 0) → Grandes ganancias ocasionales. Favorable para traders.

Interpretación práctica:
    skew < -0.5  → Riesgo de “crash” más frecuente.
    -0.5 < skew < 0.5 → Asimetría leve.
    skew > 0.5 → Buenas sorpresas al alza.

----------------------------------------------------------------------
KURTOSIS (Curtosis)
----------------------------------------------------------------------
Descripción:
    Mide qué tan pesadas son las colas de la distribución.
    Valores recortados (mesokurtic) ≈ 3.

Interpretación:
    < 3 → Colas ligeras (movimientos extremos poco probables).
    ~ 3 → Normal.
    > 3 → Colas pesadas. Mayor probabilidad de eventos extremos.

Rangos prácticos:
    3 - 5 → Riesgo de movimientos grandes superior a lo normal.
    5 - 10 → Riesgo alto de shocks repentinos.
    > 10 → Muy peligroso. Típico de cripto o memestocks.

----------------------------------------------------------------------
JB_STAT y P_VALUE (Test de Jarque-Bera)
----------------------------------------------------------------------
Descripción:
    Prueba si la distribución de retornos es normal.

Interpretación del p-value:
    p > 0.05 → NO rechazas normalidad. Distribución “compatible” con normal.
    p < 0.05 → Rechazas normalidad. Colas pesadas o asimetría presentes.

Interpretación para trading:
    Normality = rara en mercados reales.
    p < 0.05 suele ser normal y marca:
        “Ten cuidado: los eventos extremos ocurren más de lo que crees.”

----------------------------------------------------------------------
IS_NORMAL
----------------------------------------------------------------------
Descripción:
    Booleano que interpreta el p-value.

Interpretación:
    True → Distribución suficientemente normal (estadísticamente).
    False → Comportamiento con colas pesadas, riesgos ocultos o asimetría.

----------------------------------------------------------------------
Resumen general para swing trading
----------------------------------------------------------------------
    • Retorno alto es bueno, pero más importante es la consistencia (Sharpe).
    • Alta volatilidad crea oportunidades pero exige buen control de riesgos.
    • Skew positivo y kurtosis moderada favorecen estrategias momentum.
    • Kurtosis elevada indica probabilidad alta de velas violentas.
    • El VaR te ayuda a dimensionar stops realistas.
    • La normalidad rara vez se cumple: no confíes ciegamente en modelos estadísticos.

"""

"""
Cómo lidiar con cambios macroeconómicos (regime shifts) cuando analizas series históricas
-------------------------------------------------------------------------------

Problema
--------
Cuando ocurre un cambio macro significativo (p. ej. 2025 con un nuevo régimen de crecimiento),
usar todo el historial indiscriminadamente puede introducir 'ruido' o señales muertas:
- parámetros entrenados en un antiguo régimen que ya no aplican,
- estimaciones de volatilidad/expectativas sesgadas,
- tamaños de posición incorrectos.

Opciones prácticas (y cuándo usar cada una)
-------------------------------------------

1) **Recorte de la muestra (rolling lookback fijo)**
   - Idea: usar solo los datos más recientes (p. ej. desde 2025).
   - Cuándo: si el nuevo régimen es claramente distinto, persistente y esperas que dure.
   - Ventajas: elimina "ruido" histórico irrelevante.
   - Inconvenientes: reduce muestras → mayor varianza de estimadores; posible
     perder información útil sobre reversiones raras.
   - Recomendación práctica para swing trading:
       * Ventana corta: 3–6 meses (60–120 sesiones) para señales rápidas.
       * Ventana media: 6–12 meses (120–252 sesiones) para parámetros (ATR, vol).
       * Ventana larga (solo si régimen muy estable): 12–24 meses.
   - Nota: documenta la razón del recorte y registra performance fuera de muestra.

2) **Ponderación exponencial (EWMA) / decaimiento**
   - Idea: conservar todo el historial pero dar más peso a lo reciente.
   - Fórmula típica: w_t ∝ λ^(0..n) con λ ∈ (0,1). Valores comunes:
       * λ = 0.94 (uso financiero clásico, más sensible)
       * λ = 0.97 (menos sensible)
       * λ = 0.99 (muy lento)
   - Ventaja: suaviza transición entre regímenes, aprovechas historial pero priorizas lo nuevo.
   - Recomendación: para swing, probar λ = 0.96–0.98 para medias/volatilidad.

3) **Detección de rupturas y segmentación (regime detection)**
   - Idea: testar si hay breakpoints y estimar parámetros por sub-periodo:
       * Tests: CUSUM, Chow test, Bai–Perron (estructural). (Implementación: librerías estadísticas)
   - Uso: si detectas 1–2 rupturas, modelas parámetros por régimen y aplicas el régimen actual.
   - Ventaja: formal, permite comparar rendimientos/riesgos por régimen.
   - Inconveniente: requiere datos y cuidado estadístico.

4) **Modelos con variables macro (conditioning)**
   - Idea: añadir variables macro (PIB, inflación, tasas, tipo de cambio, indicador de spread)
     como features que modulan señales o tamaño de la posición.
   - Ejemplo: si "macroecon_status == growth" entonces parámetros A, else parámetros B.
   - Ventaja: decisiones explícitas y reproducibles; más interpretabilidad.

5) **Backtest por escenarios / stress testing**
   - Idea: correr backtests separados: (pre-régimen, post-régimen, ambos) y comparar.
   - Incluir shocks hipotéticos (picos de vol, tasas, FX) para ver robustez.

6) **Downweight / winsorize eventos extremos**
   - Idea: limitar influencia de outliers (winsorize) o usar estimadores robustos
     (mediana, MAD en vez de media/std) para evitar que eventos raros desvíen parámetros.

7) **Walk-forward / rolling optimization**
   - Idea: optimizar parámetros en ventana de entrenamiento y validar en ventana siguiente.
   - Muy recomendable: evita sobreajuste a un único periodo histórico.

8) **Mantener historial pero con vigilancia (monitoring)**
   - Implementa métricas de “drift”:
       * cambio en media/volatilidad > X% respecto a ventana anterior
       * incremento de drawdown promedio
     Cuando drift > umbral → recalibrar o reducir tamaño de posición.

Recomendaciones concretas para swing trading (acciónable)
----------------------------------------------------------
- Para señales y gestión de trade (entrada/stop/exit):
    * usar ATR_14 o vol_20d calculados sobre ventanas 20–60 días.
    * recalibrar stops/targets cuando la vol_20d cambie > 20% respecto a la mediana 60d.
- Para estimación de parámetros (tamaño de posición, VaR):
    * usar EWMA con λ = 0.96–0.98 o ventana truncada de 120 días.
- Para decisión de recorte completo a 2025+:
    * solo si hay evidencia clara (test de ruptura) y tu muestra desde 2025 tiene
      al menos 120–252 observaciones. Si tiene < 120, mejor ponderar o usar 6–12m ventanas.
- Validación:
    * siempre backtestea sobre 3 particiones: pre-régimen, post-régimen, y combinado.
    * usa walk-forward con re-optimización cada 20–60 sesiones.
- Control de riesgo dinámico:
    * si detectas régimen de mayor volatilidad → reduz tamaño posición (por ejemplo, multiplicador del tamaño = vol_target / vol_actual).
- Documentación:
    * guarda la fecha y razón de cualquier recorte o cambio de parámetro (reproducibilidad).

Ejemplo práctico (parámetros sugeridos)
---------------------------------------
- Swing típico:
    * señal: mean_20d > threshold y momentum_5d > 0
    * stop: 1 × ATR_14 (o 1.5× si mercado muy volátil)
    * tamaño: riesgo por trade = 0.5% del equity; tamaño = (equity * 0.005) / (stop_distance)
    * recalibración de parámetros: cada 20 sesiones (rolling)
- EWMA para volatilidad:
    * vol_EWMA_today = sqrt( (1-λ)*sum_{i=0}^{n} λ^{i} * r_{t-i}^2 )
    * λ = 0.97 recomendado inicialmente

Peligros y buenas prácticas
---------------------------
- No recortar únicamente por conveniencia: evita data-snooping.
- Siempre comparar performance out-of-sample.
- Evita eliminar datos que muestren colas relevantes (podrías subestimar riesgo).
- Guarda snapshots (archivos) de datasets y parámetros para auditar decisiones.
- Si usas reglas macro (p. ej. “usar datos desde 2025”), prueba su performance histórica vs alternativa (EWMA/rolling).

Conclusión operacional
---------------------
- No hay una única respuesta correcta. La mejor práctica combina: **ponderación (EWMA)** + **detección de rupturas** + **walk-forward** + **ajuste dinámico de riesgo**.
- Para swing trading, preferir ventanas 60–252 días o EWMA λ=0.96–0.98 y recalibrar cada 20–60 sesiones.
- Sólo recortar a 2025+ si tienes suficientes datos post-2025 (≥120–252 observaciones) y / o si tests estadísticos confirman ruptura estructural.

"""

"""
En general, un swing trader no utiliza una temporalidad fija para analizar datos históricos.
La ventana de tiempo depende de la dinámica reciente del mercado, porque el swing trading opera
en horizontes de días a semanas, y por lo tanto usar varios años de datos introduce información
vieja que ya no refleja el comportamiento actual del activo.

1. Temporalidad típica usada en swing trading
   Para análisis cuantitativo, un swing trader comúnmente utiliza ventanas entre:
       - 3 a 12 meses (60 a 252 días hábiles)

   Estas ventanas son ideales para estimar:
       - volatilidad reciente
       - momentum real
       - ATR
       - skewness y kurtosis
       - media de retorno
       - VaR
       - drawdowns
       - correlaciones recientes

   La razón es que estas métricas deben reflejar el régimen actual del mercado; usar
   3, 5 o 10 años diluye completamente la información reciente.

2. Ventanas típicas para señales operativas (entradas/salidas)
   Las señales de trading suelen necesitar ventanas más cortas:
       - 5 días
       - 10 días
       - 20 días
       - 40 días

   Estas ventanas capturan la estructura de precio inmediata, ideal para medir momentum,
   rupturas de rango y comportamiento reciente del precio.

3. Ventanas más largas para estabilidad estadística
   Algunas métricas, como la volatilidad anualizada o el Sharpe Ratio, requieren ventanas un poco más estables:
       - 120 días
       - 180 días
       - 252 días (1 año completo)

   Esto evita que una o dos semanas atípicas distorsionen las métricas de riesgo.

4. ¿Por qué alguien usaría 2 o 3 años?
   Esta longitud no se usa para generar señales ni estimar riesgo operativo. Se utiliza para:
       - detectar cambios de régimen macroeconómico
       - entender la estructura cíclica del activo
       - análisis conceptual del comportamiento histórico del mercado

   Pero no para medir parámetros que se usen en trading activo.

5. ¿Depende de las fluctuaciones macro?
   Sí. Si el trader identifica que existe un nuevo régimen macroeconómico (por ejemplo,
   si en 2025 se produce un cambio estructural en tasas, inflación, liquidez o dinámica del mercado),
   entonces debe ajustarse la ventana de análisis.

   Opciones:
       a) Restringir el histórico al nuevo régimen. Por ejemplo:
          tomar solo datos desde 2025 si se considera que toda la estructura previa ya no es representativa.
       b) Usar ponderación exponencial (EWMA) para asignar más peso a los datos recientes,
          usando lambdas como 0.96–0.98 para volatilidad y otras métricas.

6. Regla de oro del swing trading cuantitativo
   La ventana debe ser suficientemente corta para capturar el régimen actual y suficientemente
   larga para obtener métricas estadísticamente estables.

   En resumen:
       - Señales rápidas: 5–20 días
       - Señales lentas: 20–60 días
       - Parámetros de riesgo: 60–252 días
       - Estructura macro: 1–3 años (solo como referencia conceptual)

Este enfoque permite que el trader capture la dinámica reciente del mercado sin contaminar
las métricas con datos antiguos que ya no son relevantes bajo el régimen macro actual.
"""
