# Cotizador 
import pandas as pd
import numpy as np
import importlib
import class_cot
importlib.reload(class_cot)

# Crear los datos
cnsf_m_2013 = [
    0.000433, 0.000433, 0.000434, 0.000434, 0.000435, 0.000436, 0.000438, 0.000440, 0.000443, 0.000446,
    0.000449, 0.000453, 0.000457, 0.000463, 0.000468, 0.000475, 0.000482, 0.000489, 0.000498, 0.000507,
    0.000517, 0.000528, 0.000540, 0.000553, 0.000567, 0.000582, 0.000598, 0.000616, 0.000635, 0.000656,
    0.000678, 0.000703, 0.000729, 0.000757, 0.000788, 0.000821, 0.000857, 0.000896, 0.000938, 0.000983,
    0.001033, 0.001087, 0.001145, 0.001208, 0.001278, 0.001353, 0.001435, 0.001525, 0.001623, 0.001730,
    0.001848, 0.001977, 0.002119, 0.002274, 0.002446, 0.002635, 0.002844, 0.003074, 0.003329, 0.003612,
    0.003926, 0.004275, 0.004664, 0.005096, 0.005579, 0.006119, 0.006723, 0.007400, 0.008160, 0.009015,
    0.009977, 0.011061, 0.012285, 0.013668, 0.015235, 0.017009, 0.019024, 0.021312, 0.023915, 0.026879,
    0.030257, 0.034110, 0.038509, 0.043533, 0.049274, 0.055833, 0.063329, 0.071889, 0.081660, 0.092798,
    0.105476, 0.119875, 0.136184, 0.154594, 0.175291, 0.198441, 0.224184, 0.252613, 0.283760, 0.317576,
    0.353919, 0.392540, 0.433078, 0.475068, 0.517949, 0.561099, 0.603861, 0.645589, 0.685682, 0.723620,
    0.758991
]

# Crear DataFrame
CNSF2013M = pd.DataFrame()
CNSF2013M['t'] = np.arange(len(cnsf_m_2013))
CNSF2013M['qx'] = cnsf_m_2013
CNSF2013M['px'] = 1-CNSF2013M['qx']
CNSF2013M['tpx'] = CNSF2013M['px'][0]
for i in range(1, len(cnsf_m_2013)):
    CNSF2013M.loc[i, 'tpx'] = CNSF2013M.loc[i-1, 'px'] * CNSF2013M.loc[i-1, 'tpx']

cot = class_cot.Cotizador(
    tabla=CNSF2013M,
    edad=19,
    tasaAnual=11, 
    diferimiento=1, # Incluye plazo de pago de primas
    plazoPrimas=1, # 1 ES PARA PRIMA ÚNICA # temporalidad 
    SAF=130000,
    SAS=25000,
    SARentas=8500,
    SAFun=30000,
    SALeg=30000,
    temporalidadADQ=1,
    temporalidadADM=1,
    temporalidadMG=1,
    gastosA= [5], # Lo ideal  es que se anoten todos los porcentajes, si la temporalidad es de 5 entonces que haya 5 valores dentro de esta lista
    gastosO=[5],
    gastosMU=[5], # Si hay más o menos valores de porcentajes dentro de la lista que la temporalidad del gasto indicada, no falla el código.
    plazoPagoRentas=0 # 0 ES PARA VITALICIO
)

cot.PrimaRiesgo_Fallecimiento()
cot.rentasVitalicias()
# cot.t_P_x_conFuncion() # Fue para comprobar que los valores de t_P_x fueran correctos
cot.PrimaRiesgo_Sobrevivencia()
cot.PrimaRiesgo_Funeraria()
cot.CALCULO_rentas()

cot2 = class_cot.Cotizador(
    tabla=CNSF2013M,
    edad=40,
    tasaAnual=5.15, 
    diferimiento=25, # Incluye plazo de pago de primas
    plazoPrimas=1, # 1 ES PARA PRIMA ÚNICA
    SAF=500000,
    SAS=4800,
    SARentas=8500,
    SAFun=30000,
    SALeg=30000,
    temporalidadADQ=1,
    temporalidadADM=1,
    temporalidadMG=1,
    gastosA= [5], # Lo ideal  es que se anoten todos los porcentajes, si la temporalidad es de 5 entonces que haya 5 valores dentro de esta lista
    gastosO=[5],
    gastosMU=[5], # Si hay más o menos valores de porcentajes dentro de la lista que la temporalidad del gasto indicada, no falla el código.
    plazoPagoRentas=0 # 0 ES PARA VITALICIO
)
cot2.PrimaRiesgo_Fallecimiento()
cot2.rentasVitalicias()
# cot.t_P_x_conFuncion() # Fue para comprobar que los valores de t_P_x fueran correctos
cot2.PrimaRiesgo_Sobrevivencia()
cot2.PrimaRiesgo_Funeraria()
cot2.CALCULO_rentas()



