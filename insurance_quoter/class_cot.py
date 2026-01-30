import pandas as pd
import numpy as np

def CALCULAR_Q_x(t, edadContrata, tabla):
    buscar_xt = edadContrata + t
    valor = tabla.loc[tabla['t']  == buscar_xt,'qx'].item()
    return valor

def CALCULAR_t_P_x(t, P_xt):
    """
    Se le debe de dar una lista del tipo[1_P_x, 1_P_x+1, 1_P_x+2]

    si t = 0 entonces calcula 0_P_x
    si t = 1 entonces calula 1_P_x
    
    Retorna un solo valor t_P_x
    """
    if t == 0:
        return 1
    else:
        return CALCULAR_t_P_x(t - 1, P_xt) * P_xt[t - 1]
    
def CALCULAR_tpx(temporalidad, edad, tabla):
    """
    La función primero crea la lista del tipo[1_P_x, 1_P_x+1, 1_P_x+2]
    esto en la variable p_xt
    
    Retorna una lista con los valores de tPx
    Ejemplo:
        si temporalidad = 2, edad = 19, tabla
        retorna [0_P_x, 1_P_x]
    """
    p_xt = [tabla.loc[tabla['t']  == i,'px'].item() for i in edad+np.arange(temporalidad+1)]

    # Función para obtener los valores t_P_x
    # temporalidad = t = 0 entonces calcula 0_P_x con base a la lista dad q_x y recordemos que la función usa INDICES, entonces no trabaja con todos los valores de la lista de p_x, solo con el INDICE que se pida - en este caso t= 0 pero se generaliza 
    list_tpx= np.array([CALCULAR_t_P_x(i, p_xt) for i in np.arange(0, temporalidad)])
    return list_tpx

    
def CALCULAR_vt(temporalidad, tasa):
    """
    Retorna una lista de valores v^t para el flujo"""
    lista_vt = np.array([(1/(1+(tasa/100)))**i for i in range(temporalidad)])
    return lista_vt

def VPA(temporalidad, edad, tabla, tasa): # = t_p_x * v^t
    """
    Retorna """
    t = np.arange(0, temporalidad)
    # Obtenemos por valores de P_x+t - ESTOS VALORES NOS SIRVEN PARA LA PARTE DE RECURSIVIDAD
    t_p_x = CALCULAR_tpx(temporalidad, edad, tabla)
    # Ahora necesitamos la otra parte de nuestro flujo, los valores presentes
    vt = CALCULAR_vt(temporalidad, tasa)
    # Retornará un flujo => OBTUVIMOS Ax:n^1
    vpa = t_p_x * vt
    return vpa


class Cotizador:
	# Creamos el constructor
    def __init__(self, tabla, edad, tasaAnual, diferimiento, plazoPrimas, SAF, SAS, SARentas, SAFun, SALeg, temporalidadADQ = 1, temporalidadADM = 1, temporalidadMG = 1,  gastosA = [16], gastosO= [5], gastosMU= [5], plazoPagoRentas = 0):
		# Plazo primas se usa para el calculo de primas de riesgo, si se desea calcular PRIMA UNICA, el plazo de primas es 1 
        self.tabla = tabla
        self.edad = edad
        self.tasaAnual = tasaAnual
        self.diferimiento = diferimiento
        self.plazoPrimas = plazoPrimas
        self.SAF = SAF
        self.SAS = SAS
        self.SARentas = SARentas
        self.SAFun = SAFun
        self.SALeg = SALeg
        self.temporalidadADQ = temporalidadADQ
        self.temporalidadADM = temporalidadADM
        self.temporalidadMG = temporalidadMG
        self.gastosA = gastosA
        self.gastosO = gastosO
        self.gastosMU = gastosMU
        self.plazoPagoRentas = plazoPagoRentas
        self.PrimaRiesgoFall = None
        self.lista_qx = None
        self.lista_tpx= None
        self.lista_vt =None
        self.rentas = None
        # self.funcion = None # Funcionó para comprobación de valores... es verdad que el primer valor t=1 * t=2 y asi sucesivamente sobre la tabla original, el procedimiento esta en class_correction, el como se obtuvo tPx columna, es correcto y lo mismo
        self.PrimaRiesgoSobr = None
        self.primaFuneraria = None
        self.rentas = None
        
		

	# MÉTODOS CALCULADORES
    
    # def t_P_x_conFuncion(self):
        # t = self.diferimiento 
        # array = [self.tabla.loc[self.tabla['t'] == i, 'px'].item() for i in np.arange(self.edad, self.edad + self.diferimiento)]
        # self.funcion = np.array([CALCULAR_t_P_x(i, array) for i in range(0, t)])
        # Para comprobar que fuera el mismo se uso self.funcion y self.lista_tpx y se comparó
        

    def PrimaRiesgo_Fallecimiento(self):
        t = self.diferimiento 
        '''
        lista_Q_x = []
        lista_t_P_x_vt = []
        for i in np.arange(t): 
            lista_Q_x.append(CALCULAR_Q_x(i, self.edad, self.tabla))
            lista_t_P_x_vt.append(VPA(i, self.edad, self.tabla, self.tasaAnual))

        Q_x = np.array(lista_Q_x)
        t_P_x_Vt = np.array(lista_t_P_x_vt)

        flujoContingente = Q_x * t_P_x_Vt
        self.PrimaRiesgoFall = flujoContingente.sum() * self.SAF'''
        
        """
        self.lista_qx = np.array([CALCULAR_Q_x(i, self.edad, self.tabla) for i in range(t)])
        self.lista_tPx_vt = np.array([VPA(i, self.edad, self.tabla, self.tasaAnual) for i in range(t)])
        flujoContingente = self.lista_qx * self.lista_tPx_vt
        self.PrimaRiesgoFall = flujoContingente.sum() * self.SAF
        """
        self.lista_qx = np.array([CALCULAR_Q_x(i, self.edad, self.tabla) for i in range(t)])
        self.lista_tpx= np.array(CALCULAR_tpx(self.diferimiento, self.edad, self.tabla))
        self.lista_vt = np.array([CALCULAR_vt(t, self.tasaAnual)])
        flujoContingente = self.lista_qx * self.lista_tpx * self.lista_vt
        self.PrimaRiesgoFall = flujoContingente.sum() * self.SAF
        
        
    def rentasVitalicias(self):
        '''
        Función que calcula cuanto dinero se necesita para cubrir las rentas solicitadas por el cliente
        Desde edad + diferimiento (Inicio de pagos) hasta Omega
        
        El codigo comentado es el mismo que el que se dejó para que funcionara la función, solo que era menos limpio.
        
        Retorna el monto que se necesitará para cubrir las rentas vitalicias hasta omega
        '''
        omega = self.tabla['t'].max()
        tasa_mensual = (1 + self.tasaAnual / 100) ** (1/12) - 1
        """
        edadInicioDif = self.diferimiento + self.edad

        r = np.arange(12)
        y = np.arange(edadInicioDif, omega, step=1)

        paraRec_t = np.arange(self.edad, omega, step=1)

        P_xy = []
        for i in paraRec_t:
            P_xy.append(self.tabla.loc[self.tabla['t'] == i, 'px'].item())

        y_new = np.arange(self.diferimiento, omega - self.edad, step=1)
        valores = []
        for i in y_new:
            iP_x = CALCULAR_t_P_x(i, P_xy)
            y_uno_P_x = CALCULAR_t_P_x(i+1, P_xy)
            for j in r:
                valorPresente = ((1/(1+(tasa_mensual)))**(i*12 + j + 1))
                valores.append((iP_x - (j/12)*(iP_x - y_uno_P_x)) * valorPresente)

        valores = np.array(valores).sum()
        self.rentas = self.SARentas * valores
        # Funciona igual que el código de abajo
        """
        edadInicioDif = self.diferimiento + self.edad

        r = np.arange(12) # MESES - PARA RENTAS YA QUE SON MENSUALES
        y = np.arange(self.diferimiento, omega - self.edad, step=1) # AÑOS RESPECTO AL SEGURO

        P_xy = [self.tabla.loc[self.tabla['t'] == i, 'px'].item() for i in np.arange(self.edad, omega, step=1)]

        valores = []
        for i in y:
            iP_x = CALCULAR_t_P_x(i, P_xy)
            y_uno_P_x = CALCULAR_t_P_x(i+1, P_xy)
            for j in r:
                valorPresente = ((1/(1+(tasa_mensual)))**(i*12 + j + 1))
                valores.append((iP_x - (j/12)*(iP_x - y_uno_P_x)) * valorPresente)
                
        valores_sum = np.array(valores).sum()
        self.rentas = self.SARentas * valores_sum
        
    def CALCULO_rentas(self):
        '''if self.plazoPagoRentas == 0:  # Renta vitalicia
            omega = self.tabla['t'].max()
        else:
            omega = self.edad+self.diferimiento+self.plazoPrimas+self.plazoPagoRentas
            
        tasa_mensual = (1 + self.tasaAnual / 100) ** (1/12) - 1

        edadStart = self.diferimiento + self.edad + self.plazoPrimas

        r = np.arange(12)
            
        y_p_x = CALCULAR_tpx(omega-edadStart, edadStart, self.tabla)
            
        valores = []
        for i in range(0, omega-edadStart-1):
            y_P_x = y_p_x[i]
            y_1_P_x = y_p_x[i+1]
            for j in r:
                valorPresente = ((1/(1+(tasa_mensual)))**(i*12 + j + 1))
                valores.append((y_P_x - (j/12)*(y_P_x - y_1_P_x)) * valorPresente)
        valores = np.array(valores).sum()
        self.rentas = self.SARentas * valores'''
        if self.plazoPagoRentas == 0:
            omega = self.tabla['t'].max()
            
        tasa_mensual = (1 + self.tasaAnual / 100) ** (1/12) - 1

        # edadInicioDif = self.diferimiento + self.edad # edad15 + dif5 = 20 SE SUPONE QUE EL DIFERIMIENTO YA INCLUYE EL PLAZO DE PAGO DE RENTAS

        r = np.arange(12)
        t = np.arange(self.edad, omega, step=1) # [15, ..., 109]

        P_xy = []
        for i in t:
            P_xy.append(self.tabla.loc[self.tabla['t'] == i, 'px'].item()) #[1_P_15, 1_P_16, ..., 1_P_109] len = 95
            
        y = np.arange(self.diferimiento, omega - self.edad, step=1) # [5, 6, ..., 110-15 = 94] - len = 95 - A PARTIR de su edad INICIANDO en diferimiento (dif_P_x) HASTA A PARTIR de su edad  hasta (omega-edad) porque asi obtenemos desde 15 hasta omega hay 95 o sea 95_P_x
        
        valores = []
        for i in y:
            y_P_x = CALCULAR_t_P_x(i, P_xy)
            # 5_P_x la persona de edad x sobreviva despues del diferimiento y así hasta 94_P_x la persona de edad 15 sobreviva 94 años osea hasta los 109 años
            y_1_P_x = CALCULAR_t_P_x(i+1, P_xy)
            for j in r:
                valorPresente = ((1/(1+(tasa_mensual)))**(i*12 + j + 1))
                valores.append((y_P_x - (j/12)*(y_P_x - y_1_P_x)) * valorPresente)

        valores = np.array(valores).sum()

        self.rentas = self.SARentas * valores

        
    def PrimaRiesgo_Sobrevivencia(self):
        """
        Calculo de un seguro de sobrevivencia Ax:n^1 =  VPA = n_P_x*v^n = n_E_x
        """
        vpa = VPA(self.plazoPrimas, self.edad, self.tabla, self.tasaAnual)
        self.PrimaRiesgoSobr = self.SAS * vpa
	
    def PrimaRiesgo_Funeraria(self):
        """
        Formula a producir en codigo:
        SA_Fun = Suma_{y=0}^{110-x-1} Suma_{r=0}{{12-1} [ y_P_x - (r/12) * (y_P_x - {y+1}_P_x}) ] * [ (y_P_x - {y+1}_P_x) / 12 ]*v^{12y+r+1}
        """
        omega = self.tabla['t'].max()
        tasa_mensual = (1 + self.tasaAnual / 100) ** (1 / 12) - 1
        r = np.arange(12)
        y = np.arange(0, omega - self.edad -1)
        valores = []
        # Obtengo la lista de valores de y_P_x y {y+1}_P_x 
        y_p_x = CALCULAR_tpx(omega-self.edad, self.edad, self.tabla) # [1, 1_p_x,  ..., ...] para edad 19... 110-19-1 = 90 valores dentro de esta lista pero crearemos 91 valores, de los cuales ocuparemos los primeros 90 para y_P_x, en y+1_P_x ocupamos el restante para evitar problemas de index out of range 
        for i in y: # len(y) = 90
            y_P_x = y_p_x[i]
            y_1_P_x = y_p_x[i+1]
            for j in r:
                valorPresente = (1 / (1 + tasa_mensual)) ** ((12 * i) + j + 1)
                valores.append(
                    (y_P_x - (j / 12) * (y_P_x - y_1_P_x)) *
                    ((y_P_x - y_1_P_x) / 12) * valorPresente)
        valores = np.sum(valores)
        self.primaFuneraria = self.SAFun * valores
        
    
