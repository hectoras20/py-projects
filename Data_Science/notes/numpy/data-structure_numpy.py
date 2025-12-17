import numpy as np

#Escalares, cuando estamos obteniendo un único valor, ejem. temperatura

#Simplemente interpretalo como un elemento de dimension 0, estas trabajando en el campo R1 osea un ESCALAR es de dimension 0.
escalar = np.array(1)
print('Escalar: ', escalar)

#Almacenamiento de varias entradas, elementos, creación de un VECTOR... es de dimension 1
vector = np. array([1,2,3,4])
print('Vector: ', vector)

#Una matriz es de dimensión 2 un ejemplo clasico es el uso o procesamiento de imagenes donde almacenamos cada pixel en un espacio/entrada de esta matriz que puede ser la escala de grises o el valor que tiene en RGB, incluso si eres analista de ventas puedes guardar cual es la compra recurrente de la tienda en x periodo, en la fila puedes representar un producto y en la columna meses

matrix = np.array([[1,2,3],[4,5,6],[7,8,9]]) #Las filas deben estar dentro de corchetes igual, esto indica que es de dimensión 2 
print('Matrix: ', matrix)

#En general puedes manejar de 3 a n dimensiones, las que necesites para poder trabajar 

#En el caso de trabajar con imágenes, descomposición de imagenes RGB, neecesitas trabajar con muchas imágenes pero tenemos lo que es ancho, altura y su descomposciion en RGB, si quieres manejar toda esta información entonces estarias trabajando con 3 dimensiones...

#Como se ve representado el tensor que maneja todos estos conceptos...

tensor = np.array([[[1,2], [3,4], [5,6], [7,8]]]) #Analiza los últimos corchetes que tienes, son 3 entonces es de 3 dimensiones.
print('tenson: ', tensor)

#Vamos a usar ahora el método arange
array_arange = np.arange(10)
print('Uso de array_arange: ', array_arange) #Del 0 al 9

#Matriz identidad
eye_matrix = np.eye(3) #Como argumento va la dimensión, en este caso una matriz de 3x3
print('Eye/Identidad Matrix: ', eye_matrix)

#Matriz diagonal
diag = np.diag([1,2,3,4,5]) #Le paso los valores que quiero que tenga como diagonal, lo demas son valores de 0
print('Matriz con valores de diagonal asignados: ', diag)

#Transpuesta
tranpose = np.transpose(matrix)
print('Matriz Transpuesta de Matrix', tranpose)

#Random desde numpy, método
#De numpy random quiero invocar a random, lo pienso como: del módulo random quiero la función random, le doy una tupla como argumento
random = np.random.random((1,2,3))
print('Uso del método random de Numpy: ', random)

#UN ARRAY PUEDE SER REPRESENTADO POR VARIAS DIMENSIONES

#CLASE 3
print('\nCLASE 3\n')

#Los arrays son instancias que pertenecen a una clase por lo que vamos a poder acceder a sus atributos...

array_3 = np.array([[1,2],[2,3]])
#Para concoer la dimensión de algún array hacemos
print(array_3)
print('Dimensión NUMPY: ', array_3.ndim)
#Dimensión pero ahora de la matriz...
print('Dimensión MATRIX: ', array_3.shape)

#DATA TYPES
print('Data Type: ',array_3.dtype)

"""
Es un objeto de numpy que describe el tipo de elemntos dentro del array, entero, flotante o bool, despues sigue el tamaño del dato, 8 bits, 32 o 64 y otros atributos

Primero tenemos:
uint8 - Es un entero sin signo de 8 bits, este tipo de datos puede representar valores entre 0 y 255, es util para datos que NO PUEDEN SER NEGATIVOS, como la actividad de colores

float32 - Flotante de 32 bits puedes represntar valores con decimales, para calculos cientificos que requiren una precisión moderada pero necesitan ahorrar memoria. Podría ser útil para Matemáticas financieras, todo el temario

float64 - Mayor precisión si lo comparamos con el float32, es el tipo de dato por defecto de numpy para números con decimales
"""
print('\nPruebas de data type\n')
int8 = np.array(3, dtype=np.uint8) #Especificar dentro del argumento de array el tipo de dato que es.
print(int8)

float_int = np.array([1,2,3], dtype='d') #Lo pasa a decimal los valores de entrada inclusive si son enteros, los imprime en formato decimal
print(float_int)

#Puedo hacer conversiones entre el tipo de dato que es con el método ASTYPE
int_float64 = int8.astype(np.float64)
print(int_float64)

#Los demás métodos que veremos como sum, mean y std, corresponden a los vectores/array de numpy para cálculo estadístico
print('\nEstadística Métodos Operadores\n')
#Suma de entradas - producto interno por el vector de dimension correspondiente al otro vector y entradas 1...
print('Con respecto al vector "vector"...')
sum = np.array(vector)
print('Suma: ', sum)
#Promedio
mean = np.mean(vector)
print('Mean: ', mean)
#Desviación estandar
std = np.std(vector)
print('Desv. Est,: ', std)

"""
Algunas operaciones extra fundamentales en el algebra lineal
"""
print('\nÁlgebra Lineal\n')
#Producto
dot = np.dot(vector, vector)
print('Producto Interno, vector: ', dot)

#Determinante
det = np.linalg.det(matrix)
print('Determinante de Matrix: ', det)

#Inversa de una matriz... recuerda que la matriz por su inversa es la matriz identidad. La matriz identidad se comprueba con eye(dimensión)
inv = np.linalg.inv(matrix)
print('Inversa de Matrix: ', inv)

#Resolución de sistema de ecuaciones
A = np.array([[2, 3], [1, 2]])
B = np.array([[4, 5], [5, 4]])
X = np.linalg.solve(A, B)
print("Solución del sistema AX = B:\n", X)

#No olvidar como resolver este tipo de sistemas por tu cuenta.