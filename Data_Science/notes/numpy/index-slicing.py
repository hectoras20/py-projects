#Te permite seleccionar y manipular subconjuntos de datos sin necesidad de copiar grande volúmenes de información

#Recuerda el manejo en listas de la indexación y slicing

import numpy as np

array = np.array([10,20,30,40])
#Recuerda que los indices comienzan desde 0, el valor con índice 0 es 10, el -1 sería el último osea 40
print('Indice 1 de la lista ', array, ' es: ', array[1])
print('Indice -1 de la lista ', array, ' es: ', array[-1])

#Tambien puedo acceder a un subconjunto del conjunto/vector anterior, no revolvamos definiciones.

subset = array[1:3]
print('Subcojunto: ', subset)

#Si nos pasamos del índice hacia el valor de la derecha nos regresa desde el valor indicado hasta el valor máximo que podría retornar, en este caso es 40 pero si nos pasamos del índice de derecha a izquierda (usando indices negativos) me retornaría una lista vacía, ambos resultados son diferentes...
print(array[1:10])
print(array[-1:-10])

"""
Indexación booleana
A esto nos referimos con consulta de valores bajo cierta condición
"""
bool_index = array > 25
#Que elementos son mayores a 25, como tal no regresa los elementos que cumplen esa condición, mas bien nos indican con base a booleanos si cada elementos cumple o no.
print('Los elementos que cumplen ser mayores a 24 del vector dado es son los siguientes: ', bool_index) #Es de tipo ARRAY-numpy

#Puede que queramos acceder a elementos de un array/vector que conozco su índice de la sigueinte manera:
index = [2,0,3] 
#Respeta le orden que tú le das
#A diferencia del ejercicio de subconjuntos, en este caso si te pasas del índice permitido lo tomará como un ERROR
print('Los índices dados dentro de una variable son:\n', array[index])

array = np.random.randint(1,10, size=(3,3))
#Como argumentos recibe de que intervalo va a retornar valores aleatorios (son los primeros dos argumentos) y como los presentará, en este caso los representará como una matriz de 3x3
print('Matriz generada aleatoriamente:\n', array)

#Como funciona la indexación aquí?
print('El valor con índice [fila = 0, columna = 1] es el elemento', array[0,1])

#Tambien podemos realizar cortes que sean de menor longitud dentro de mi matriz, por ejemplo mi matriz de 3x3 la puedo reducir a una matriz de 2x2
print('Matriz original:\n', array)
print('Corte de matriz:\n', array[0:3, :3])
    #LA PRIMERA FILA VA de la fila 0 hasta el tercer elemento de esa fila, LA SEGUNDA FILA VA la última fila (ya que no se indicó nada) hasta el 3 elemento de esa fila.
     
    #Recuerda que si no especifico el valor de la izquierda se toma por defecto el primer valor, primera fila o columna, si no especifico el segundo valor de la derecha toma por defecto el último elemento de mi fila.
