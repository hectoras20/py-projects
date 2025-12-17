import csv

"""
def read_csv(path):
    with open(path, 'r') as csvfile: 
        reader = csv.reader(csvfile,  delimiter=',') #La lectura ahora ya no es con OPEN, ahora usaremos la FUNCIÓN READER del modulo importado csv para OBTENER la información, cabe mencionar que el delimiter es con base a como esta presentado tu csv y es un argumento
        for row in reader:
            print('***' * 5)
            print(row)
"""

#Hasta ahorita todo funciona bien con nuestra función, sin embargo todo esto lo botenemos en formato LISTA y ahora lo pasaremos a formato diccionario...

def read_csv(path):
    with open(path, 'r') as csvfile: 
        reader = csv.reader(csvfile,  delimiter=',') #La lectura ahora ya no es con OPEN, ahora usaremos la FUNCIÓN READER del modulo importado csv para OBTENER la información, cabe mencionar que el delimiter es con base a como esta presentado tu csv y es un argumento
        header = next(reader) #Obtenemos la primera linea del csv que no son mas que los encabezados de cada columna.
        data = []
        for row in reader:
            iterable = zip(header,row) #Genera una tupla zip
            #A partir de ese iterable vamos a crear con diccionary comprehension el diccionario
            country_dic = {key:value for key, value in iterable}
            #Necesito almacenar este diccionario en una lista que es mas fácil de manipular, por ello agregamos un data = []
            data.append(country_dic)
        return data


if __name__=='__main__':
    data = read_csv('./ruta') #Guardamos en una variable lo que nos retorna
    print(data)