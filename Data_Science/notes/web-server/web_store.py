#Esto es un MÓDULO

import requests

def get_categories():
    #Para hacer la solicitud hacemos...
    r = requests.get('https://api.escuelajs.co/api/v1/categories')
    #Definimos el tipo de información que podriamos obtener:
    #Estado en HTTP
    print(r.status_code)
    #Algo importante es saber la respuesta osea cual es el retorno de informacion que nos eta dando esta api
    print(r.text)

    #Observemos que de hecho, la información que se obtiene es de tipo texto, a pesar de observar que es una lista de diccionarios...
    print(type(r.text))

    #Un string no lo puedo iterar entonces tengo que transformarlo y esto lo haremos con la liberia requests, para transformarlo entonces a un formato, python...

    categories = r.json()
    #En vez de consultarlo en text, le dire que lo convierta en formato JSON QUE DE FORMA AUTOMATICA ME LO TRANSFORMARÁ EN FORMATO LISTA que python va a recnocer de la misma forma en ese formato y los elementos seran diccionarios gracias a que lo reconoció como una lista, CON ELLO YA PUEDO HACER ITERACIONES

    #Supongamos que quiero acceder solo a los titulos, entonces hacemos:
    for category in categories:
        print(category['name'])
