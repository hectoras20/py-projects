#Pandas es una liberia que ya se encarga de funcionalidades como leer un csv, filtrar columnas, filtrar datos, todo ya viene incorporado
#Recuerda que esto se puede realizar inclusive con la creación de una función propia sin la necesidad de este módulo.
import pandas as pd

#Vamos a ver como generar un gráfico con pandas directamente 

#Los DATAFRAMES son todos los datos en donde vamos a obtener la información, por ende crearemos una variable con dicho nombre o simplemente resumirla a df

df = pd.read_csv('data.csv')
#Lo cual nos da el csv en formato diccionario! DE CSV A DICCIONARIO EN PYTHON!

#Ahora, podemos manipular y usar nuestro dataframe a nuestro antojo una vez obtenida la información y esto lo podemos hacer de manerea algorítmicala funcion filter de listas metida dentro de una lista pues filter es una clase y eseta retornaría el objeto de forma... rara y junto con una lambda function o simplemente usando PANDAS  de la siguiente manera:

df = df[df['Continent'] == "South America"]

#data = list(filter(lamnda item : item['Continent'] == 'South America', data(es el from que)))

