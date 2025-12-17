#Prueba el módulo creado que consulta información web

import web_store as ws
from fastapi.responses import HTMLResponse

#***Vamos a empezar a crear nuestro servidor web aquí 

from fastapi import FastAPI
#Crearemos una instancia de esa aplicación, NO OLVIDES ()
app = FastAPI()

#Añadimos la sintaxis de un DECORADOR, recuerda que las clases tienen decoradores get y set
@app.get("/") #Le damos la ruta la cual desde la web van a poder ingresar (esta sería la principal)
def get_list():
    return [1,2,3,4]

#Con eso empezariamos a crear un primer recurso que se va a poder ver desde un navegador web, desde el recurso que es el primer decorador.

#***Podemos agregar otra ruta diciendole que tenemos una ruta de contacto por ejemplo y podriamos tener otro tipo de información, en vez de una lista ahora podriamos tener un archivo JSON (diccionario en automático para Python por su estructura natural)

#@app.get("/contac")
#def get_list():
    #return {'name' : 'Platzi'}


#Haremos ahora un response de HTML

@app.get("/contact", response_class=HTMLResponse)
def get_contact():
    return """
<h1>Hola</h1>
    """

def run():
    #module.function()
    ws.get_categories()

if __name__=="__main__":
    run()