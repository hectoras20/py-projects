file = open('./ruta')
print(file.read()) #Lee el texto, lo obtiene todo directamente 

#Podemos leerlo linea a linea de la misma forma con métodos de open()
print(file.readline())

#ES IMPORTANTE SABER CUANDO CERRAR EL ARCHIVO PARA LIBERAR MEMORIA!
file.close()

#Podemos usar un for/iteraciones para leer linea poor linea
for line in file:
    print(line)
    #Obtendrá TODO pero linea a linea usando MENOR MEMORIA

#CERRAR

#Hay una instruccion que abre el archivo y lo cierra automáticamente y es con WITH
with open('./ruta') as file: #Es como una función
    for line in file:
        print(line)
    #Todo esto nos ayuda a leer un archivo! Pero, ¿ahora como podemos escribir en el?
    file.write('Write this line on the file') #ESTO ASI SIMPLEMENTE CAUSARIA UN ERROR

#RECUERDA QUE OPEN VIENE POR DEFECTO SOLO CON LA OPCIÓN DE READER, esto se configura dentro de sus parámetros

with open('./ruta', 'r'): #Asi viene por defecto
    pass

with open('./ruta', 'w'): #Así tenemos permisos de escritura pero causaria un error pues ahora no tenemos permisos de lectura y recuerda que write es muy delicado a tal grado que podría eliminar el contenido de nuestro archivo
    pass

#Para los dos permisos es 
with open('./ruta','r+') as file:
    for line in file:
        print(line)
    file.write('\nOn file\n')
    #Con ello evitamos que lo agregue pegado al último caracter del file.txt

#En el último algorítmo estamos respetando en mayor medida la lectura y por ende lo que queremos añadir simplemente lo agrega respetando la lectura

with open('./ruta','w+') as file:
    for line in file:
        print(line)
    file.write('\nOn file\n')
    #HAY QUE SER MÁS CUIDADOSOS pues, sobreescribe en tu archivo, respeta tu escritura añadida y borra todo lo que hubo antes, la información que ta tenía.


