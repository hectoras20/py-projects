"""
Programa: Robot.py
Proposito: Modelar el comportamiento sencillo de un robot.
Autor: team “THE ALGORITHM AVENGERS”
Fecha: 23/04/2024
"""
class Robot:
  #Creamos el constructor
  def __init__(self,*params):
   #Si no recibo ningun param:
   if len(params) == 0:
     self.__nombre = "Omis"
     self.__modelo = "AB"
     self.__serie = "001"
     self.__bateria = 100
     self.__encendido = False
    #Si recibo 5 params
   elif len(params) == 4:
    self.__nombre = params[0]
    self.__modelo = params[1]
    self.__serie = params[2]
    self.__bateria = params[3]
    self.__encendido = False
     #Si recibo 3 params
   elif len(params) == 3:
      self.__nombre = params[0]
      self.__modelo = params[1]
      self.__serie = params[2]
       #En automatico tendra 80 de bateria (dado 3 parametros)
      self.__bateria = 80
       #En automatico estara apagada (dado 3 parametros)
      self.__encendido = False
   #Si recibo 5 params(será de ayuda al crear un segundo robot para compararlo)
   elif len(params) == 5:
     self.__nombre = params[0]
     self.__modelo = params[1]
     self.__serie = params[2]
     self.__bateria = params[3]
     self.__encendido = params[4]
  
  #GET para Nombre
  @property  
  def nombre(self):
    if self.encendido: #No necesitamos hacer == True
       return self.__nombre
    else:
      return "El robot esta apagado"

  #El siguiente método es un SET para el nombre
  @nombre.setter
  def nombre(self,nombre):
  #Por si las dudas aqui si hubieramos puesto en los parametros "nombreDelRobot entonces tambien cambiaria en la parte del condicional self.__nombre = nombreDelRobot"
    if self.encendido: #Si el robot esta encendido...
      if nombre.isalpha(): #Si el nombre es alfabetico... sin numeros, para eso el isalpha
        self.__nombre = nombre
      else: #Si no...
        print("Cuida la escritura del nombre de tu Robot! (No puede llevar digitos o ser vacio)")
    else:
      print("El robot no esta encendido")
      
  #GET para Modelo
  @property
  def modelo(self):
    #La idea es que si esta encendida le devuelva el modelo, si no, devuelva False, hasta que encienda el robot.
    if self.encendido is True:
       return self.__modelo
    else:
       return False
  #El siguiente método es un SET para el modelo
  @modelo.setter
  def modelo(self, modelo: str):
    if self.encendido:
      #Este condicional nos ayuda a poner orden, limites en como el usuario debe ingresar el modelo:
      if not modelo.isalpha() and len(modelo) != 2: #Lo mismo, limitamos el modelo a 2 caracteres y solo letras
        print("El modelo debe contener unicamente 2 letras mayusculas")
      else:
         self.__modelo = modelo.upper()
    else:
      print("El robot no esta encendido")

  #GET para Serie
  @property
  def serie(self):
    #Si esta encendida le devuelva la serie, si no, devuelva False, hasta que encienda el robot.
    if self.encendido: 
       return self.__serie
    else:
       return False
      
  #El siguiente método es un SET para la serie
  @serie.setter
  def serie(self,serie):
    if self.encendido:
      #Este condicional nos ayuda a poner orden, limites en como el usuario debe ingresar la serie:
      if serie.isdigit and len(serie) == 3:
        #Usamos el metodo isdigit ya que estamos trabajando con str, esto debe concordar con lo que estamos haciendo, de no ser asi habría un error.
        self.__serie = serie
      else: #Si no se cumplio...
        print("La serie debe ser 3 numeros")
    else:
      print("El robot no esta encendido")

  @property
  #El siguiente método es un GET para la bateria
  def bateria(self):
    #Si esta encendida le devuelva la bateria, si no, devuelva False, hasta que encienda el robot.
    if self.encendido: 
       return self.__bateria
    else:
       return False
  #El siguiente método es un SET para la bateria
  @bateria.setter
  def bateria(self,bateria):
    if self.encendido:
      #Este condicional nos ayuda a poner orden, limites en como el usuario debe ingresar la bateria:
      if bateria in range(0,101): #Es valido ya que con la bateria estamos trabajando con tipos de dato numericos (int).  
        self.__bateria = bateria
      else:
        print("La bateria debe estar en un rango de 0 a 100")
    else:
    #O bien pudimos hacer: if bateria < 0 and bateria > 100: hacer un print que indique que esta mal, no esta en el RANGO de 1 a 100.
      print("El robot no esta encendido")


  #El siguiente método es un GET para el estado del robot (encendido o apagado)
  @property
  def encendido(self):
    """
    Método para saber si la Cafetera está encendida o no
    :return: True si está encendida, False en caso contrario
    :rtype: bool
    """
    return self.__encendido
    #Al ser un bool ya no necesitamos else
  #El siguiente método es un SET para el estado del robot (encendido o apagado)
  #Set para el metodo encendido
  @encendido.setter
  def encendido(self,encendido):
    self.__encendido = encendido

  #CALCULADORES
  #CALCULADOR - IMPRIMIR UN MENSAJE CON BASE AL NIVEL DE BATERIA
  def mensajes(self):
    """
    Método para imprimir un mensaje. Con base a un menú establecido en este mismo metodo
    """
    #La verdad este op no se cual es su principal funcion pero si es importante para el funcionamiento de elegir una opcion
    op = 0
    #Mientras la opcion no sea diferente de 2 (salir)
    while op != "2":
        print("1. Imprimir mensaje n veces")
        print("2. Salir")
        #Aqui pone el usuario su opcion
        op = input("Escribe la opción deseada: ")
        #Si lo que ingreso no es 1 o 2 y la longitud de la opcion sea mayor a uno... 
        if op not in "12" or len(op) != 1:
            print("No se que deseas hacer!\n")
        else:
            #Hacemos un match/relacionamos las opciones
            match op:
                #Si escribe uno
                case "1":
                    if self.bateria == 0: #Checa si la bateria esta en 0, de ser asi pues se que sirve que continue el usuario, si necesita pila
                       print("El robot no tiene pila!\n")
                    else:
                       while True:
                         print(f"OJO: El robot tiene {self.bateria}% de bateria, costo: 2% por mensaje\n") #Para que tenga en cuenta lo que puede hacer
                         mensaje = input("Ingrese el mensaje a imprimir: ") #Cual sea el mensaje, no hay problema aqui
                         numdeseado = int(input("Ingrese el numero de veces que desea imprimirlo: ")) #Aqui si debemos de tener cuidado ya que por imprimir un solo mensaje el costo es de 2%
                         if numdeseado > 0: #Primero que el numero n sea mayor a 0, si es asi pass, que continue
                           pass 
                         else:
                           print("El numero de veces debe ser mayor a 0!\n")
                           break
                         numpermitido = self.bateria / 2 #Para que lo considere y de igual forma con este identificador haremos una validacion para ver si le alcanza ma bateria o no para imprimirlo n veces
                         if numdeseado > numpermitido: #Aqui ya le avisa que no le alcanza respecto a la bateria
                            print(f"Cuidado! El robot solo puede enviar {int(numpermitido)} mensajes") #Le indica cuantos puede
                            break
                         else:
                            print(f"{mensaje}\n" * numdeseado) #Aqui ya imprime los mensajes si todo paso bien y correctamente
                            bateria = self.bateria - numdeseado*2 #El costo por imprimir los mensajes debe de ser considerado! y lo hacemos aqui
                            self.bateria = bateria #Creo que estaba de mas pero por si las dudas
                            print(f"La bateria del robot ahora es de {self.bateria}%!\n") #Indica cuanto le queda de bateria despues se esta operacion
                            break
                case "2":
                    print("Saliendo al menú principal...\n")
                  
  #CALCULADOR - CALCULADORA 
  def calculadora (self):
    """
    Método para que el robot reciba 2 valores y ejecute una operacion deseada 
    """
    #La misma nocion que el metodo mensajes
    op = 0
    while op != "5":
        print("1. Producto de 2 valores")
        print("2. Division de 2 valores")
        print("3. Potencia de 2 valores")
        print("4. Raiz de 2 valores")
        print("5. Salir")
        op = input("Escribe la opción deseada: ")
      #Checa que este correcta la eleccion del usuario
        if op not in "12345" or len(op) != 1:
            print("No se que deseas hacer!\n")
        else: 
            match op:
                case "1":
                    #Pudimos colocar los siguientes avisos dentro de una funcion, poniendo como identificador de tipo numerico el costo por operación de cada uno y realizar el mismo proceso dentro de la funcion para posteriormente llamarla dentro del codigo de las 4 operaciones, seria mas dinámico, pero lo dejamos estatico para no perdernos.
                    if self.bateria == 0: #Esta de mas pero como aviso y a consideracion del usuario
                        print("El robot no tiene pila!\n")
                    elif self.bateria - 5 < 0: #Esta de mas este aviso pero para que se considere
                        print("No hay pila suficiente para realizar la operacion!\n")
                    else:
                        while True:
                            print(f"OJO: El robot tiene {self.bateria}% de bateria\n")
                            if self.bateria - 5 == 0: #Este aviso tambien esta de mas, pero para que se tenga en cuenta
                                print("Aviso, necesitaras recargar tu robot despues de esta operacion!\n")
                                pass #Ya que solo fue un aviso, no interviene con la operacion como tal
                            
                            num1 = float(input("Ingrese el primer numero: "))
                            num2 = float(input("Ingrese el segundo numero: "))
                            operacion = num1 * num2 
                            bateria = self.bateria - 5 #Costo de operacion ya ejecutada
                            self.bateria = bateria
                            print(f"El resultado de la operacion es: {round(operacion, 4)}\n")
                            break
                case "2":
                #Se sigue lo mismo, la misma nocion, los mismos avisos a cosiderar por el usuario
                    if self.bateria == 0:
                        print("El robot no tiene pila!\n")
                    elif self.bateria - 5 < 0:
                        print("No hay pila suficiente para realizar la operacion!\n")
                    else:
                        while True:
                            print(f"OJO: El robot tiene {self.bateria}% de bateria\n")
                            if self.bateria - 5 == 0:
                                print("Aviso, necesitaras recargar tu robot despues de esta operacion!\n")
                                pass
                            num1 = float(input("Ingrese el dividendo: "))
                            num2 = float(input("Ingrese el divisor: ")) 
                            if num2 != 0: #No podemos dividir entre 0! pero si es diferente de 0 que pase, no pasa nada
                                pass
                            else:
                                print("Recuerda que no puedes dividir entre 0!\n")
                                break
                            operacion = num1 / num2
                            bateria = self.bateria - 5 #Toma el costo de operacion
                            self.bateria = bateria
                            print(f"El resultado de la operacion es: {round(operacion,4)}\n")
                            break
                case "3":
                #Misma nocion, mismos avisos a considerar por el usuario
                    if self.bateria == 0:
                        print("El robot no tiene pila!\n")
                    elif self.bateria - 10 < 0:
                        print("No hay pila suficiente para realizar la operacion!\n")
                    else:
                        while True:
                            print(f"OJO: El robot tiene {self.bateria}% de bateria\n")
                            if self.bateria - 10 == 0:
                                print("Aviso, necesitaras recargar tu robot despues de esta operacion!\n")
                                pass
                            num1 = float(input("Ingrese el valor base: "))
                            num2 = float(input("Ingrese el valor exponente: ")) 
                            operacion = num1 ** num2 #No hay restricciones
                            bateria = self.bateria - 10
                            self.bateria = bateria
                            print(f"El resultado de la operacion es: {round(operacion,4)}\n")
                            break
                case "4":
                #Misma nocion, mismos avisos a considerar por el usuario
                    if self.bateria == 0:
                        print("El robot no tiene pila!\n")
                    elif self.bateria - 10 < 0:
                        print("No hay pila suficiente para realizar la operacion!\n")
                    else:
                        while True:
                            print(f"OJO: El robot tiene {self.bateria}% de bateria\n")
                            if self.bateria - 10 == 0:
                                print("Aviso, necesitaras recargar tu robot despues de esta operacion!\n")
                                pass
                            valor = float(input("Ingrese el valor a sacar su raiz cuadrada: "))
                            if valor < 0:
                                print("Recuerda utilizar numeros positivos!") #Aqui si no podemos sacar la raiz de un numero negativo
                            else:
                                pass
                            operacion = valor ** 0.5
                            bateria = self.bateria - 10
                            self.bateria = bateria
                            print(f"La raiz cuadrada de {valor} es {operacion}\n")
                            break

  #CALCULADOR - BATERIA
  def bateriacompleta(self):
    """
    Método para cargar la bateria del robot al 100
    """
    if self is None: #Si no hay un robot, que recarga?
        print("No hay un Robot creado!\n")
    elif not self.encendido:
        print("Debes encender primero el Robot!\n") #Debe de estar encendido para cargar 
    else:
        if self.bateria >= 0 and self.bateria <= 99: #Si esta en ese rango que si recarge
            self.bateria = 100 #trabajamos con type int
            print("Se ha cargado el Robot al 100%!\n")
        else:
            if self.bateria == 100: #Pues que va a recargar si ya esta al 100
                print("El robot ya estaba en su bateria máxima!\n")

  #CALCULADOR - ENCENDER/APAGAR
  def onoff(self):
    """
    Método para encender y apagar el robot.
    """
    #Si esta encendido pues que se apage
    if self.encendido:
        self.__encendido = False
    else: #Lo contrario
        self.__encendido = True

  def __str__(self):
    """
    Método que permite imprimir un Robot en formato cadena.
    :return: La cadena en formato str
    :rtype: str
    """
    if self.encendido:
      return "Robot:\nNombre: " + str(self.nombre) + \
        " \nModelo: " + str(self.modelo) + \
        " \nSerie: " + str(self.serie) + \
        " \nBateria: " + str(self.bateria) + \
        " %\nEl robot esta encendio\n"
    else:
        return "El Robot está apagado!\n"

  def __eq__(self, robot):
    """
    Metodo para determinar si dos Robot son iguales o no
    :return: True si son iguales, False si no lo son
    :rtype: bool
    """
    return self.nombre == robot.nombre and\
        self.modelo == robot.modelo and\
        self.serie == robot.serie and\
        self.bateria == robot.bateria and\
        self.encendido == robot.encendido
  
