"""
Programa: PruebaRobot.py
Objetivo: Escribe un programa que modele el comportamiento sencillo de un robot.
Autor: team “THE ALGORITHM AVENGERS”“THE ALGORITHM AVENGERS”
Fecha: 23/04/2024
"""
import Robot as r
opcion = "10"
# Declaración del objeto
robot = None  # El objeto Robot no existe todavía
while opcion != "0":
    print("1. Crear un Robot") #Por omision o por 3 o 4 parámetros dados por el usuario
    print("2. Encender/Apagar el Robot") #Metodo onoff
    print("3. Conocer el estado del Robot") #Gets
    print("4. Cambiar el estado del Robot") #Sets
    print("5. Imprimir mensajes") #Metodo mensajes
    print("6. Calculadora de 2 valores") #Metodo calculadora
    print("7. Cargar Robot al 100") #Metodo bateriacompleta
    print("8. Mostrar robot") #Metodo STR
    print("9. Comparar Robots") #Metodo EQ
    print("0. Salir")
    opcion = input("Escribe la opción deseada: ")
    # Validar la opción ingresada, que sea un valor correcto y que sea sólo 1, en este caso si es diferente de una longitud 1…
    if opcion not in "1234567890" or len(opcion) != 1:
        print("Ingresa un valor correcto del Menú principal!\n")
    else:
        #Pudimos realizar un condicional pero nos ahorramos más tiempo y memoria con match, en este caso es como si este match opción fuera el principal ya que es del menú principal así que…
        match opcion:
            case "1":  # Crear Robot
                print("1. Constructor por omisión")
                print("2. Constructor que recibe 2 parámetros")
                print("3. Constructor que recibe 3 parámetros")
                op = input("Escribe la opción deseada: ")
                if op not in "123" or len(op) != 1:
                    print("No sé qué deseas hacer!\n")
                else:
                     #Este match y al igual que otros más serán nuestros match secundarios, veámoslo de esa forma para no perder sentido en lo que estamos realizando…
                    match op:
                        case "1":  # Constructor por omisión
                            robot = r.Robot() #0 params
                        case "2":  # Constructor 3 parámetros
                            nombre = input("Ingrese el nombre del Robot: ") #STR nombre
                            while not nombre.isalpha(): #isalpha checa que la cadena sea alfabetica, mientras no se cumpla que le pida de nuevo hasta que lo ponga bien.
                              print("El nombre no puede contener números!\n")
                              nombre = input("Ingrese el nombre nuevamente: ")
                            while True:
                              modelo = input("Ingrese el modelo del robot: ")
                              modelo = modelo.upper()
                              if len(modelo) != 2 or not modelo.isalpha():
                                print("El modelo debe contener unicamente 2 letras!")
                                modelo = input("Ingrese el modelo del robot: ")
                                modelo = modelo.upper()
                              else: 
                                break
                            serie = input("Ingrese la serie (3 dígitos) del Robot: ")#INT serie 
                            while not serie.isdigit() or len(serie) != 3: #Hasta el el usuario ponga una serie valida
                                print("La serie debe tener 3 dígitos!\n")
                                serie = input("Ingresa la serie nuevamente: ")
                            robot = r.Robot(nombre, modelo, serie) 
                        case "3":  # Constructor 4 parámetros
                            nombre = input("Ingrese el nombre del Robot: ") #STR nombre
                            while not nombre.isalpha(): #Lo mismo, ni debe llevar digitos
                              print("El nombre no puede contener números!\n")
                              nombre = input("Ingrese el nombre nuevamente: ")
                            while True:
                              modelo = input("Ingrese el modelo del robot: ")
                              modelo = modelo.upper()
                              if len(modelo) != 2 or not modelo.isalpha():
                                print("El modelo debe contener unicamente 2 letras!")
                                modelo = input("Ingrese el modelo del robot: ")
                                modelo = modelo.upper()
                              else: 
                                break
                            serie = input("Ingrese la serie (3 dígitos) del Robot: ")#INT serie 
                            while not serie.isdigit() or len(serie) != 3:
                                print("La serie debe tener 3 dígitos!\n")
                                serie = input("Ingresa la serie nuevamente: ")
                            while True: #NO QUEREMOS QUE SE DETENGA EL CODIGO SI EL USUARIO ESCRIBE UNA LETRA, de ser asi le pide de nuevo que lo ingrese
                              bateria = input("Ingrese el nivel de la batería (entre 0 y 100): ")
                              # Verificamos si la entrada contiene letras
                              if not bateria.isdigit():
                                  print("Error: Debe ingresar un número.")
                                  continue
                              # Una vez haciendo la primera validacion entonces convertimos la entrada a un valor int, esto con el fin de hacer la validaciond e que este el rango y para ello debe ser un numero.
                              bateria = int(bateria)
                              # Verificamos si el valor está dentro del rango
                              while not 0 <= bateria <= 100:
                                  print("Error: El nivel de la batería debe estar entre 0 y 100.")
                                  #Y se repite el proceso
                                  bateria = input("Ingrese el nivel de la batería (entre 0 y 100): ")
                                  bateria = int(bateria)
                              break
                            robot = r.Robot(nombre, modelo, serie, int(bateria)) #Crea el robot 
                    print("Haz creado un nuevo Robot!\n")
            case "2":  # Encender/Apagar el Robot
                if robot is None:
                    print("No hay un Robot creado!\n")
                else: #Si existe el robot
                    if robot.encendido: #Si esta encendido
                      print("Apagaste el Robot!\n")
                    else:
                      print("Encendiste el Robot!\n")
                    robot.onoff()
            case "3":  # Conocer el estado del Robot
                if robot is None:
                   print("No hay un Robot creado!\n")
                elif not robot.encendido:
                    print("Debes encender primero el Robot!\n")
                else:
                  op = 0
                  while op != "5":
                    print("1. Conocer el nombre del Robot")
                    print("2. Conocer el modelo del Robot")
                    print("3. Conocer la serie del Robot")
                    print("4. Conocer la bateria del Robot")
                    print("5. Salir")
                    op = input("Escribe la opción deseada: ")
                    if op not in "12345" or len(op) != 1:
                        print("No sé qué deseas hacer!\n")
                    else:
                        match op:
                          case "1":
                            print(f"El nombre del Robot es: {robot.nombre}\n")
                          case "2":
                            print(f"El modelo del Robot es: {robot.modelo}\n")
                          case "3":
                            print(f"La serie del Robot es: {robot.serie}\n")
                          case "4":
                            print(f"La bateria del Robot es: {robot.bateria}%\n")
                          case "5":
                            print("Saliendo del menú de estado del Robot...\n")
            case "4":  # Cambiar el estado del Robot
                if robot is None:
                    print("No hay un Robot creado!\n")
                elif not robot.encendido:
                    print("Debes encender primero el Robot!\n")
                else:
                  op = 0
                  while op != "5":
                    print("1. Cambiar el nombre del Robot")
                    print("2. Cambiar el modelo del Robot")
                    print("3. Cambiar la serie del Robot")
                    print("4. Cambiar la bateria del Robot")
                    print("5. Salir")
                    op = input("Escribe la opción deseada: ")
                    if op not in "12345" or len(op) != 1:
                        print("No sé qué deseas hacer!\n")
                    else:
                        match op:
                          case "1":
                              nombre = input("Ingrese el nombre del Robot: ")
                              while True:
                               if not nombre.isalpha():
                                print("El nombre no puede llevar numeros o estar vacio!")
                                nombre = input("Ingrese el nombre del Robot: ")
                               else:
                                 robot.nombre = nombre
                                 print("Se ha actualizado el nombre del Robot!\n")
                                 break
                          case "2":
                              modelo = input("Ingrese el modelo del robot: ")
                              modelo = modelo.upper()
                              while True:
                                if len(modelo) != 2 or not modelo.isalpha():
                                  print("El modelo debe contener unicamente 2 letras!")
                                  modelo = input("Ingrese el modelo del robot: ")
                                  modelo = modelo.upper()
                                else: 
                                  robot.modelo = modelo
                                  print("Se ha actualizado el modelo del Robot!\n")
                                  break
                          case "3":
                              serie = input("Ingrese la serie (3 dígitos) del Robot: ")
                              while True:
                                if not serie.isdigit() or len(serie) != 3:
                                  print("La serie debe ser 3 numeros!")
                                  serie = input("Ingrese la serie (3 dígitos) del Robot: ")
                                else:
                                  robot.serie = serie
                                  print("Se ha actualizado la serie del Robot!\n")
                                  break
                          case "4":
                              while True:
                                bateria = input("Ingrese el nivel de la batería (entre 0 y 100): ")
                                # Verificamos si la entrada contiene letras
                                if not bateria.isdigit():
                                    print("Error: Debe ingresar un número.")
                                    continue
                                # Una vez haciendo la primera validacion entonces convertimos la entrada a un valor int, esto con el fin de hacer la validaciond e que este el rango y para ello debe ser un numero.
                                bateria = int(bateria)
                                # Verificamos si el valor está dentro del rango
                                while not 0 <= bateria <= 100:
                                    print("Error: El nivel de la batería debe estar entre 0 y 100.")
                                    #Y se repite el proceso
                                    bateria = input("Ingrese el nivel de la batería (entre 0 y 100): ")
                                    bateria = int(bateria)
                                break
                              print("Se ha actualizado la bateria del Robot!\n")
                          case "5":
                            print("Saliendo al menú principal...\n")
            case "5": #IMPRIMIR MENSAJE N VECES
                if robot is None:
                    print("No hay un Robot creado!\n")
                elif not robot.encendido:
                    print("Debes encender primero el Robot!\n")
                else:
                    robot.mensajes()

            case "6":
                if robot is None:
                    print("No hay un Robot creado!\n")
                elif not robot.encendido:
                    print("Debes encender primero el Robot!\n")
                else:
                    robot.calculadora()

            case "7": #CARGAR AL 100
                if robot is None:
                  print("No hay un Robot creado!\n")
                else:
                  robot.bateriacompleta()

            case "8": #MOSTRAR AL ROBOT
                if robot is None: #Si la cafetera no existe
                    print("No hay un Robot creado!\n")
                else:
                    print(robot)
            case "9":
                if robot is None:
                    print("Debes primero crear un Robot!\n")
                else:
                    print("Debes crear un segundo Robot")
                    nombre = input("Escribe el nombre del segundo Robot: ")
                    while not nombre.isalpha(): #Mismas validaciones de siempre
                      print("El nombre no debe llevar numeros!")
                      nombre = input("Escribe el nombre del segundo Robot: ")
                    while True:
                      modelo = input("Ingrese el modelo del robot: ")
                      modelo = modelo.upper()
                      if len(modelo) != 2 or not modelo.isalpha():
                        print("El modelo debe contener unicamente 2 letras!")
                        continue
                      else:
                        break
                    serie = input("Escribe la serie: ")
                    while not serie.isdigit() or len(serie) != 3:
                      print("La serie debe ser 3 numeros!")
                      serie = input("Escribe la serie: ")
                    while True: #NO QUEREMOS QUE SE DETENGA EL CODIGO SI EL USUARIO ESCRIBE UNA LETRA, de ser asi le pide de nuevo que lo ingrese
                      bateria = input("Ingrese el nivel de la batería (entre 0 y 100): ")
                      # Verificamos si la entrada contiene letras
                      if not bateria.isdigit():
                        print("Error: Debe ingresar un número.")
                        continue
                        # Una vez haciendo la primera validacion entonces convertimos la entrada a un valor int, esto con el fin de hacer la validaciond e que este el rango y para ello debe ser un numero.
                      bateria = int(bateria)
                      # Verificamos si el valor está dentro del rango
                      while not 0 <= bateria <= 100:
                          print("Error: El nivel de la batería debe estar entre 0 y 100.")
                          #Y se repite el proceso
                          bateria = input("Ingrese el nivel de la batería (entre 0 y 100): ")
                          bateria = int(bateria)
                      break
                    #No sirve del todo el encendido ): tuvimos problemas
                    encendido = int(input("Escribe si el robot está encendido o apagado: (1/0) ")) 
                    while encendido != 0 and encendido != 1:
                        print("Elige entre 0 y 1") 
                        encendido = int(input("Escribe si el robot está encendido o apagado: (1/0) ")) 
                    if encendido == 1:
                      encendido = True
                    elif encendido == 0:
                      encendido = False
                    else:
                        encendido = int(input("0 para apagado, 1 para encendido:"))
                    otrorobot = r.Robot(nombre, modelo, serie, int(bateria), encendido)
                    print("El segundo Robot se creó!\n")
                    print(otrorobot) #Usamos metodo STR para imprimirlo
                    print("Vamos a comparar el primer Robot con el segundo")
                    if robot == otrorobot: #Revisamos que los 5 atributos sean iguales
                        print("Los Robots son iguales!\n")
                    else:
                        print("Los robots son diferentes!\n")
            case "0":  # Salir
                print("El codigo a finalizado!\n")