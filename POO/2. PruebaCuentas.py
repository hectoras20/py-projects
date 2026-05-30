"""
Programa: PruebaCuentas.py
Objetivo: Escribe un programa que te permita administrar Cuentas de débito.
Autor: team “THE ALGORITHM AVENGERS”“THE ALGORITHM AVENGERS”
Fecha: 23/04/2024

"""

# Importación del módulo Cunetas de debito
import Cuentas_de_debito as db
# Inicialización de la variable opcion
opcion = "9"
cuentas = [] # Creación de la lista para almacenar las cuentas creadas
# Bucle principal del programa, menú
while opcion != "8":
    print("\n--- Menú ---")
    print("1. Crear una cuenta")
    print("2. Consultar el saldo de la cuenta")
    print("3. Retirar dinero")
    print("4. Depositar dinero")
    print("5. Cancelar cuenta")
    print("6. Cambiar NIP")
    print("7. Mostrar cuenta")
    print("8. Salir")
    opcion = input("Escribe la opción deseada: ")
    # Validación de la opción ingresada
    if opcion not in "123456789" or len(opcion) != 1:
        print("Opción no válida.\n")
    else:
        # El match y los case no permiten controlar el menu por "casos"
        match opcion:
            case "1":
                print("Crear una cuenta")
                print("1. Constructor por omisión")
                print("2. Constructor por parámetros")
                op = input("Escribe la opción deseada: ")
                if op not in "12" or len(op) != 1:
                    print("No sé qué deseas hacer!\n")
                else:
                    match op:
                        case "1":  #Crear cuenta utilizando el constructor por omisión
                            cuentas.append(db.Cuenta_debito())

                        case "2": # Crear cuenta utilizando el constructor por parámetros
                            numero_cuenta = None # aun no se ha proporcionado ningún valor válido por parte del usuario
                            while True:  # Para crear un bucle
                                try: #try se utiliza para encapsular el código que puede tener errores
                                    numero_cuenta = int(input("Ingresa el número de cuenta (6 dígitos): "))
                                    if len(str(numero_cuenta)) != 6:
                                        raise ValueError("El número de cuenta debe tener exactamente 6 dígitos enteros.")
                                    break # Si el número de cuenta es válido, salir del bucle
                                except ValueError:
                                    # Si ocurre un error al convertir la entrada a un entero o si el número de cuenta no tiene 6 dígitos,
                                    # se imprime una excepción ValueError con un mensaje explicativo.
                                    print("Error: El número de cuenta debe tener exactamente 6 dígitos enteros")

                            nombre_cliente = input("Ingresa el nombre del cliente: ")

                            saldo = None # aun no se ha proporcionado ningún valor válido por parte del usuario
                            while True:  # Para crear un bucle
                                try: #try se utiliza para encapsular el código que puede tener errores
                                    saldo = int(input("Ingresa el saldo inicial: "))
                                    break #Salir del bucle
                                except ValueError:
                                    # Si ocurre un error al ingresar la entrada
                                    # se imprime una excepción ValueError con un mensaje explicativo.
                                    print("Error: El saldo debe ser un número entero.")

                            nip = None # aun no se ha proporcionado ningún valor válido por parte del usuario
                            while True:  # Para crear un bucle
                                try: #try se utiliza para encapsular el código que puede tener errores
                                    nip = int(input("Ingresa el nip de la cuenta: "))
                                    break #Salir del bucle
                                except ValueError:
                                    # Si ocurre un error al ingresar la entrada
                                    # se imprime una excepción ValueError con un mensaje explicativo.
                                    print("Error: El NIP debe ser un número entero.")
                            # se agrega el objeto y sus atributos a la lista
                            cuentas.append(db.Cuenta_debito(numero_cuenta, nombre_cliente, saldo, nip))

                    print("Cuenta creada correctamente")

            case "2":
                # Consultar saldo de la cuenta
                encontrada = False # Indica que aún no la hemos encontrado
                while True: # Para crear un bucle
                    try: #try se utiliza para encapsular el código que puede tener errores
                        # Solicitar al usuario el número de cuenta y validar la entrada
                        numero_cuenta = int(input("Ingresa el número de cuenta (6 dígitos): "))
                        if len(str(numero_cuenta)) != 6:
                            # raise se utiliza para indicar que a ocurrido un problema en la ejecución del programa
                            raise ValueError("El número de cuenta debe tener exactamente 6 dígitos.")
                        break # Si el número de cuenta es válido, salir del bucle
                    except ValueError:
                        # Si ocurre un error al convertir la entrada a un entero o si el número de cuenta no tiene 6 dígitos,
                        # se imprime una excepción ValueError con un mensaje explicativo.
                        print("Error: El número de cuenta debe tener exactamente 6 dígitos")

                for cuenta in cuentas: # Buscar la cuenta correspondiente en la lista de cuentas
                    if cuenta.numero_cuenta == numero_cuenta:
                        print("El saldo de la cuenta es de: $", cuenta.saldo)
                        encontrada = True # Marcar que se encontró la cuenta
                        break # Detener la operacion una vez que se encuentra la cuenta

                if not encontrada:  #Si la cuenta no se encontró en la lista
                    print("La cuenta no fue encontrada.")

            case "3":
                encontrada = False  # Indica que aún no la hemos encontrado
                while True: # Para crear un bucle
                    try:  #try se utiliza para encapsular el código que puede tener errores
                        # Solicitar al usuario el número de cuenta y validar la entrada
                        numero_cuenta = int(input("Ingresa el número de cuenta (6 dígitos): "))
                        if len(str(numero_cuenta)) != 6:
                            raise ValueError("El número de cuenta debe tener exactamente 6 dígitos.")
                        break # Si el número de cuenta es válido, salir del bucle
                    except ValueError:
                        # Si ocurre un error al ingresar la entrada
                        # se imprime una excepción ValueError con un mensaje explicativo.
                        print("Error: El número de cuenta debe tener exactamente 6 dígitos")

                for cuenta in cuentas: # Buscar la cuenta correspondiente en la lista de cuentas
                    if cuenta.numero_cuenta == numero_cuenta:
                        while True:
                            try: #try se utiliza para encapsular el código que puede tener errores
                                # Solicitar al usuario la cantidad a retirar y validar la entrada
                                cantidad = int(input("Ingrese la cantidad a retirar: $"))
                                break # Si la cantidad es válido, salir del bucle
                            except ValueError:
                                print("Error: La cantidad debe ser un número entero.")
                        while True: # Para crear un bucle
                            try: #try se utiliza para encapsular el código que puede tener errores
                                # Solicitar al usuario el NIP y validar la entrada
                                nip = int(input("Ingrese el NIP de la cuenta: "))
                                break # Si el nip es válido, salir del bucle
                            except ValueError:
                                # Si ocurre un error al ingresar la entrada
                                # se imprime una excepción ValueError con un mensaje explicativo.
                                print("Error: El NIP debe ser un número entero.")
                        cuenta.retirar(cantidad, nip)
                        encontrada = True  # Marcar que se encontró la cuenta
                        break  # Detener la operacion una vez que se encuentra la cuenta

                if not encontrada:  # Si la cuenta no se encontró en la lista
                    print("La cuenta no fue encontrada.")


            case "4":
                encontrada = False # Indica que aún no la hemos encontrado
                while True: # Para crear un bucle
                    try:  #try se utiliza para encapsular el código que puede tener errores
                        # Solicitar al usuario el número de cuenta y validar la entrada
                        numero_cuenta = int(input("Ingresa el número de cuenta (6 dígitos): "))
                        if len(str(numero_cuenta)) != 6:
                            raise ValueError("El número de cuenta debe tener exactamente 6 dígitos.")
                        break # Si el número de cuenta es válido, salir del bucle
                    except ValueError:
                        # Si ocurre un error al ingresar la entrada
                        # se imprime una excepción ValueError con un mensaje explicativo.
                        print("Error: El número de cuenta debe tener exactamente 6 dígitos")
                for cuenta in cuentas: # Buscar la cuenta correspondiente en la lista de cuentas
                    if cuenta.numero_cuenta == numero_cuenta:
                        while True: #Crear bucle
                            try: #try se utiliza para encapsular el código que puede tener errores
                                # Solicitar al usuario la cantidad depositar y validar la entrada
                                cantidad = int(input("Ingrese la cantidad a depositar: $"))
                                break # Si la cantidad es válido, salir del bucle
                            except ValueError:
                                print("Error: La cantidad debe ser un número entero.")
                        cuenta.depositar(cantidad)
                        encontrada = True  # Marcar que se encontró la cuenta
                        break  # Detener la operacion una vez que se encuentra la cuenta

                if not encontrada:  # Si la cuenta no se encontró en la lista
                    print("La cuenta no fue encontrada.")
            case "5":
                encontrada = False # Indica que aún no la hemos encontrado
                while True: # Para crear un bucle
                    try:  #try se utiliza para encapsular el código que puede tener errores
                        # Solicitar al usuario el número de cuenta y validar la entrada
                        numero_cuenta = int(input("Ingresa el número de cuenta (6 dígitos): "))
                        if len(str(numero_cuenta)) != 6:
                            raise ValueError("El número de cuenta debe tener exactamente 6 dígitos.")
                        break # Si el número de cuenta es válido, salir del bucle
                    except ValueError:
                        # Si ocurre un error al ingresar la entrada
                        # se imprime una excepción ValueError con un mensaje explicativo.
                        print("Error: El número de cuenta debe tener exactamente 6 dígitos")
                for cuenta in cuentas: # Buscar la cuenta correspondiente en la lista de cuentas
                    if cuenta.numero_cuenta == numero_cuenta:
                        while True: # Para crear un bucle
                            try: #try se utiliza para encapsular el código que puede tener errores
                                # Solicitar al usuario el NIP y validar la entrada
                                nip = int(input("Ingrese el NIP de la cuenta: "))
                                break # Si el nip es válido, salir del bucle
                            except ValueError:
                                # Si ocurre un error al ingresar la entrada
                                # se imprime una excepción ValueError con un mensaje explicativo.
                                print("Error: El NIP debe ser un número entero.")
                        cuenta.cancelar_cuenta(nip)
                        cuentas.remove(cuenta) #Se eliminia de la lista
                        encontrada = True  # Marcar que se encontró la cuenta
                        break  # Detener la operacion una vez que se encuentra la cuenta
                if not encontrada:  # Si la cuenta no se encontró en  la lista
                    print("La cuenta no fue encontrada.")
            case "6":
                encontrada = False # Indica que aún no la hemos encontrado
                while True: # Para crear un bucle
                    try:  #try se utiliza para encapsular el código que puede tener errores
                        # Solicitar al usuario el número de cuenta y validar la entrada
                        numero_cuenta = int(input("Ingresa el número de cuenta (6 dígitos): "))
                        if len(str(numero_cuenta)) != 6:
                            raise ValueError("El número de cuenta debe tener exactamente 6 dígitos.")
                        break # Si el número de cuenta es válido, salir del bucle
                    except ValueError:
                        # Si ocurre un error al ingresar la entrada
                        # se imprime una excepción ValueError con un mensaje explicativo.
                        print("Error: El número de cuenta debe tener exactamente 6 dígitos")
                for cuenta in cuentas: # Buscar la cuenta correspondiente en la lista de cuentas
                    if cuenta.numero_cuenta == numero_cuenta:
                        while True: # Para crear un bucle
                            try: #try se utiliza para encapsular el código que puede tener errores
                                # Solicitar al usuario el NIP y validar la entrada
                                antiguo_nip = int(input("Ingrese el NIP de la cuenta: "))
                                break # Si el nip es válido, salir del bucle
                            except ValueError:
                                # Si ocurre un error al ingresar la entrada
                                # se imprime una excepción ValueError con un mensaje explicativo.
                                print("Error: El NIP debe ser un número entero.")
                        while True:
                            try: #try se utiliza para encapsular el código que puede tener errores
                                # Solicitar al usuario el NIP y validar la entrada
                                nuevo_nip = int(input("Ingrese el nuevo NIP de la cuenta: "))
                                break # Si el nip es válido, salir del bucle
                            except ValueError:
                                # Si ocurre un error al ingresar la entrada
                                # se imprime una excepción ValueError con un mensaje explicativo.
                                print("Error: El NIP debe ser un número entero.")
                        cuenta.cambiar_nip(nuevo_nip, antiguo_nip)
                        encontrada = True  # Marcar que se encontró la cuenta
                        break  # Detener la operacion una vez que se encuentra la cuenta
                if not encontrada:  # Si la cuenta no se encontró en la lista
                    print("La cuenta no fue encontrada.")
            case "7":
                encontrada = False
                while True: # Para crear un bucle
                    try:  #try se utiliza para encapsular el código que puede tener errores
                        # Solicitar al usuario el número de cuenta y validar la entrada
                        numero_cuenta = int(input("Ingresa el número de cuenta (6 dígitos): "))
                        if len(str(numero_cuenta)) != 6:
                            raise ValueError("El número de cuenta debe tener exactamente 6 dígitos.")
                        break # Si el número de cuenta es válido, salir del bucle
                    except ValueError:
                        # Si ocurre un error al ingresar la entrada
                        # se imprime una excepción ValueError con un mensaje explicativo.
                        print("Error: El número de cuenta debe tener exactamente 6 dígitos")
                for cuenta in cuentas: # Buscar la cuenta correspondiente en la lista de cuentas
                    if cuenta.numero_cuenta == numero_cuenta:
                        print(str(cuenta)) #Utilzamos el metodo __str__
                        encontrada = True  # Marcar que se encontró la cuenta
                        break  # Detener la operacion una vez que se encuentra la cuenta
                if not encontrada:  # Si la cuenta no se encontró en la lista
                    print("La cuenta no fue encontrada.")
            case "8": #salir
                print("Byeeeee")
