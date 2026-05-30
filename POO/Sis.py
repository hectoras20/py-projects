# Programa: Sis.py - SISTEMA PRINCIPAL DE LA INSTITUCIÓN BANCARIA
# Objetivo: Programa que permite controlar un conjunto de personas a través de una lista, para ver sus operaciones más comunes.
# Author: Astudillo Santiago Hector Rodolfo 
# Fecha: 03/06/2024

from validate_email import validate_email # type: ignore
import CD as cd # type: ignore
import CC as cc # type: ignore
import CN as cn # type: ignore
import csv
import E as e # type: ignore
from datetime import datetime
from datetime import date
import random

#Creamos una lista vacia
cuentasT = []
ejecutivos = []

while True:
    print("Sistema Bancario")
    print("1. Cargar información existente")
    print("2. Consultar cuenta por medio de un parámetro")
    print("3. Consultar cuenta por medio de dos parámetros")
    print("4. Dar de alta una nueva cuenta")
    print("5. Actualizar datos de una cuenta existente")
    print("6. Dar de alta un nuevo empleado")
    print("7. Consultar datos de un empleado")
    print("8. Actualizar los datos de un empleado existente")
    print("[E] Eliminar cuentas por criterio")
    print("[D] Depositar dinero (Débito/Crédito)")
    print("[R] Retirar dinero (Débito/Crédito)")
    print("[A] Actualizar el sistema")
    print("[S] Salir")
    opcion = input("¿Qué deseas hacer? ").upper()
      #Condicional de siempre para validar que el usuario eligió correctamente alguna de las opciones 
    if opcion not in "1,2,3,4,5,6,7,8,S,E,D,R,A" or len(opcion) > 1:
        print("No sé qué deseas hacer!\n")
        continue #Al estar en un bucle lo que hace continue es regresarlo al inicio imprimiendo de nuevo el menu y preguntandole de nuevo al usuario 
    else:
        match opcion:
            case "1":  #Cargar información existente
              while True:
                print("¿Qué archivo desea cargar?")
                print("1. Cuentas")
                print("2. Empleados")
                print("[S] Salir")
                opcion = input("¿Qué deseas hacer? ").upper()
                if opcion not in "1,2,S" or len(opcion) > 1:
                    print("No sé qué deseas hacer!\n")
                    continue
                else:
                    match opcion:
                        case "1":
                            archivo = "C:\\Users\\hecto\\Downloads\\Proyecto\\Cuentas.csv"
                                #Al principio el usuario ingresaba el nombre del archivo, posteriormente decidí que el archivo se abriera por defecto
                            with open(archivo, encoding="UTF8", newline="") as file:
            
                                lector = csv.reader(file)
                                # Dado que hay header, vamos a saltar esa línea
                                lector.__next__() #Esto se utiliza para saltar la primera línea del archivo CSV, que generalmente contiene encabezados o nombres de columnas.
                                if cuentasT: #SI NO ESTA VACIA, RECUERDA QUE PONDRIAMOS UN IF NOT que quiere decir SI ESTA VACIA, pero en este caso es solo IF, osea que quiere decir que SI NO ESTA VACIA
                                    print("Ya hay cuentas cargadas en el sistema")
                                    resp = input("¿Deseas agregar al sistema las cuentas del archivo?(y/n): ")
                                    if resp.lower()[0] == "n":
                                        cuentasT = []  # Vaciamos la lista. Esto significa que los datos del archivo CSV sobrescribirán los existentes en la lista, no se agrega nada en pocas palabras. Si el usuario responde "y" (sí) o cualquier otra cosa, la lista de personas no se vacía y los datos del archivo CSV se agregarán a los existentes.
                                for fila in lector:
                                    if fila[0] == "Débito":
                                        cuenta = cd.CuentaDebito(fila[0], fila[1],  # Nombre del cliente
                                                                    int(fila[2]),  # Número de cliente
                                                                    fila[3],  # Número de cuenta
                                                                    float(fila[4]),  # Saldo de la cuenta 
                                                                    fila[5],  # Fecha de apertura de la cuenta
                                                                    fila[6],  # Fecha de corte de la cuenta
                                                                    int(fila[7]),  # Número de sucursal
                                                                    fila[8],  # Correo del cliente
                                                                    fila[9],  # Estado de la cuenta
                                                                    fila[10], # Teléfono del cliente
                                                                    fila[11])  # RFC del cliente
                                        cuentasT.append(cuenta)
                                    elif fila[0] == "Crédito":
                                        cuenta = cc.CuentaCredito(fila[0], fila[1],  # Nombre del cliente
                                                                    int(fila[2]),  # Número de cliente
                                                                    fila[3],  # Número de cuenta
                                                                    float(fila[4]),  # Saldo de la cuenta
                                                                    float(fila[5]),  # Límite de crédito
                                                                    fila[6],  # Fecha de apertura de la cuenta
                                                                    fila[7],  # Fecha de corte de la cuenta
                                                                    fila[8],  # Fecha de vencimiento del crédito
                                                                    int(fila[9]),  # Número de sucursal
                                                                    fila[10],  # Estado de la cuenta
                                                                    fila[11],  # Correo del cliente
                                                                    fila[12],  # Teléfono del cliente
                                                                    fila[13])  # RFC del cliente
                                        cuentasT.append(cuenta)
                                    elif fila[0] == "Nómina":
                                        cuenta = cn.CuentaNomina(fila[0], fila[1],  # Nombre del cliente
                                                                    int(fila[2]),  # Número de cliente
                                                                    fila[3],  # Número de cuenta
                                                                    float(fila[4]),  # Saldo
                                                                    fila[5],  # Fecha de apertura de la cuenta
                                                                    fila[6],  # Fecha de corte de la cuenta
                                                                    int(fila[7]),  # Número de sucursal
                                                                    fila[9],  # Correo del cliente
                                                                    fila[8], # Estado de la cuenta
                                                                    fila[10],  # Teléfono del cliente
                                                                    fila[11], # RFC del cliente
                                                                    fila[12], # Nombre de la empresa
                                                                    fila[13]) # RFC de la empresa
                                        cuentasT.append(cuenta)
                            print("Se han cargado las cuentas al sistema!\n")
                            break
                        case "2":
                            archivo = "C:\\Users\\hecto\\Downloads\\Proyecto\\Ejecutivos.csv"
                            with open(archivo, encoding="UTF8", newline="") as file:
                                lector = csv.reader(file)
                                # Dado que hay header, vamos a saltar esa línea
                                lector.__next__() #Esto se utiliza para saltar la primera línea del archivo CSV, que generalmente contiene encabezados o nombres de columnas.
                                for fila in lector:
                                    if fila[0] =="E":
                                        empleado = e.EjecutivosCuenta(int(fila[1]), fila[2],  # Numero de empleado y RFC
                                                                                fila[3],  # Nombre
                                                                                fila[4],  # Dirección
                                                                                fila[5],  # Telefono
                                                                                float(fila[6]), #Sueldo
                                                                                int(fila[7])) 
                                        ejecutivos.append(empleado)
                                print("Se han cargado los ejecutivos al sistema!\n")
                                if ejecutivos: #SI NO ESTA VACIA, RECUERDA QUE PONDRIAMOS UN IF NOT que quiere decir SI ESTA VACIA, pero en este caso es solo IF, osea que quiere decir que SI NO ESTA VACIA
                                    print("Ya hay cuentas cargadas en el sistema")
                                    resp = input("¿Deseas agregar al sistema los ejecutivos del archivo?(y/n): ")
                                    if resp.lower()[0] == "n":
                                        ejecutivos = []  # Vaciamos la lista. Esto significa que los datos del archivo CSV sobrescribirán los existentes en la lista, no se agrega nada en pocas palabras. Si el usuario responde "y" (sí) o cualquier otra cosa, la lista de personas no se vacía y los datos del archivo CSV se agregarán a los existentes.
                            break
                        case "S":
                            print("Saliendo al menú principal...\n")
                            break        
            case "2": #Buscar cuenta por un parametro dado
                if not cuentasT:  # Comprueba si la lista está vacía con esta sintaxis IF NOT
                    print("\n"+"*"*10 + "Por favor, cargue el sistema de cuentas antes para consultar información!" + "*"*10+"\n")
                else:  # La lista no está vacía
                    while True:
                        print("Buscar por medio de:")
                        print("1. Nombre del Cliente")
                        print("2. Número de Cliente")
                        print("3. Tipo de Cuenta")
                        print("4. Numero de tarjeta o cuenta")
                        print("5. Numero de sucursal")
                        print("6. RFC del ejecutivo")
                        print("7. RFC de la empresa")
                        print("8. Estado")
                        print("[S] Salir")
                        opcion = input("¿Qué deseas hacer? ").upper()
                        if opcion not in "1,2,3,4,5,6,7,8,S" or len(opcion) > 1:
                            print("No sé qué deseas hacer!\n")
                            continue
                        else: 
                            match opcion:
                                case "1": #Por nombre del cliente que es formato STR
                                    encontro = False  # Indica que aún no la hemos encontrado. no necesariamente tenemos que utilizar la palabra encontro, incluso declararla, simplemente para darnos una idea de que esta buscando algo
                                    nombre = input("Escribe el Nombre del Cliente: ")
                                    # Comenzamos la búsqueda
                                    for cuenta in cuentasT:  # Cada cuenta en la lista CuentaT...
                                        if cuenta.nombreCliente == nombre: #Estamos buscando en el array CuentasT, que esta conformado por objetos, y estos a su vez estan conformados por atributos, es por ello que usamos nombreCliente.
                                            print(cuenta)  # Encontramos la cuenta y la imprimimos
                                            encontro = True  # Indica que ya no necesito seguir buscando
                                            print()
                                            #break  LO COMENTAMOS PORQUE NO QUEREMOS QUE NOS MUESTRE UNA SOLA PERSONA, QUEREMOS QUE NOS MUESTRE A TODAS LAS PERSONAS CON EL MISMO NOMBRE
                                    if not encontro:  # Si se recorrió la lista y no encontró nada
                                        print("La cuenta con el Nombre de Cliente {} no fue encontrada".format(nombre))
                                    
                                case "2": #Por numero de cliente
                                    encontro = False  # Indica que aún no la hemos encontrado.
                                    numCliente = input("Escribe el Numero de Cliente: ")
                                    # Comenzamos la búsqueda
                                    for cuenta in cuentasT:  # Cada cuenta en la lista CuentaT...
                                        if cuenta.numeroCliente == int(numCliente):
                                            print(cuenta)  # Encontramos la cuenta y la imprimimos
                                            encontro = True  # Indica que ya no necesito seguir buscando
                                            print()
                                            #break, LO MISMO, NO QUEREMOS UNA SOLA PERSONA 
                                    if not encontro:  # Si se recorrió la lista y no encontró nada
                                        print("La cuenta con numero de Cliente {} no fue encontrada".format(numCliente))
                                case "3": #Por tipo de cuenta TYPE STR
                                    #Se comenta igual
                                    encontro = False
                                    print("Cuida tus acentos! (Débito, Crédito, Nómina)")
                                    tipo = input("Escribe el Tipo de Cuenta: ").title()
                                    for cuenta in cuentasT:
                                        if cuenta.tipo == tipo:
                                            print(cuenta)  # Encontramos la cuenta y la imprimimos
                                            encontro = True  # Indica que ya no necesito seguir buscando
                                            print()
                                            #break
                                    if not encontro:  # Si se recorrió la lista y no encontró nada...
                                        print("Cuentas de tipo {} no fueron encontradas".format(tipo))
                                case "4": #Por numero de tarjeta STR
                                    #Se comenta igual
                                    encontro = False
                                    numTarjeta = input("Escribe el Numero de tarjeta o cuenta: ")
                                    for cuenta in cuentasT:
                                        if cuenta.numeroCuenta == numTarjeta:
                                            print(cuenta)  # Encontramos la cuenta y la imprimimos
                                            encontro = True  # Indica que ya no necesito seguir buscando
                                            print()
                                            #break
                                    if not encontro:  # Si se recorrió la lista y no encontró nada...
                                        print("No se encontraron cuentas con numero de Tarjeta {}".format(numTarjeta))
                                    pass
                                case "5": #Por numero de sucursal TYPE INT
                                    #Se comenta igual
                                    encontro = False
                                    sucursal = int(input("Escribe el Numero de Sucursal: "))
                                    for cuenta in cuentasT:
                                        if cuenta.numeroSucursal == sucursal:
                                            print(cuenta)  # Encontramos la cuenta y la imprimimos
                                            encontro = True  # Indica que ya no necesito seguir buscando
                                            print()
                                            #break
                                    if not encontro:  # Si se recorrió la lista y no encontró nada...
                                        print("No se encontraron cuentas con la sucursal numero {}\n".format(sucursal))
                                case "6": #Por RFC del ejecutivo
                                    #Se comenta igual
                                    encontro = False
                                    RFCeje = input("Escribe el RFC del Ejecutivo: ")
                                    for cuenta in cuentasT:
                                        if cuenta.RFCpersonal == RFCeje:
                                            print(cuenta)  # Encontramos la cuenta y la imprimimos
                                            encontro = True  # Indica que ya no necesito seguir buscando
                                            print()
                                            #break
                                    if not encontro:  # Si se recorrió la lista y no encontró nada...
                                        print("No se encontraron cuentas con RFC {}\n".format(RFCeje))
                                case "7": #Por RFC de la empresa
                                    #Se comenta igual
                                    encontro = False
                                    empresa = input("Escribe el Nombre de la Empresa: ")
                                    for cuenta in cuentasT:
                                        if cuenta.nombreEmpresa == empresa:
                                            print(cuenta)  # Encontramos la cuenta y la imprimimos
                                            encontro = True  # Indica que ya no necesito seguir buscando
                                            print()
                                            #break
                                    if not encontro:  # Si se recorrió la lista y no encontró nada...
                                        print("No se encontraron cuentas con la empresa {}\n".format(empresa))
                                case "8": #Por estado
                                    #Se comenta igual
                                    encontro = False
                                    estado = input("Escribe el Estado: ")
                                    for cuenta in cuentasT:
                                        if cuenta.estado == estado:
                                            print(cuenta)  # Encontramos la cuenta y la imprimimos
                                            encontro = True  # Indica que ya no necesito seguir buscando
                                            print()
                                            #break
                                    if not encontro:  # Si se recorrió la lista y no encontró nada...
                                        print("No se encontraron cuentas con estado {}\n".format(estado))
                                case "S":
                                    print("Saliendo al menú principal...\n")
                                    break
                                
            case "3": #Buscar cuenta por dos parametros dados
                if not cuentasT:  # Comprueba si la lista está vacía con esta sintaxis IF NOT
                    print("\n"+"*"*10 + "Por favor, cargue el sistema de cuentas antes para poder consultar información!" + "*"*10+"\n")
                else:  # La lista no está vacía
                    while True:
                        print("Buscar por medio de:")
                        print("1. Tipo de cuenta y número de sucursal")
                        print("2. Numero de Sucursal y rango de saldo")
                        print("3. Fecha de apertura e importe mayor a alguna cantidad")
                        print("4. Numero de sucursal y nombre de la empresa")
                        print("5. Mes y año de apertura")
                        print("[S] Salir")
                        opcion = input("¿Qué deseas hacer? ").upper()
                        if opcion not in "1,2,3,4,5,S" or len(opcion) > 1:
                            print("No sé qué deseas hacer!\n")
                            continue
                        else: 
                            match opcion:
                                case "1": #Por tipo de cuenta y numero de sucursal
                                    encontro = False  # Indica que aún no la hemos encontrado.
                                    tipo = input("Escribe el Tipo de Cuenta: ")
                                    sucursal = int(input("Escribe el Numero de Sucursal: "))
                                    for cuenta in cuentasT:  # Cada cuenta en la lista CuentaT...
                                        if cuenta.tipo == tipo and cuenta.numeroSucursal == sucursal:
                                            print(cuenta)  # Encontramos la cuenta y la imprimimos
                                            encontro = True  # Indica que ya no necesito seguir buscando
                                            print()
                                            #break EN ESTE CASO de la misma forma, no solo queremos una cuenta, queremos las cuentas que cumplan con ese criterio
                                    if not encontro:  # Si se recorrió la lista y no encontró nada...
                                        print("No se encontraron cuentas con esas caracteristicas\n")
                                case "2": #Por numero de sucursal y rango de saldo
                                    encontro = False  # Indica que aún no la hemos encontrado.
                                    sucursal = int(input("Escribe el Numero de Sucursal: "))
                                    saldo1 = int(input("Escribe el saldo minimo: "))
                                    saldo2 = int(input("Escribe el saldo maximo: "))
                                    for cuenta in cuentasT:  # Cada cuenta en la lista CuentaT...
                                        if cuenta.numeroSucursal == sucursal and cuenta.saldoCredito in range(saldo1, saldo2):
                                            print(cuenta)  # Encontramos la cuenta y la imprimimos
                                            encontro = True  # Indica que ya no necesito seguir buscando
                                            print()
                                            #break, queremos las cuentas
                                    if not encontro:  # Si se recorrió la lista y no encontró nada...
                                            print("No se encontraron cuentas con esas caracteristicas\n")
                                case "3": #Por fecha de apertura e importe mayor a alguna cantidad
                                    encontro = False  # Indica que aún no la hemos encontrado.
                                    try:
                                        fechaapertura = input("Escribe la fecha de apertura (dd-mm-aaaa): ")
                                    except ValueError:
                                        print("Fecha no valida")
                                        fechaapertura = input("Escribe la fecha de apertura (dd-mm-aaaa): ")
                                    cantidad = float(input("Escribe la cantidad minima de importe: "))
                                    for cuenta in cuentasT:  # Cada cuenta en la lista CuentaT...
                                        if cuenta.fecha1 == fechaapertura and cuenta.saldoCredito >= cantidad:
                                            print(cuenta)  # Encontramos la cuenta y la imprimimos
                                            encontro = True  # Indica que ya no necesito seguir buscando
                                            print()
                                            #break No, ya que queremos las cuentas
                                    if not encontro:  # Si se recorrió la lista y no encontró nada...
                                            print("No se encontraron cuentas con esas caracteristicas\n")
                                case "4": #Por numero de sucursal y nombre de la empresa
                                    encontro = False  # Indica que aún no la hemos encontrado.
                                    sucursal = int(input("Escribe el Numero de Sucursal: "))
                                    empresa = input("Escribe el Nombre de la Empresa: ")
                                    for cuenta in cuentasT:  # Cada cuenta en la lista CuentaT...
                                        if cuenta.numeroSucursal == sucursal and cuenta.nombreEmpresa == empresa:
                                            print(cuenta)  # Encontramos la cuenta y la imprimimos
                                            encontro = True  # Indica que ya no necesito seguir buscando
                                            print()
                                            #break  
                                    if not encontro:  # Si se recorrió la lista y no encontró nada...
                                        print("No se encontraron cuentas con esas caracteristicas\n")
                                        
                                case "5": #Por mes y año de apertura
                                    encontro = False  # Aún no se ha encontrado la persona
                                    mes = input("Escribe el mes a buscar: ")
                                    while mes not in "1,2,3,4,5,6,7,8,9,10,11,12":
                                        print("El mes debe ser un valor entre 1 y 12!")
                                        mes = input("Escribe el mes a buscar: ")
                                    año = int(input("Escribe el año a buscar: "))
                                    while 0 > año > 9999:
                                        print("El año debe estar entre 1 y 9999")
                                        año = int(input("Escribe el año a buscar: "))
                                    # Comenzamos la búsqueda en la lista
                                    for cuenta in cuentasT:  # Recorremos la lista
                                        if cuenta.fecha1.year == año and \
                                                cuenta.fecha1.month == int(mes):
                                            encontro = True  # Si encotramos al menos una, la mostramos
                                            print(cuenta)
                                            print("*"*20)
                                    print()
                                    # break  # Este break serviría, si solo quisiera una persona
                                    if not encontro:  # La persona con la fecha indicada, no estaba
                                        print("No hay cuentas con esa fecha de apertura \n")#.format(__nombremes(int(mes)), año))
                                case "S":
                                    print("Saliendo al menú principal...\n")
                                    break
            case "4": #Dar de alta una nueva cuenta
              if not cuentasT:
                    print("\n"+"*"*10 + "Por favor, cargue el sistema de cuentas antes para evitar perdida de información!" + "*"*10+"\n")
              else:
                while True:
                    print("\nCreación de cuentas:")
                    print("1. Cuenta para ejecutivos")
                    print("2. Cuenta para clientes")
                    print("[S] Salir")
                    opcion = input("¿Qué deseas hacer? ").upper()
                    if opcion not in "1,2,S" or len(opcion) > 1:
                        print("No sé qué deseas hacer!\n")
                        continue
                    else:
                        match opcion:
                            case "1": #Cuenta ejecutivo
                                if not ejecutivos:
                                    print("No hay ejecutivos cargados en el sistema!\n")
                                else:
                                                encontro = False
                                                rfcbuscar = input("Escribe el RFC del ejecutivo: ")
                                                if rfcbuscar == "NONE":
                                                    print("No se puede crear una cuenta sin RFC")
                                                    continue
                                                for empleado in ejecutivos:
                                                    if empleado.RFCe == rfcbuscar:
                                                        encontro = True
                                                        print(f"Bienvenido, {empleado.nombreEjecutivo}")
                                                        while True:
                                                            print("\n¿Qué tipo de cuenta deseas tener?")
                                                            print("1. Cuenta de Débito")
                                                            print("2. Cuenta de Crédito")
                                                            print("3. Cuenta de Nómina")
                                                            print("[S] Salir")
                                                            opcion = input("Escribe el número de la opción: ").upper()
                                                            if opcion not in "1,2,3,S" or len(opcion) > 1:
                                                                print("Opción no válida")
                                                                continue
                                                            else:
                                                                match opcion:
                                                                    case "1": # EJECUTIVO - TIPO DÉBITO
                                                                        print("\nInstrucción: Ingrese datos solicitados...\n")
                                                                        tipo = "Débito"
                                                                        nombreEjecutivo = empleado.nombreEjecutivo
                                                                        numeroEjecutivo = int(''.join([str(random.randint(0, 9)) for i in range(8)]))
                                                                        numeroCuenta = str(int(''.join([str(random.randint(0, 9)) for i in range(8)])))
                                                                        while True:
                                                                            try:
                                                                                saldoCredito = float(input("Ingrese el importe del saldo: "))
                                                                            except ValueError:
                                                                                print("El importe debe ser un número")
                                                                                continue
                                                                            if saldoCredito < 0:
                                                                                print("El saldo debe ser mayor a 0")
                                                                                continue
                                                                            else: 
                                                                                break 
                                                                        while True:
                                                                            fecha1 = input("Ingrese la fecha de apertura de la cuenta (dd-mm-yyyy): ")
                                                                            try:
                                                                                fecha1 = datetime.strptime(fecha1, "%d-%m-%Y").date()
                                                                            except ValueError:
                                                                                print("Fecha no valida")
                                                                                continue
                                                                            fecha1 = fecha1.strftime("%d-%m-%Y")
                                                                            break
                                                                        while True:
                                                                            fecha2 = input("Ingrese la fecha de corte de la cuenta (dd-mm-yyyy): ")
                                                                            try: 
                                                                                fecha2 = datetime.strptime(fecha2, "%d-%m-%Y").date()
                                                                            except ValueError:
                                                                                print("Fecha no valida")
                                                                                continue
                                                                            fecha2 = fecha2.strftime("%d-%m-%Y")
                                                                            break
                                                                        numeroSucursal = int(''.join([str(random.randint(1, 6)) for i in range(1)]))
                                                                        estado = input("Ingrese el estado: ")
                                                                        while True:
                                                                            correo = input("Ingrese el correo del cliente: ")
                                                                            if validate_email(correo):
                                                                                break
                                                                            else:
                                                                                print("Correo no valido")
                                                                                continue
                                                                        telefono = empleado.telefono
                                                                        rfceje = empleado.RFCe
                                                                        #Creamos la cuenta
                                                                        cuentaCreada = cd.CuentaDebito(tipo, nombreEjecutivo, numeroEjecutivo, numeroCuenta, saldoCredito, fecha1, fecha2, numeroSucursal, estado, correo, telefono, rfceje)
                                                                        cuentasT.append(cuentaCreada)
                                                                        print("\n", "*"*10 + "BIENVENIDO" + "*"*10)
                                                                        print(f"Tu nueva cuenta de débito ha sido creada con éxito, {nombreEjecutivo}\n")
                                                                        empleado.cantidadCuentas += 1 #SE AGREGA AL NUMERO DE CUENTAS QUE TIENE +1
                                                                        print("Estos son tus datos: \n", cuentaCreada)
                                                                        break
                                                                    case "2": #EJECUTIVO TIPO CRÉDITO
                                                                        print("\nInstrucción: Ingrese datos solicitados...\n")
                                                                        tipo = "Crédito"
                                                                        nombreEjecutivo = empleado.nombreEjecutivo

                                                                        numeroEjecutivo = int(''.join([str(random.randint(0, 9)) for i in range(8)]))
                                                                        numeroCuenta = str(int(''.join([str(random.randint(0, 9)) for i in range(4)])))
                                                                        while True:
                                                                            try:
                                                                                saldoCredito = float(input("Ingrese el importe del credito: "))
                                                                            except ValueError:
                                                                                print("El importe debe ser un número")
                                                                                continue
                                                                            if saldoCredito < 0:
                                                                                print("El saldo debe ser mayor a 0")
                                                                                continue
                                                                            else: 
                                                                                break
                                                                        while True:
                                                                            try:
                                                                                fecha1 = input("Ingrese la fecha de apertura de la cuenta (dd-mm-yyyy): ")
                                                                                fecha1 = datetime.strptime(fecha1, "%d-%m-%Y").date()
                                                                            except ValueError:
                                                                                print("Fecha no valida")
                                                                                continue
                                                                            break
                                                                        while True:
                                                                            try:
                                                                                fecha2 = input("Ingrese la fecha de corte de la cuenta (dd-mm-yyyy): ")
                                                                                fecha2 = datetime.strptime(fecha2, "%d-%m-%Y").date()
                                                                            except ValueError:
                                                                                print("Fecha no valida")
                                                                                continue
                                                                            break
                                                                        numeroSucursal = int(''.join([str(random.randint(1, 6)) for i in range(1)]))
                                                                        estado = input("Ingrese el estado: ")
                                                                        while True:
                                                                            correo = input("Ingrese el correo del cliente: ")
                                                                            if validate_email(correo):
                                                                                break
                                                                            else:
                                                                                print("Correo no valido")
                                                                                continue
                                                                        telefono = empleado.telefono
                                                                        while True:
                                                                            creditoUtilizado = float(input("Ingrese el importe utilizado del credito: "))
                                                                            if creditoUtilizado < 0:
                                                                                print("El crédito utilizao no puede ser menor a 0")
                                                                                continue
                                                                            else: 
                                                                                break
                                                                        while True:
                                                                            try:
                                                                                fechaVencimiento = input("Ingrese la fecha de vencimiento del credito (dd-mm-yyyy): ")
                                                                                fechaVencimiento = datetime.strptime(fechaVencimiento, "%d-%m-%Y").date()
                                                                            except ValueError:
                                                                                print("Fecha no valida")
                                                                                continue
                                                                            break
                                                                        rfceje = empleado.RFCe
                                                                        #Creamos la cuenta
                                                                        cuentaCreada = cc.CuentaCredito(tipo, nombreEjecutivo, numeroEjecutivo, numeroCuenta, saldoCredito, creditoUtilizado,fecha1.strftime("%d-%m-%Y"),fecha2.strftime("%d-%m-%Y"),fechaVencimiento.strftime("%d-%m-%Y"), numeroSucursal, estado, correo, telefono, rfceje)
                                                                        cuentasT.append(cuentaCreada)
                                                                        empleado.cantidadCuentas += 1
                                                                        print("\n", "*"*10 + "BIENVENIDO" + "*"*10)
                                                                        print(f"Tu nueva cuenta de crédito ha sido creada con éxito, {nombreEjecutivo}\n")
                                                                        print("Estos son tus datos: \n", cuentaCreada)
                                                                        break
                                                                    case "3": #EJECUTIVO TIPO NÓMINA
                                                                        print("\nInstrucción: Ingrese datos solicitados...\n")
                                                                        tipo = "Nómina"
                                                                        nombreEjecutivo = empleado.nombreEjecutivo
                                                                        numeroEjecutivo = int(''.join([str(random.randint(0, 9)) for i in range(8)]))
                                                                        numeroCuenta = str(int(''.join([str(random.randint(0, 9)) for i in range(8)])))
                                                                        while True:
                                                                            try:
                                                                                saldoCredito = float(input("Ingrese el importe del credito: "))
                                                                            except ValueError:
                                                                                print("El saldo debe ser un número")
                                                                                continue
                                                                            if saldoCredito < 0:
                                                                                print("El saldo debe ser mayor a 0")
                                                                                continue
                                                                            else: 
                                                                                break

                                                                        while True:
                                                                            try:
                                                                                fecha1 = input("Ingrese la fecha de apertura de la cuenta (dd-mm-yyyy): ")
                                                                                fecha1 = datetime.strptime(fecha1, "%d-%m-%Y").date()
                                                                            except ValueError:
                                                                                print("Fecha no valida")
                                                                                continue
                                                                            break
                                                                        while True:
                                                                            try:
                                                                                fecha2 = input("Ingrese la fecha de corte de la cuenta (dd-mm-yyyy): ")
                                                                                fecha2 = datetime.strptime(fecha2, "%d-%m-%Y").date()
                                                                            except ValueError:
                                                                                print("Fecha no valida")
                                                                                continue
                                                                            break
                                                                        numeroSucursal = int(''.join([str(random.randint(1, 6)) for i in range(1)]))
                                                                        estado = input("Ingrese el estado: ")
                                                                        while True:
                                                                            correo = input("Ingrese el correo del Ejecutivo: ")
                                                                            if validate_email(correo):
                                                                                break
                                                                            else:
                                                                                print("Correo no valido")
                                                                                continue
                                                                        telefono = empleado.telefono
                                                                        RFC = input(f"Ingrese el RFC de la empresa: ")
                                                                        nombreEmpresa = input("Ingrese el nombre de la empresa del ejecutivo: ").title()
                                                                        rfceje = empleado.RFCe
                                                                        #Creamos la cuenta
                                                                        cuentaCreada = cn.CuentaNomina(tipo, nombreEjecutivo, numeroEjecutivo, numeroCuenta, saldoCredito,fecha1.strftime("%d-%m-%Y"),fecha2.strftime("%d-%m-%Y"), numeroSucursal, estado, correo, telefono, RFC, nombreEmpresa, rfceje)
                                                                        cuentasT.append(cuentaCreada)
                                                                        empleado.cantidadCuentas += 1
                                                                        print("\n", "*"*10 + "BIENVENIDO" + "*"*10)
                                                                        print(f"Tu nueva cuenta de nómina ha sido creada con éxito, {nombreEjecutivo}\n")
                                                                        print("Estos son tus datos: \n", cuentaCreada)
                                                                        break
                                                                    case "S":
                                                                        print("Saliendo al menú principal...\n")
                                                                        break

                                                if not encontro:
                                                    print("Ejecutivo no encontrado, asegurate de haberlo registrado previamente (Opción 6 - Menú principal)")
                                                
                            case "2": #CUENTAS PARA CLIENTES
                                while True:
                                    print("\nCuentas Clientes\n" + "Dar de alta una nueva cuenta del tipo:")
                                    print("1. Cuenta de Débito")
                                    print("2. Cuenta de Crédito")
                                    print("3. Cuenta de Nómina")
                                    print("[S] Salir")
                                    opcion = input("Escribe el número de la opción: ").upper()
                                    if opcion not in "1,2,3,S" or len(opcion) > 1:
                                        print("Opción no válida")
                                        continue
                                    else:
                                        match opcion:
                                            case "1": #Clientes - TIPO DÉBITO
                                                print("\nInstrucción: Ingrese datos solicitados...\n")
                                                tipo = "Débito"
                                                while True:
                                                    nombreCliente = input("Ingrese el nombre del Cliente: ").title()
                                                    if nombreCliente.isalpha():
                                                        break
                                                    else:
                                                        print("No puedes ingresar digitos o caracteres especiales!")
                                                        continue
                                                numeroCliente = int(''.join([str(random.randint(0, 9)) for i in range(8)]))
                                                numeroCuenta = str(int(''.join([str(random.randint(0, 9)) for i in range(8)])))
                                                while True:
                                                    try:
                                                        saldoCredito = float(input("Ingrese el importe del saldo: "))
                                                    except ValueError:
                                                        print("El importe debe ser un número")
                                                        continue
                                                    if saldoCredito < 0:
                                                        print("El saldo debe ser mayor a 0")
                                                        continue
                                                    else: 
                                                        break 
                                                while True:
                                                    try: 
                                                        fecha1 = input("Ingrese la fecha de apertura de la cuenta (dd-mm-yyyy): ")
                                                        fecha1 = datetime.strptime(fecha1, "%d-%m-%Y").date()
                                                    except ValueError:
                                                        print("Fecha no valida")
                                                        continue
                                                    break
                                                while True:
                                                    try: 
                                                        fecha2 = input("Ingrese la fecha de corte de la cuenta (dd-mm-yyyy): ")
                                                        fecha2 = datetime.strptime(fecha2, "%d-%m-%Y").date()
                                                    except ValueError:
                                                        print("Fecha no valida")
                                                        continue
                                                    break
                                                numeroSucursal = int(''.join([str(random.randint(1, 6)) for i in range(1)]))
                                                estado = input("Ingrese el estado: ")
                                                while True:
                                                    correo = input("Ingrese el correo del cliente: ")
                                                    if validate_email(correo):
                                                        break
                                                    else:
                                                        print("Correo no valido")
                                                        continue
                                                while True:
                                                    telefono = input("Ingrese el teléfono del Cliente (10 dígitos): ")
                                                    if telefono.isdigit() and len(telefono) == 10:
                                                        break
                                                    else:
                                                        print("El telefono debe ser de 10 digitos")
                                                        continue
                                                rfceje = "NONE"
                                                #Creamos la cuenta
                                                cuentaCreada = cd.CuentaDebito(tipo, nombreCliente, numeroCliente, numeroCuenta, saldoCredito,fecha1.strftime("%d-%m-%Y"),fecha2.strftime("%d-%m-%Y"), numeroSucursal, estado, correo, telefono, rfceje) 
                                                cuentasT.append(cuentaCreada)
                                                print("\n", "*"*10 + "BIENVENIDO" + "*"*10)
                                                print(f"Tu nueva cuenta de débito ha sido creada con éxito, {nombreCliente}\n")
                                                print("Estos son tus datos: \n", cuentaCreada)
                                                break

                                            case "2": #EJECUTIVO TIPO CRÉDITO
                                                print("\nInstrucción: Ingrese datos solicitados...\n")
                                                tipo = "Crédito"
                                                while True:
                                                    nombreCliente = input("Ingrese el nombre del Ejecutivo: ")
                                                    if nombreCliente.isalpha():
                                                        break
                                                    else:
                                                        print("No puedes ingresar digitos o caracteres especiales!")
                                                        continue

                                                numeroCliente = int(''.join([str(random.randint(0, 9)) for i in range(8)]))
                                                numeroCuenta = str(int(''.join([str(random.randint(0, 9)) for i in range(4)])))
                                                while True:
                                                    try:
                                                        saldoCredito = float(input("Ingrese el importe del credito: "))
                                                    except ValueError:
                                                        print("El importe debe ser un número")
                                                        continue
                                                    if saldoCredito < 0:
                                                        print("El saldo debe ser mayor a 0")
                                                        continue
                                                    else: 
                                                        break
                                                while True:
                                                    try:
                                                        fecha1 = input("Ingrese la fecha de apertura de la cuenta (dd-mm-yyyy): ")
                                                        fecha1 = datetime.strptime(fecha1, "%d-%m-%Y").date()
                                                    except ValueError:
                                                        print("Fecha no valida")
                                                        continue
                                                    break
                                                while True:
                                                    try:
                                                        fecha2 = input("Ingrese la fecha de corte de la cuenta (dd-mm-yyyy): ")
                                                        fecha2 = datetime.strptime(fecha2, "%d-%m-%Y").date()
                                                    except ValueError:
                                                        print("Fecha no valida")
                                                        continue
                                                    break
                                                numeroSucursal = int(''.join([str(random.randint(1, 6)) for i in range(1)]))
                                                estado = input("Ingrese el estado: ")
                                                while True:
                                                    correo = input("Ingrese el correo del Cliente: ")
                                                    if validate_email(correo):
                                                        break
                                                    else:
                                                        print("Correo no valido")
                                                        continue
                                                while True:
                                                    telefono = input("Ingrese el número telefónico del Cliente (10 dígitos): ")
                                                    if telefono.isdigit() and len(telefono) == 10:
                                                        break
                                                    else:
                                                        print("El telefono debe contener 10 DÍGITOS")
                                                        continue
                                                while True:
                                                    creditoUtilizado = float(input("Ingrese el importe utilizado del credito: "))
                                                    if creditoUtilizado < 0:
                                                        print("El crédito utilizao no puede ser menor a 0")
                                                        continue
                                                    else: 
                                                        break
                                                while True:
                                                    try:
                                                        fechaVencimiento = input("Ingrese la fecha de vencimiento del credito (dd-mm-yyyy): ")
                                                        fechaVencimiento = datetime.strptime(fechaVencimiento, "%d-%m-%Y").date()
                                                    except ValueError:
                                                        print("Fecha no valida")
                                                        continue
                                                    break
                                                rfceje = "NONE"
                                                #Creamos la cuenta
                                                cuentaCreada = cc.CuentaCredito(tipo, nombreCliente, numeroCliente, numeroCuenta, saldoCredito, creditoUtilizado,fecha1.strftime("%d-%m-%Y"),fecha2.strftime("%d-%m-%Y"), fechaVencimiento.strftime("%d-%m-%Y"), numeroSucursal, estado, correo, telefono, rfceje)
                                                cuentasT.append(cuentaCreada)
                                                print("\n", "*"*10 + "BIENVENIDO" + "*"*10)
                                                print(f"Tu nueva cuenta de crédito ha sido creada con éxito, {nombreCliente}\n")
                                                print("Estos son tus datos: \n", cuentaCreada)
                                                break
                                            case "3":
                                                print("\nInstrucción: Ingrese datos solicitados...\n")
                                                tipo = "Nómina"
                                                while True:
                                                    nombreCliente = input("Ingrese el nombre del Cliente: ").title()
                                                    if nombreCliente.isalpha():
                                                        break
                                                    else:
                                                        print("No puedes ingresar digitos o caracteres especiales!")
                                                        continue
                                                numeroCliente = int(''.join([str(random.randint(0, 9)) for i in range(8)]))
                                                numeroCuenta = str(int(''.join([str(random.randint(0, 9)) for i in range(8)])))
                                                while True:
                                                    try:
                                                        saldoCredito = float(input("Ingrese el importe del credito: "))
                                                    except ValueError:
                                                        print("El saldo debe ser un número")
                                                        continue
                                                    if saldoCredito < 0:
                                                        print("El saldo debe ser mayor a 0")
                                                        continue
                                                    else: 
                                                        break

                                                while True:
                                                    try:
                                                        fecha1 = input("Ingrese la fecha de apertura de la cuenta (dd-mm-yyyy): ")
                                                        fecha1 = datetime.strptime(fecha1, "%d-%m-%Y").date()
                                                    except ValueError:
                                                        print("Fecha no valida")
                                                        continue
                                                    break
                                                while True:
                                                    try:
                                                        fecha2 = input("Ingrese la fecha de corte de la cuenta (dd-mm-yyyy): ")
                                                        fecha2 = datetime.strptime(fecha2, "%d-%m-%Y").date()
                                                    except ValueError:
                                                        print("Fecha no valida")
                                                        continue
                                                    break
                                                numeroSucursal = int(''.join([str(random.randint(1, 6)) for i in range(1)]))
                                                estado = input("Ingrese el estado: ")
                                                while True:
                                                    correo = input("Ingrese el correo del Cliente: ")
                                                    if validate_email(correo):
                                                        break
                                                    else:
                                                        print("Correo no valido")
                                                        continue
                                                while True:
                                                    telefono = input("Ingrese el teléfono del Cliente (10 dígitos): ")
                                                    if telefono.isdigit() and len(telefono) == 10:
                                                        break
                                                    else:
                                                        print("El telefono debe contener 10 DÍGITOS")
                                                        continue
                                                RFC = input("Ingrese el RFC de la empresa: ")
                                                nombreEmpresa = input("Ingrese el nombre de la empresa del Cliente: ").title()
                                                rfceje = "NONE"
                                                #Creamos la cuenta
                                                cuentaCreada = cn.CuentaNomina(tipo, nombreCliente, numeroCliente, numeroCuenta, saldoCredito,fecha1.strftime("%d-%m-%Y"),fecha2.strftime("%d-%m-%Y"), numeroSucursal, estado, correo, telefono, RFC, nombreEmpresa, rfceje)
                                                cuentasT.append(cuentaCreada)
                                                print("\n", "*"*10 + "BIENVENIDO" + "*"*10)
                                                print(f"Tu nueva cuenta de nómina ha sido creada con éxito, {nombreCliente}\n")
                                                print("Estos son tus datos: \n", cuentaCreada)
                                                break
                                            case "S":
                                                print("Salinedo al menú principal...\n")
                                                break
                            case "S":
                                print("Salinedo al menú principal...\n")
                                break
            case "5": #Actualizar datos de una cuenta existente
                if not cuentasT:
                    print("\n"+"*"*10 + "Por favor, cargue el sistema de cuentas antes para poder actualizar información!" + "*"*10+"\n")
                else:
                    encontro = False  # Indica que aún no la hemos encontrado.
                    while True:
                        try:
                            numCliente = int(input("Ingrese el Número de Cliente:"))
                        except ValueError:
                            print("El número de cliente debe ser un número entero")
                            continue
                        break
                    # Comenzamos la búsqueda
                    for cuenta in cuentasT:  # Cada cuenta en la lista CuentaT...
                        if cuenta.numeroCliente == numCliente:
                            print(f"\nSe encontró la cuenta con el número de cliente indicado del tipo {cuenta.tipo}!\n") # Encontramos la cuenta y la imprimimos
                            print(cuenta)
                            while True:
                                print("¿Qué desea actualizar?")
                                print("0. Nombre")
                                print("1. Fecha de apertura")
                                print("2. Fecha de corte")
                                print("3. Número de sucursal")
                                print("4. Estado")
                                print("5. Correo")
                                print("6. Teléfono")
                                print("7. RFC personal")
                                print("8. Nombre de la empresa")
                                print("9. RFC de la empresa")
                                print("[S] Salir")
                                op = input("Escribe el número de la opción: ").upper()
                                if op not in "1234567890S" or len(op) > 1:
                                    print("Opción no válida")
                                    continue
                                else: 
                                    match op:
                                        case "0":
                                            while True:
                                                cuenta.nombreCliente = input("Ingrese el nuevo nombre del cliente: ")
                                                if cuenta.nombreCliente.isalpha():
                                                    print(f"\nEl nombre del cliente ha sido actualizado a {cuenta.nombreCliente}\n")
                                                    break
                                                else:
                                                    print("No puedes ingresar digitos o caracteres especiales!")
                                                    continue
                                        case "1":
                                            while True:
                                                try:
                                                    cuenta.fecha1 = input("Ingrese la nueva fecha de apertura de la cuenta (dd-mm-yyyy): ")
                                                    cuenta.fecha1 = datetime.strptime(cuenta.fecha1, "%d-%m-%Y").date()
                                                except ValueError:
                                                    print("Fecha no valida")
                                                    continue
                                                print(f"\nLa fecha de apertura ha sido actualizada a {cuenta.fecha1}!\n")
                                                break
                                        case "2":
                                            while True:
                                                try:
                                                    cuenta.fecha2 = input("Ingrese la nueva fecha de corte de la cuenta (dd-mm-yyyy): ")
                                                    cuenta.fecha2 = datetime.strptime(cuenta.fecha2, "%d-%m-%Y").date()
                                                except ValueError:
                                                    print("Fecha no valida")
                                                    continue
                                                print(f"La fecha de corte de la cuenta ha sido actualizada a {cuenta.fecha2}")
                                                break
                                        case "3":
                                            while True:
                                                try:
                                                    cuenta.numeroSucursal = int(input("Ingrese el nuevo número de sucursal: "))
                                                except ValueError:
                                                    print("El número de sucursal debe ser un número")
                                                    continue
                                                if cuenta.numeroSucursal <= 0 or cuenta.numeroSucursal > 6:
                                                    print("El número de sucursal debe estar entre 0 y 6")
                                                    continue
                                                else:
                                                    print(f"El número de sucursal ha sido actualizado a {cuenta.numeroSucursal}")
                                                break
                                        case "4": #ESTADO
                                            while True:
                                                cuenta.estado = input("Ingrese el nuevo estado: ")
                                                if cuenta.estado.isalpha():
                                                    print(f"El estado ha sido actualizado a {cuenta.estado}")
                                                    break
                                                else:
                                                    print("No puedes ingresar digitos o caracteres especiales!")
                                                    continue
                                        case "5":
                                            while True:
                                                correo = input("Ingrese el nuevo correo del Cliente: ")
                                                if validate_email(correo):
                                                    cuenta.correo = correo
                                                    print(f"El correo ha sido actualizado a {cuenta.correo}")
                                                    break
                                                else:
                                                    print("Correo no valido")
                                                    continue
                                        case "6":
                                            while True:
                                                cuenta.telefono = input("Ingrese el nuevo teléfono: ")
                                                if cuenta.telefono.isdigit() and len(cuenta.telefono) == 10:
                                                    print(f"El teléfono ha sido actualizado a {cuenta.telefono}")
                                                    break
                                                else:
                                                    print("El teléfono debe contener 10 DÍGITOS")
                                                    continue
                                        case "7":
                                            if cuenta.RFCpersonal == "NONE":
                                                print("Es una cuenta de cliente, no se puede actualizar el RFC personal")
                                            else:
                                                cuenta.RFC = input("Ingrese el nuevo RFC: ")
                                                print(f"El RFC ha sido actualizado a {cuenta.RFC}")
                                        case "8":
                                            if cuenta.tipo != "Nómina":
                                                print(f"Tu cuenta es de tipo {cuenta.tipo}, no se puede actualizar el nombre de la empresa")
                                            else:
                                                cuenta.nombreEmpresa = input("Ingrese el nuevo nombre de la empresa: ").title()
                                                print(f"El nombre de la empresa ha sido actualizado a {cuenta.nombreEmpresa}")
                                        case "9":
                                            if cuenta.tipo != "Nómina":
                                                print(f"Tu cuenta es de tipo {cuenta.tipo}, no se puede actualizar el RFC de la empresa")
                                            else:
                                                cuenta.rfceje = input("Ingrese el nuevo RFC de la empresa: ")
                                                print(f"El RFC de la empresa ha sido actualizado a {cuenta.rfceje}")   
                                        case "S":
                                            print("Saliendo al menú principal...\n")
                                            break        
                            encontro = True  # Indica que ya no necesito seguir buscando
                            print()
                            break  # Rompemos el ciclo for, para ya no buscar
                    if not encontro:  # Si se recorrió la lista y no encontró nada
                        print("La cuenta con numero de Cliente {} no fue encontrada".format(numCliente))
    
            case "6": #Dar de alta un nuevo empleado
              if not ejecutivos:
                    print("\n"+"*"*10 + "Por favor, cargue el sistema de ejecutivos antes para evitar perdida de información!" + "*"*10+"\n")
              else:  
                print("Ingrese los datos del nuevo empleado")
                while True:
                    try:
                        numeroEmpleado = int(input("Ingrese el número de empleado (6 dígitos): "))
                        pass
                    except ValueError:
                        print("El número de empleado debe ser un número")
                        continue
                    if len(str(numeroEmpleado)) != 6:
                        print("El número de empleado debe tener 6 dígitos")
                        continue
                    else:
                        break
                RFCe = input("Ingrese el RFC del empleado: ")
                while True:
                    nombreEmpleado = input("Ingrese el nombre del empleado: ").title()
                    if nombreEmpleado.isalpha():
                        break
                    else:
                        print("No puedes ingresar digitos o caracteres especiales!")
                        continue
                direccion = input("Ingrese la dirección del empleado: ")
                while True:
                    telefono = input("Ingrese el teléfono del empleado (10 dígitos): ")
                    if telefono.isdigit() and len(telefono) == 10:
                        break
                    else:
                        print("El teléfono debe contener 10 DÍGITOS")
                        continue
                while True:
                    try:
                        sueldoMensual = float(input("Ingrese el sueldo mensual del empleado: "))
                    except ValueError:
                        print("El sueldo mensual debe ser un número")
                        continue
                    break
                numerodeCuentas = 0
                ejecutivoCreado = e.EjecutivosCuenta(numeroEmpleado, RFCe, nombreEmpleado, direccion, telefono, sueldoMensual, numerodeCuentas)
                ejecutivos.append(ejecutivoCreado)
                print(f"Bienvenido {nombreEmpleado}!")
                print(f"Estos son tus datos:\n{ejecutivoCreado}\n")
            case "7":
                if not ejecutivos or not cuentasT:
                    print("\n"+"*"*10 + "Por favor, cargue TODO el sistema!" + "*"*10+"\n")
                else:
                    encontro = False
                    rfc = input("Escribe el RFC de Empleado: ")
                    for ejecutivo in ejecutivos:  # Cada ejecutivo en la lista Ejecutivos...
                        if ejecutivo.RFCe == rfc:
                            print("Se encontró al ejecutivo con numero de empleado {}".format(ejecutivo.numeroEmpleado))  # Encontramos la cuenta y la imprimimos
                            print(f"Estos son sus datos:\n{ejecutivo}")
                            encontro = True
                    for cuenta in cuentasT:
                        if cuenta.RFCpersonal == rfc:
                            print(f"*Cuenta activa de: {cuenta.tipo}*")
                            encontro = True
                    print()
                    if not encontro:
                        print(f"No se encontró ningún empleado con RFC: {rfc}\n")
            case "8": #Actualizar los datos de un empleado existente
                if not ejecutivos:
                    print("\n"+"*"*10 + "Por favor, cargue el sistema de ejecutivos antes para poder consultar su información!" + "*"*10+"\n")
                else:
                    encontro = False  # Indica que aún no la hemos encontrado.
                    numeroEmpleado = input("Escribe el Numero de Empleado: ")
                    # Comenzamos la búsqueda
                    for ejecutivo in ejecutivos:  # Cada cuenta en la lista CuentaT...
                        if ejecutivo.numeroEmpleado == int(numeroEmpleado):
                            print("Se encontró al ejecutivo con numero de empleado {}".format(ejecutivo.numeroEmpleado))  # Encontramos la cuenta y la imprimimos
                            print(f"Estos son tus datos:\n{ejecutivo}\n")
                            while True:
                                print("¿Qué desea actualizar?")
                                print("1. Numero de empleado")
                                print("2. RFC")
                                print("3. Nombre")
                                print("4. Dirección")
                                print("5. Teléfono")
                                print("6. Sueldo mensual")
                                print("[S] Salir")
                                op = input("Escribe el número de la opción: ").upper()
                                if op not in "1,2,3,4,5,6,S" or len(op) > 1:
                                    print("Opción no válida")
                                    continue
                                else: 
                                    match op:
                                        case "1":
                                            while True:
                                                try:
                                                    ejecutivo.numeroEmpleado = int(input("Ingrese el nuevo Numero de Empleado: "))
                                                except ValueError:
                                                    print("El número de empleado deben ser solo números")
                                                    continue
                                                if len(str(ejecutivo.numeroEmpleado)) != 6:
                                                    print("El número de empleado debe tener 6 dígitos")
                                                    continue
                                                else:
                                                    print(f"El Numero de empleado de {ejecutivo.nombreEjecutivo} se actualizó a {ejecutivo.numeroEmpleado}")
                                                break
                                        case "2":
                                            while True:
                                                ejecutivo.RFC = input("Ingrese el nuevo RFC: ")
                                                print(f"El RFC de {ejecutivo.nombreEjecutivo} se actualizó a {ejecutivo.RFC}\n")
                                                break
                                        case "3":
                                            while True:
                                                ejecutivo.nombreEjecutivo = input("Ingrese el nuevo Nombre: ")
                                                if ejecutivo.nombreEjecutivo.isalpha():
                                                    print(f"Se actualizó el nombre a {ejecutivo.nombreEjecutivo}\n")
                                                    break
                                                else:
                                                    print("No puedes ingresar digitos o caracteres especiales!")
                                                    continue
                                        case "4":
                                            while True:
                                                ejecutivo.direccion = input("Ingrese la nueva Dirección: ")
                                                print(f"Se actualizó su dirección a {ejecutivo.direccion}\n")
                                                break
                                        case "5":
                                            while True:
                                                ejecutivo.telefono = input("Ingrese el nuevo número Telefónico (10 dígitos): ")
                                                if ejecutivo.telefono.isdigit() and len(ejecutivo.telefono) == 10:
                                                    print(f"Se actualizó su teléfono a {ejecutivo.telefono}\n")
                                                    break
                                                else:
                                                    print("El teléfono debe contener 10 DÍGITOS")
                                                    continue
                                        case "6":
                                            while True:
                                                try:
                                                    ejecutivo.sueldoMensual = float(input("Ingrese el nuevo Sueldo Mensual: "))
                                                    print(f"Se actualizó su sueldo mensual a {ejecutivo.sueldoMensual}!\n")
                                                    break
                                                except ValueError:
                                                    print("El sueldo mensual debe ser un número")
                                                    continue
                                        case "S":
                                            print("Saliendo al menú principal...\n")
                                            break        
                            encontro = True  # Indica que ya no necesito seguir buscando
                            print()
                            break  # Rompemos el ciclo for, para ya no buscar
                    if not encontro:  # Si se recorrió la lista y no encontró nada
                        print("No se encontró al ejecutivo con numero de empleado {}!\n".format(numeroEmpleado))
            case "E": #Eliminar un conjunto de cuentas
                if not cuentasT:  # Comprueba si la lista está vacía
                    print("\n"+"*"*10 + "Por favor, cargue el sistema de cuentas antes para poder continuar!" + "*"*10+"\n")
                else:  # La lista no está vacía
                        #Estamos realizando otra busqueda LINEAL! del tipo O(n) Lineal BIG O notation. ESTAMOS USANDO METODOS DE ARRAYS! IMPORTANTE, es donde se estan guardando nuestros objetos, en una lista. Algo asi
                    print("Cuenta/s por eliminar")
                    print("1. Eliminar UNA cuenta por:\nNombre y Número de cuenta")
                    print("2. Eliminar UN CONJUNTO de cuentas por:\nSucursal y Estado")
                    print("[S] Salir")
                    op = input("Escribe el número de la opción: ").upper()
                    if op not in "1,2,S" or len(op) > 1:
                        print("Opción no válida")
                        continue
                    else:
                        match op:
                            case "1":
                                encontro = False  # Indica que aún no la hemos encontrado
                                #Parametro por el cual se buscará
                                nombre = input("Escribe el nombre de la persona: ")
                                numCuenta = input("Escribe el número de cuenta: ")
                                # Comenzamos la búsqueda de personas por nombre y apellidos
                                for cuenta in cuentasT:
                                    if (cuenta.nombreCliente == nombre and
                                            cuenta.numeroCuenta == numCuenta):
                                        cuentasT.remove(cuenta)  # Encontramos a la persona y la borramos
                                        print("Se ha eliminado la cuenta con nombre {} y número de cuenta {}\n".format(nombre, numCuenta))
                                        print("***No te olvides de actualizar el Sistema (A) para mantener los cambios***\n")
                                        encontro = True  # Indica que ya no necesito seguir buscando
                                        break  # Rompemos el ciclo for, para ya no buscar, ELIMINAMOS ASI UNA SOLA CUENTA
                                if not encontro:  # Si se recorrió la lista y no encontró nada
                                    print("La cuenta con nombre {} y numero de Cuenta {} no fue eliminada ya que no se encontró!\n".format(nombre, numCuenta))
                            case "2":
                                encontro = False  # Indica que aún no la hemos encontrado
                                #Parametro por el cual se buscará
                                while True:
                                    try:
                                        sucursal = int(input("Ingresa el número de la sucursal: "))
                                    except ValueError:
                                        print("El número de sucursal debe ser solo números")
                                        continue
                                    break
                                estado = input("Escribe el estado de la sucursal: ")
                                # Comenzamos la búsqueda de personas por nombre y apellidos
                                for cuenta in cuentasT:
                                    if cuenta.numeroSucursal == sucursal and cuenta.estado == estado:
                                        cuentasT.remove(cuenta)  # Encontramos a la persona y la borramos
                                        print("Se han eliminado las cuentas con sucursal {} y estado {}\n".format(sucursal, estado))
                                        print("***No te olvides de actualizar el Sistema (A) para mantener los cambios***\n")
                                        encontro = True  # Indica que ya no necesito seguir buscando
                                        #break  ELIMINAMOS ASI UN CONJUNTO DE CUENTAS
                                if not encontro:  # Si se recorrió la lista y no encontró nada
                                    print("No se encontraron cuentas con sucursal {} y estado {}!\n".format(sucursal, estado))
                                    
            case "D": #Depositar dinero (Débito/Crédito)
                if not cuentasT:  # Comprueba si la lista está vacía
                    print("\n"+"*"*10 + "Por favor, cargue el sistema de cuentas antes para poder continuar!" + "*"*10+"\n")
                else:
                    while True:
                        print("1. Depositar a cuenta DÉBITO")
                        print("2. Depositar a cuenta NÓMINA")
                        print("[S] Salir")
                        op = input("Escribe el número de la opción: ").upper()
                        if op not in "1,2,S" or len(op) > 1:
                            print("Opción no válida")
                            continue
                        else:
                            match op:
                                case "1":
                                    encontro = False  # Indica que aún no la hemos encontrado
                                    #Parametro por el cual se buscará
                                    numeroCuenta = input("Escribe el numero de Cuenta a depositar:")
                                    for cuenta in cuentasT:
                                        if cuenta.numeroCuenta == numeroCuenta:
                                            print("El numero de cuenta es válido")
                                            saldo = float(input("Ingrese el monto a depositar: "))
                                            cuenta.saldoCredito += saldo
                                            print("Se ha depositado {} a la cuenta con número {}".format(saldo, numeroCuenta))
                                            encontro = True  # Indica que ya no necesito seguir buscando
                                            break
                                    if not encontro:  # Si se recorrió la lista y no encontró nada
                                        print("No se encontró una cuenta de DÉBITO con número {}".format(numeroCuenta))
                                case "2":
                                    encontro = False  # Indica que aún no la hemos encontrado
                                    #Parametro por el cual se buscará
                                    numeroCuenta = input("Escribe el numero de Cuenta a depositar:")
                                    for cuenta in cuentasT:
                                        if cuenta.numeroCuenta == numeroCuenta:
                                            print("El numero de cuenta es válido")
                                            empresa = input("Ingrese el nombre de la empresa que realiza el deposito: ")
                                            if empresa != cuenta.nombreEmpresa:
                                                print("La empresa ingresada no puede hacer depositos a este numero de cuenta")
                                            else:
                                                saldo = float(input("Ingrese el monto a depositar: "))
                                                cuenta.saldoCredito += saldo
                                                print("Se ha depositado {} a la cuenta con número {}".format(saldo, numeroCuenta))
                                                encontro = True  # Indica que ya no necesito seguir buscando
                                    if not encontro:  # Si se recorrió la lista y no encontró nada
                                        print("No se encontró la cuenta con número {}".format(numeroCuenta))
                                case "S":
                                    print("Saliendo al menú principal...\n")
                                    break
            case "R": #Retirar dinero (Débito/Crédito)
                if not cuentasT:  # Comprueba si la lista está vacía
                    print("\n"+"*"*10 + "Por favor, cargue el sistema de cuentas antes para poder continuar!" + "*"*10+"\n")
                else:
                    while True:
                        print("1. Retirar a cuenta DÉBITO")
                        print("2. Retirar a cuenta CRÉDITO")
                        print("3. Retirar a cuenta NÓMINA")
                        print("[S] Salir")
                        op = input("Escribe el número de la opción: ").upper()
                        if op not in "1,2,3,S" or len(op) > 1:
                            print("Opción no válida")
                            continue
                        else:
                            match op:
                                case "1":
                                    encontro = False  # Indica que aún no la hemos encontrado
                                    #Parametro por el cual se buscará
                                    numeroCuenta = input("Escribe el numero de Cuenta:")
                                    for cuenta in cuentasT:
                                        if cuenta.numeroCuenta == numeroCuenta:
                                            print("El numero de cuenta es válido")
                                            print(f"Tu saldo es de {cuenta.saldoCredito}")
                                            cantidad = float(input("Ingrese el monto a retirar: "))
                                            if cantidad > cuenta.saldoCredito:
                                                print("No hay suficiente saldo en la cuenta")
                                            elif cantidad <= 0:
                                                print("No se puede retirar un monto negativo o 0")
                                            else:
                                                cuenta.saldoCredito -= cantidad
                                                print("Se ha retirado {} a la cuenta con número {}".format(cantidad, numeroCuenta))
                                                print("Tu saldo actual es de {}".format(cuenta.saldoCredito))
                                                encontro = True  # Indica que ya no necesito seguir buscando
                                    if not encontro:  # Si se recorrió la lista y no encontró nada
                                        print("No se encontró una cuenta de DÉBITO con número {}".format(numeroCuenta))
                                case "2":
                                    encontro = False  # Indica que aún no la hemos encontrado
                                    #Parametro por el cual se buscará
                                    numeroCuenta = input("Escribe el numero de Cuenta:")
                                    for cuenta in cuentasT:
                                        if cuenta.numeroCuenta == numeroCuenta:
                                            print("El numero de cuenta es válido")
                                            print(f"Tu monto de credito que puedes utilizar es de ${cuenta.saldoCredito}")
                                            try:
                                                cantidad = float(input("Ingrese el monto a retirar: "))
                                            except ValueError:
                                                print("Ingrese un número válido")
                                                break
                                            if cantidad > cuenta.saldoCredito:
                                                print("No hay suficiente crédito en la cuenta")
                                            elif cantidad <= 0:
                                                print("No se puede retirar un monto negativo o 0")
                                            else:
                                                cuenta.saldoCredito -= cantidad
                                                cuenta.creditoUtilizado += (cantidad * 0.05 + cantidad)
                                                print("Se ha retirado {} a la cuenta con número {}".format(cantidad, numeroCuenta))
                                                print("Tu saldo actual es de {}".format(cuenta.saldoCredito))
                                    encontro = True  # Indica que ya no necesito seguir buscando
                                    if not encontro:  # Si se recorrió la lista y no encontró nada
                                        print("No se encontró una cuenta de CRÉDITO con número {}".format(numeroCuenta))
                                case "3":
                                    encontro = False  # Indica que aún no la hemos encontrado
                                    #Parametro por el cual se buscará
                                    numeroCuenta = input("Escribe el numero de Cuenta:")
                                    for cuenta in cuentasT:
                                        if cuenta.numeroCuenta == numeroCuenta:
                                            print("El numero de cuenta es válido")
                                            print(f"Tu saldo es de {cuenta.saldoCredito}")
                                            cantidad = float(input("Ingrese el monto a retirar: "))
                                            if cantidad > cuenta.saldoCredito:
                                                print(f"No hay suficiente saldo en la cuenta para retirar {cantidad}")
                                            elif cantidad <= 0:
                                                print("No se puede retirar un monto negativo o 0")
                                            else:
                                                cuenta.saldoCredito -= cantidad
                                                print("Se ha retirado {} a la cuenta con número {}".format(cantidad, numeroCuenta))
                                                print("Tu saldo actual es de {}\n".format(cuenta.saldoCredito))
                                    encontro = True  # Indica que ya no necesito seguir buscando
                                    if not encontro:  # Si se recorrió la lista y no encontró nada
                                        print("No se encontró una cuenta de NÓMINA con número {}".format(numeroCuenta))
                                case "S":
                                    print("Saliendo al menú principal...\n")
                                    break
            case "A": #Actualizar el sistema
                while True:
                    print("1. Actualizar Cuentas")
                    print("2. Actualizar Ejecutivos")
                    print("[S] Salir")
                    op = input("Escribe el número de la opción: ").upper()
                    if op not in "1,2,3, S" or len(op) > 1:
                        print("Opción no válida")
                        continue
                    else: 
                        match op:
                            case "1":
                                if not cuentasT:
                                    print("\n"+"*"*10 + "Cuidado!, cargue el sistema de cuentas antes para evitar perdida de información!" + "*"*10+"\n")
                                    #Si bien podriamos preguntarle al usuario si en realidad quiere borrar todo, en prumera no hubieramos agregado al menú como borrar sistema, lo cual no es habitual. Aquí podriamos agregar un input para preguntar si quiere en realidad continuar con esa acción el cual borraria todos los datos de Cuentas, con un condicional si dice que si entonces lo dejamos pasar y que se ejecute el codigo, si desea no hacerlo pues hacemos un break, es solo una idea.
                                    break
                                archivo = "C:\\Users\\hecto\\Downloads\\Proyecto\\Cuentas.csv"
                                with open(archivo, "w", encoding="UTF8", newline="") as file:
                                    # Utilizando CSV
                                    header = ["tipoCuenta", "nombre", "numCliente", "numCuenta/Tarjeta", "saldo/Crédito", "fechaApertura", "fechaCierre", "numeroSucursal", "estado", "correo", "telefono", "RFCpersonal"]
                                    writer = csv.writer(file)
                                    # Escribir el encabezado del archivo
                                    writer.writerow(header)
                                    # Escribir múltiples líneas
                                    writer.writerows(cuentasT)
                                    print("El archivo {} se actualizó con éxito!\n".format(archivo))
                            case "2":
                                if not ejecutivos:
                                    print("\n"+"*"*10 + "Cuidado, cargue el sistema de ejecutivos antes para evitar perdida de información!" + "*"*10+"\n")
                                    break
                                archivo = "C:\\Users\\hecto\\Downloads\\Proyecto\\Ejecutivos.csv"
                                with open(archivo, "w", encoding="UTF8", newline="") as file:
                                    # Utilizando CSV
                                    header = ["numeroEmpleado", "RFC", "nombre", "direccion", "telefono", "sueldoMensual","CuentasActivas"]
                                    writer = csv.writer(file)
                                    # Escribir el encabezado del archivo
                                    writer.writerow(header)
                                    # Escribir múltiples líneas
                                    writer.writerows(ejecutivos)
                                    print("El archivo {} se guardó con éxito!\n".format(archivo))
                            case "S":
                                print("Saliendo al menú principal...\n")
                                break
            case "S": #Salir
                print("Saliendo del sistema...")
                break