import random
from datetime import datetime
from validate_email import validate_email
import CD as cd
import CC as cc
import CN as cn
import E as e
import csv

class SisAdm:
    def __init__(self):
        self.__cuentasT = []
        self.__ejecutivosT = []

    @property #Decorador para GET
    def cuentasT(self):
        return self.__cuentasT
    
    @property #Decorador para GET
    def ejecutivosT(self):
        return self.__ejecutivosT

    @cuentasT.setter
    def cuentasT(self, cuentasT):
        self.__cuentasT = cuentasT

    @ejecutivosT.setter
    def ejecutivos(self, ejecutivosT):
        self.__ejecutivosT = ejecutivosT
    
    def cuentaVacia(self):
        """
        Método calculador que regresa True si esta vacia la lista cuentasT
        :return: True si esta vacia, False si no lo está
        :rtype: bool
        """
        return len(self.__cuentasT) == 0
    
    def ejecutivosVacio(self):
        """
        Método calculador que regresa True si esta vacia la lista ejecutivosT
        :return: True si esta vacia, False si no lo está
        :rtype: bool
        """
        return len(self.__ejecutivosT) == 0
    
    #MÉTODO QUE NOS PERMITE INGRESAR UN ESTADO CORRECTO A TRAVEZ DE SU ABREVIATURA
    def validarEstado(estado):
        estados = ["AGS", "BC", "BCS", "CAMP", "CHIS", "CHIH", "CDMX", "COAH", "COL", "DGO", "GTO", "GRO", "HGO", "JAL", "MEX", "MICH", "MOR", "NAY", "NL", "OAX", "PUE", "QRO", "QROO", "SLP", "SIN", "SON", "TAB", "TAMPS", "TLAX", "VER", "YUC", "ZAC"]
        while True:
            estado = input("Ingresa la abreviatura del estado de la sucursal: ")
            if estado not in estados:
                print("Ingresa una abreviatura correcta!")
                continue
            else:
                return estado
    
    #CARGAR SISTEMA DE CUENTAS Y EJECUTIVOS (OPCIÓN 1)
    def cargarSistemaCuentas(self):
        #Método calculador para cargar el sistema de Cuentas y Ejecutivos
        archivo = "C:\\Users\\hecto\\Downloads\\Proyecto\\Cuentas.csv"
        with open(archivo, encoding="UTF8", newline="") as file:
            lector = csv.reader(file)
            lector.__next__() #Esto se utiliza para saltar la primera línea del archivo CSV, que generalmente son encabezados o nombres de columnas.
            if not self.cuentaVacia():
                print("Ya hay cuentas cargadas en el sistema")
                while True:
                    resp = input("¿Deseas agregar al sistema las cuentas del archivo?(y/n): ")
                    if resp.lower() == "n":
                        print("No se agregarón las cuenta!\n")
                        self.__cuentasT = []
                        break
                    elif resp.lower() == "y":
                        print("Se agregaron las cuentas!\n")
                        break
                    else:
                        print("Elige una opción correcta! (y/n)\n")
                        continue
            else:
                print("Se han cargado las cuentas al sistema!\n")
            for fila in lector:
                if fila[0] == "Débito":
                    self.__cuentasT.append(cd.CuentaDebito(fila[0], # Tipo de cuenta
                                            fila[1],  # Nombre del cliente
                                            int(fila[2]),  # Número de cliente
                                            fila[3],  # Número de cuenta
                                            float(fila[4]),  # Saldo de la cuenta 
                                            fila[5],  # Fecha de apertura de la cuenta
                                            fila[6],  # Fecha de corte de la cuenta
                                            int(fila[7]),  # Número de sucursal
                                            fila[8],  # Correo del cliente
                                            fila[9],  # Estado de la cuenta
                                            fila[10], # Teléfono del cliente
                                            fila[11]))  # RFC del cliente
                elif fila[0] == "Crédito":
                    self.__cuentasT.append(cc.CuentaCredito(fila[0], # Tipo de cuenta
                                            fila[1],  # Nombre del cliente
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
                                            fila[13]))  # RFC del cliente)
                elif fila[0] == "Nómina":
                    self.__cuentasT.append(cn.CuentaNomina(fila[0], # Tipo de cuenta
                                            fila[1],  # Nombre del cliente
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
                                            fila[13])) # RFC de la empresa
                    
    def cargarSistemaEjecutivos(self):
        archivo = "C:\\Users\\hecto\\Downloads\\Proyecto\\Ejecutivos.csv"
        with open(archivo, encoding="UTF8", newline="") as file:
            lector = csv.reader(file)
            # Dado que hay header, vamos a saltar esa línea
            lector.__next__() #Esto se utiliza para saltar la primera línea del archivo CSV, que generalmente contiene encabezados o nombres de columnas.
            if not self.ejecutivosVacio(): #SI NO ESTA VACIA, RECUERDA QUE PONDRIAMOS UN IF NOT que quiere decir SI ESTA VACIA, pero en este caso es solo IF, osea que quiere decir que SI NO ESTA VACIA
                print("***Ya hay ejecutivos cargados en el sistema!***")
                while True:
                    resp = input("¿Deseas agregar al sistema los ejecutivos del archivo?(y/n): ")
                    if resp.lower() == "n":
                        self.__ejecutivosT = [] # No se agrega nada en pocas palabras.
                        print("No se agregarón de nuevo los datos de ejecutivos!\n")
                        break
                    elif resp.lower() == "y":
                        print("Se agregarón los datos de ejecutivos!\n")
                        break
                    else:
                        print("Elige una opción valida! (y/n)")
                        continue 
            else:
                print("Se han cargado los ejecutivos al sistema!\n")
            for fila in lector:
                if fila[0] =="E":
                    self.__ejecutivosT.append((e.EjecutivosCuenta(int(fila[1]), fila[2],  # Numero de empleado y RFC
                                                            fila[3],  # Nombre
                                                            fila[4],  # Dirección
                                                            fila[5],  # Telefono
                                                            float(fila[6]), #Sueldo
                                                            int(fila[7]))))
                    
    def cargarSistemaCompleto(self):
        cuentas = self.cargarSistemaCuentas()
        ejecutivos = self.cargarSistemaEjecutivos()
        for ejecutivo in ejecutivos:
            if ejecutivo.RFCe == cuentas.RFCpersonal:
                ejecutivos.cantidadCuentas += 1
  
    #BUSCADORES PARA MENÚ (OPCIÓN 2)           
    def buscarNombre(self):
        """
        Método para buscar cuenta por nombre
        """
        encontro = False 
        nombre = input("Escribe el nombre: ").capitalize()
        for cuenta in self.__cuentasT: 
            if cuenta.nombreCliente == nombre:
                print(cuenta)
                encontro = True 
                print()
                #Quiero TODAS las cuentas con ese nombre, por eso no colocamos break
        if not encontro:
            print("La cuenta con nombre {} no fue encontrada".format(nombre))

    def buscarNumCliente(self):
        """
        Método para buscar cuenta por número de cliente - INT
        """
        encontro = False 
        while True:
            try: 
                numCliente = int(input("Escribe el numero de cliente: "))
            except ValueError:
                print("Ingresa dígitos!")
                continue
            if len(str(numCliente)) != 8:
                print("El número de cliente consta de 8 dígitos!")
                continue
            break
        for cuenta in self.__cuentasT: 
            if cuenta.numeroCliente == numCliente:
                print(cuenta)
                encontro = True 
                print()
                #Quiero TODAS las cuentas con ese nombre, por eso no colocamos break
        if not encontro:
            print("La cuenta con numero de cliente {} no fue encontrada".format(numCliente))

    def buscarTipo(self):
        """
        Método para buscar cuenta por tipo - STR
        """
        print("Cuida los acentos!")
        encontro = False 
        tipo = input("Escribe el tipo de cuenta (Débito/Crédito/Nómina)): ").capitalize()
        for cuenta in self.__cuentasT: 
            if cuenta.tipo == tipo:
                print(cuenta)
                encontro = True 
                print()
                #Quiero TODAS las cuentas con ese nombre, por eso no colocamos break
        if not encontro:
            print("Cuentas del tipo {} no fueron encontradas".format(tipo))
            
    def buscarNumCT(self):
        """
        Método para buscar cuenta por número de cuenta o tarjeta - STR
        """
        encontro = False 
        while True:
            try: 
                numCuenta = input("Escribe el numero de cuenta/tarjeta: ")
            except ValueError:
                print("Ingresa dígitos!")
                continue
            if numCuenta.isalpha():
                print("Ingresa diígitos!")
                continue
            if len(numCuenta) != 4 and len(numCuenta) != 8:
                print("El número de cuenta consta de 8 dígitos (Núm. Tarjeta de 4)!")
                continue
            break
        for cuenta in self.__cuentasT: 
            if cuenta.numeroCuenta == numCuenta:
                print(cuenta)
                encontro = True 
                print()
                #Quiero TODAS las cuentas con ese nombre, por eso no colocamos break
        if not encontro:
            print("La cuenta con número de cuenta/tarjeta {} no fue encontrada".format(numCuenta))

    def buscarSucursal(self):
        encontro = False
        try:
            sucursal = int(input("Escribe el número de sucursal: "))
        except ValueError:
            print("Debes ingresar un digito!")
            sucursal = int(input("Escribe el número de sucursal: "))
        if len(str(sucursal)) != 1:
            print("Ingresa un solo dígito!")
            sucursal = int(input("Escribe el número de sucursal: "))
        for cuenta in self.__cuentasT:
            if cuenta.numeroSucursal == sucursal:
                print(cuenta)
                encontro = True 
                print()
        if not encontro:  
            print("No se encontraron cuentas con sucursal número {}!\n".format(sucursal))

    def buscarRFCempleado(self):
        encontro = False
        RFCeje = input("Escribe el RFC del ejecutivo: ")
        for cuenta in self.__cuentasT:
            if cuenta.RFCpersonal == RFCeje:
                print(cuenta)
                encontro = True  
                print()
        if not encontro:  
            print("No se encontraron cuentas con RFC {}\n".format(RFCeje))

    def buscarRFCempresa(self):
        encontro = False
        RFCempresa = input("Escribe el RFC de la empresa: ")
        for cuenta in self.__cuentasT:
            if cuenta.RFC == RFCempresa:
                print(cuenta)
                encontro = True  
                print()
        if not encontro:  
            print("No se encontraron cuentas relacionadas a la empresa con RFC {}\n".format(RFCempresa))

    def buscarEstado(self):
        encontro = False
        estado = self.validarEstado()
        for cuenta in self.__cuentasT:
            if cuenta.estado == estado:
                print(cuenta)  # Encontramos la cuenta y la imprimimos
                encontro = True  # Indica que ya no necesito seguir buscando
                print()
        if not encontro:  # Si se recorrió la lista y no encontró nada...
            print("No se encontraron cuentas con estado {}\n".format(estado))

    #CREACIÓN DE CUENTAS POR CLIENTE O EJECUTIVO (OPCIÓN 4)
    def cuentaDebitoEje(self):
        """
        Método para creación de cuentas del tipo Débito para ejecutivos
        """
        encontro = False
        rfcbuscar = input("Escribe el RFC del ejecutivo: ")
        for empleado in self.__ejecutivosT:
            if empleado.RFCe == rfcbuscar:
                encontro = True
                print(f"Bienvenido, {empleado.nombreEjecutivo}")
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
                estado = self.validarEstado()
                while True:
                    correo = input("Ingresa correo electrónico: ")
                    if validate_email(correo):
                        break
                    else:
                        print("Correo no valido")
                        continue
                telefono = empleado.telefono
                rfceje = empleado.RFCe
                #Creamos la cuenta
                cuentaCreada = cd.CuentaDebito(tipo, nombreEjecutivo, numeroEjecutivo, numeroCuenta, saldoCredito, fecha1, fecha2, numeroSucursal, estado, correo, telefono, rfceje)
                self.__cuentasT.append(cuentaCreada)
                print("\n", "*"*10 + "BIENVENIDO" + "*"*10)
                print(f"Tu nueva cuenta de débito ha sido creada con éxito, {nombreEjecutivo}\n")
                empleado.cantidadCuentas += 1 #SE AGREGA AL NUMERO DE CUENTAS QUE TIENE +1
                print("Estos son tus datos: \n", cuentaCreada)
        if not encontro:  # Si se recorrió la lista y no encontró nada
            print("No se encontró al ejecutivo con RFC => {}".format(rfcbuscar))

    def cuentaCreditoEje(self):
        """
        Método para creación de cuentas del tipo Crédito para ejecutivos
        """
        encontro = False
        rfcbuscar = input("Escribe el RFC del ejecutivo: ")
        for empleado in self.__cuentasT:
            if empleado.RFCe == rfcbuscar:
                encontro = True
                print(f"Bienvenido, {empleado.nombreEjecutivo}")
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
                    fecha1 = input("Ingrese la fecha de apertura (dd-mm-yyyy): ")
                    try:
                        fecha1 = datetime.strptime(fecha1, "%d-%m-%Y").date()
                    except ValueError:
                        print("Fecha no valida")
                        continue
                    fecha1 = fecha1.strftime("%d-%m-%Y")
                    break
                while True:
                    fecha2 = input("Ingrese la fecha de corte (dd-mm-yyyy): ")
                    try:
                        fecha2 = datetime.strptime(fecha2, "%d-%m-%Y").date()
                    except ValueError:
                        print("Fecha no valida")
                        continue
                    fecha2 = fecha2.strftime("%d-%m-%Y")
                    break
                numeroSucursal = int(''.join([str(random.randint(1, 6)) for i in range(1)]))
                estado = self.validarEstado()
                while True:
                    correo = input("Ingrese el correo del cliente: ")
                    if validate_email(correo):
                        break
                    else:
                        print("Correo no valido")
                        continue
                telefono = empleado.telefono
                while True:
                    creditoUtilizado = float(input("Ingrese el importe ucredito: "))
                    if creditoUtilizado < 0:
                        print("El crédito utilizao no puede ser menor a 0")
                        continue
                    else: 
                        break
                while True:
                    fechaVencimiento = input("Ingrese la fecha de vencredito (dd-mm-yyyy): ")
                    try:
                        fechaVencimiento = datetime.strptime(fechaVencimiento, "%d-%m-%Y").date()
                    except ValueError:
                        print("Fecha no valida")
                        continue
                    fechaVencimiento = fechaVencimiento.strftime("%d-%m-%Y")
                    break
                rfceje = empleado.RFCe
                #Creamos la cuenta
                cuentaCreada = cc.CuentaCredito(tipo, nombreEjecutivo, numeroEjecutivo, numeroCuenta, saldoCredito, creditoUtilizado,fecha1,fecha2,numeroSucursal, estado, correo, telefono, rfceje)
                self.__cuentasT.append(cuentaCreada)
                empleado.cantidadCuentas += 1
                print("\n", "*"*10 + "BIENVENIDO" + "*"*10)
                print(f"Tu nueva cuenta de crédito ha sido cread{nombreEjecutivo}\n")
                print("Estos son tus datos: \n", cuentaCreada)
        if not encontro:  # Si se recorrió la lista y no encontró nada
            print("No se encontró al ejecutivo con RFC => {}".format(rfcbuscar))
        else:
            print("\n"+"*"*10 + "Por favor, cargue el sistema de cuentas antes para evitar perdida de información!" + "*"*10+"\n")
    
    def cuentaNominaEje(self):
        """
        Método para creación de cuentas del tipo Nómina para ejecutivos
        """
        encontro = False
        rfcbuscar = input("Escribe el RFC del ejecutivo: ")
        for empleado in self.__cuentasT:
            if empleado.RFCe == rfcbuscar:
                encontro = True
                print(f"Bienvenido, {empleado.nombreEjecutivo}")
                print("\nInstrucción: Ingrese datos solicitados...\n")
                tipo = "Nómina"
                nombreEjecutivo = empleado.nombreEjecutivo
                numeroEjecutivo = int(''.join([str(random.randint(0, 9)) for i in range(8)]))
                numeroCuenta = str(int(''.join([str(random.randint(0, 9)) for i in range(4)])))
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
                        fecha1 = input("Ingrese la fecha de apertura(dd-mm-yyyy): ")
                        fecha1 = datetime.strptime(fecha1, "%d-%m-%Y").date()
                    except ValueError:
                        print("Fecha no valida")
                        continue
                    fecha1 = fecha1.strftime("%d-%m-%Y")
                    break
                while True:
                    try:
                        fecha2 = input("Ingrese la fecha de corte(dd-mm-yyyy): ")
                        fecha2 = datetime.strptime(fecha2, "%d-%m-%Y").date()
                    except ValueError:
                        print("Fecha no valida")
                        continue
                    fecha2 = fecha2.strftime("%d-%m-%Y")
                    break
                numeroSucursal = int(''.join([str(random.randint(1, 6)) (1)]))
                estado = input("Ingrese el estado: ")
                while True:
                    correo = input("Ingrese el correo del Ejecutivo: ")
                    if validate_email(correo):
                        break
                    else:
                        print("Correo no valido")
                        continue
                telefono = empleado.telefono
                RFCempresa = input("Ingrese el RFC de la empresa: ")
                nombreEmpresa = input("Ingrese el nombre de la empresa d").title()
                rfceje = empleado.RFCe
                #Creamos la cuenta
                cuentaCreada = cn.CuentaNomina(tipo, numeroEjecutivo, nombreEjecutivo, numeroCuenta, saldoCredito,fecha1, fecha2, numeroSucursal, estado, correo, telefono, RFCempresa, nombreEmpresa, rfceje)
                self.cuentasT.append(cuentaCreada)
                empleado.cantidadCuentas += 1
                print("\n", "*"*10 + "BIENVENIDO" + "*"*10)
                print(f"Tu nueva cuenta de nómina ha sido crea{nombreEjecutivo}\n")
                print("Estos son tus datos: \n", cuentaCreada)
                break
        if not encontro:  # Si se recorrió la lista y no encontró nada
            print("No se encontró al ejecutivo con RFC => {}".format(rfcbuscar))
        else:
            print("\n"+"*"*10 + "Por favor, cargue el sistema de cuentas antes para evitar perdida de información!" + "*"*10+"\n")

    #ACTUALIZAR DATOS DE UNA CUENTA EXISTENTE (OPCIÓN 5)
    def cuentaPorNum(self):
        '''
        Funciona como buscador para la cuenta a actualizar
        Devuelve un objeto -> cuenta encontrada'''
        encontro = False  
        while True:
            try:
                numCliente = int(input("Ingrese el Número de Cliente:"))
            except ValueError:
                print("El número de cliente debe ser un número entero")
                continue
            break
        for cuenta in self.cuentasT:
            if cuenta.numeroCliente == numCliente:
                print(f"\nSe encontró la cuenta con el número de cliente indicado del tipo {cuenta.tipo}!\n")
                encontro = True
                return cuenta
        if not encontro:
            print("No se encontró la cuenta a actualizar con número {}".format(numCliente))

    def actualizarNombre(self):
        cuenta = self.cuentaPorNum()
        cuenta.nombreCliente = input("Ingrese el nuevo nombre del cliente: ") # Aquí se realiza el cambio 
        if cuenta.nombreCliente.isalpha():
            print(f"\nEl nombre del cliente ha sido actualizado a {cuenta.nombreCliente}\n")
        else:
            print("No puedes ingresar digitos o caracteres especiales!")

    def crearEjecutivo(self):
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
        self.ejecutivosT.append(ejecutivoCreado)
        print(f"Bienvenido {nombreEmpleado}!")
        print(f"Estos son tus datos:\n{ejecutivoCreado}\n")
    
    def consultarEjecutivo(self):
        encontro = False
        rfc = input("Escribe el RFC de Empleado: ")
        for ejecutivo in self.ejecutivosT:  # Cada ejecutivo en la lista Ejecutivos...
            if ejecutivo.RFCe == rfc:
                print("Se encontró al ejecutivo con numero de empleado {}".format(ejecutivo.numeroEmpleado))  # Encontramos la cuenta y la imprimimos
                print(f"Estos son sus datos:\n{ejecutivo}\n")
                encontro = True
                return ejecutivo
        if not encontro:
            print(f"No se encontró ningún empleado con RFC: {rfc}\n")

    def actualizarNumEmpleado(self):
        ejecutivo = self.consultarEjecutivo() #Retorna un objeto, estamos sobre nombrando al objeto con el identidicador "ejecutivo"
        while True:
            try:
                numEmpleado = int(input("Ingrese el nuevo número de Empleado: "))
            except ValueError:
                print("Ingresa solo dígitos!")
                continue
            if len(str(numEmpleado)) != 6:
                print("El número de empleado debe tener 6 dígitos")
                continue
            ejecutivo.numeroEmpleado = numEmpleado
            print(f"El número del ejecutivo {ejecutivo.nombreEjecutivo} se cambió!")   
            break        

    #ELIMINAR CUENTAS (OPCIÓN E)
    def eliminarCuenta(self):
        encontro = False 
        nombre = input("Escribe el nombre de la persona: ")
        numCuenta = input("Escribe el número de cuenta: ")
        for cuenta in self.cuentasT:
            if cuenta.nombreCliente == nombre and cuenta.numeroCuenta == numCuenta:
                self.cuentasT.remove(cuenta) # Remove for arrays
                print("Se ha eliminado la cuenta con nombre {} y número de cuenta {}\n".format(nombre, numCuenta))
                print("***No te olvides de actualizar el Sistema (A) para mantener los cambios***\n")
                encontro = True
                break  # Rompemos el ciclo for, para ya no buscar, ELIMINAMOS ASI UNA SOLA CUENTA
        if not encontro: 
            print("La cuenta con nombre {} y numero de Cuenta {} NO fue eliminada ya que NO se encontró!\n".format(nombre, numCuenta))

    def eliminarCuentas(self):
        encontro = False 
        while True:
            try:
                sucursal = int(input("Ingresa el número de la sucursal: "))
            except ValueError:
                print("El número de sucursal debe ser solo números")
                continue
            break
        estado = input("Escribe el estado de la sucursal: ")
        for cuenta in self.cuentasT:
            if cuenta.numeroSucursal == sucursal and cuenta.estado == estado:
                self.cuentasT.remove(cuenta)  # Encontramos a la persona y la borramos
                print("Se han eliminado las cuentas con sucursal {} y estado {}\n".format(sucursal, estado))
                print("***No te olvides de actualizar el Sistema (A) para mantener los cambios***\n")
                encontro = True  
        if not encontro:
            print("No se encontraron cuentas con sucursal {} y estado {}!\n".format(sucursal, estado))

    def actualizarCuentas(self):
        archivo = "C:\\Users\\hecto\\Downloads\\Proyecto\\Cuentas.csv"
        nombreArchivo = "Cuentas"
        with open(archivo, "w", encoding="UTF8", newline="") as file:
            header = ["tipoCuenta", "nombre", "numCliente", "numCuenta/Tarjeta", "saldo/Crédito", "fechaApertura", "fechaCierre", "numeroSucursal", "estado", "correo", "telefono", "RFCpersonal"]
            writer = csv.writer(file)
            writer.writerow(header)
            # Escribir múltiples líneas
            writer.writerows(self.cuentasT)
            return nombreArchivo

    def actualizarEjecutivos(self):
        archivo = "C:\\Users\\hecto\\Downloads\\Proyecto\\Ejecutivos.csv"
        nombreArchivo = "Ejecutivos"
        with open(archivo, "w", encoding="UTF8", newline="") as file:
            # Utilizando CSV
            header = ["numeroEmpleado", "RFC", "nombre", "direccion", "telefono", "sueldoMensual","CuentasActivas"]
            writer = csv.writer(file)
            # Escribir el encabezado del archivo
            writer.writerow(header)
            # Escribir múltiples líneas
            writer.writerow(self.ejecutivosT)
            return nombreArchivo

    def actualizarAmbos(self):
        #Función para actualizar tras cualquier cambio, si bien sería la actualización de Cuentas y Ejecutivos a su vez
        self.actualizarCuentas()
        self.actualizarEjecutivos()