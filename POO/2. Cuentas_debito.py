"""
Programa: Cuentas_de_débito.py
Proposito: Escribe un programa que te permita administrar Cuentas de débito.
Autor: team “THE ALGORITHM AVENGERS”
Fecha: 23/04/2024

"""
class Cuenta_debito:
    """
    Esta clase representa una cuenta de débito.

    Atributos:
    - numero_cuenta (int): El número de cuenta bancaria.
    - nombre_cliente (str): El nombre del titular de la cuenta.
    - saldo (float): El saldo actual de la cuenta.
    - nip (str): El NIP de la cuenta.

    Métodos:
    - __init__(): Constructor de la clase.
    - retirar(cantidad, nip): Retira dinero de la cuenta si el NIP es correcto y hay suficiente saldo.
    - conocer_saldo(saldo): Consulta el saldo de la cuenta.
    - depositar(cantidad): Deposita dinero en la cuenta.
    - cancelar_cuenta(nip): Cancela la cuenta si el NIP es correcto.
    - cambiar_nip(nuevo_nip, antiguo_nip): Cambia el NIP de la cuenta si el NIP antiguo es correcto.
    - __str__(): Retorna una representación en cadena de la cuenta.
    """
    def __init__(self, *params):
        """
        Constructor de la clase Cuenta_debito.

        :param params: Parámetros opcionales para inicializar la cuenta.
        Si no se proporcionan, se utilizan valores predeterminados.
        Si se proporcionan, se asignan según la cantidad de parámetros.
        El constructor puede inicializarse de las siguientes maneras:
        - Por omisión: Crea una cuenta con valores predeterminados.
        - Por parámetros: Crea una cuenta con los valores proporcionados.
        """
        if len(params) == 0: #Constructor por omision
            self.__numero_cuenta = 123456
            self.__nombre_cliente = "Larry Lovestein"
            self.__saldo = 1500
            self.__nip = 6977

        if len(params) == 4: #Constructor por parametros
            self.__numero_cuenta = params[0] if len(str(params[0])) == 6 else 123456
            self.__nombre_cliente = params[1]
            self.__saldo = params[2] if int(params[2]) > 1500 else 1500
            self.__nip = int(params[3]) if len(str(params[3])) == 4 else 6977
    #Agramos lod metodos GET y SET
    @property
    def numero_cuenta(self):
        """
        Metodo para obtener el numero de cuenta.

        :return: El numero de cuenta.
        :rtype: int
        """
        return self.__numero_cuenta

    @numero_cuenta.setter
    def numero_cuenta(self, numero):
        if len(numero) == 6:
            self.__numero_cuenta = numero
        else:
            print("El número de cuenta debe tener 16 dígitos.")

    @property
    def nombre_cliente(self):
        """
        Metodo para obtener el nombre del cliente.

        :return: El nombre del cliente.
        :rtype: str
        """
        return self.__nombre_cliente

    @nombre_cliente.setter
    def nombre_cliente(self, nombre):
        self.__nombre_cliente = nombre

    @property
    def saldo(self):
        """
        Metodo para obtener el saldo de la cuenta.

        :return: El saldo.
        :rtype: int
        """
        return self.__saldo

    @saldo.setter
    def saldo(self, cantidad):
        if cantidad >= 1500:
            self.__saldo = cantidad
        else:
            print("El saldo de la cuenta debe minímo de $1500.")
    @property
    def nip(self):
        """
        Metodo para obtener el NIP de la cuenta.

        :return: El NIP.
        :rtype: int
        """
        return self.__nip

    @nip.setter
    def nip(self, nuevo_nip):
        if len(nuevo_nip) == 4:
            self.__nip = nuevo_nip
        else:
            print("El NIP debe tener 4 dígitos.")

    def retirar(self, cantidad, nip):
        """
        Retira una cantidad de dinero de la cuenta.

        :param cantidad: La cantidad de dinero a retirar.
        :type cantidad: float
        :param nip: El NIP de la cuenta.
        :type nip: str

        El método verifica si el NIP proporcionado coincide con el de la cuenta
        y si hay suficiente saldo para realizar el retiro.
        """
        if nip != self.__nip:
            print("NIP incorrecto")
            return
        if cantidad > self.__saldo:
            print("Saldo insuficiente")
            return
        self.__saldo -= cantidad
        print("Se retiraron ${}. Nuevo saldo: ${}".format(cantidad, self.__saldo))

    def conocer_saldo(self, saldo):
        """
        Metodo para consultar el saldo de la cuenta.

        :param saldo: El saldo actual de la cuenta.
        :type saldo: float

        :return: El saldo actual de la cuenta.
        :rtype: float
        """
        return self.__saldo


    def depositar(self, cantidad):
        """
        Metodo que deposita una cantidad de dinero en la cuenta.

        :param cantidad: La cantidad de dinero a depositar.
        :type cantidad: float
        """
        self.__saldo += cantidad
        print(f"Se depositaron ${cantidad}. Nuevo saldo: ${self.__saldo}")

    def cancelar_cuenta(self, nip):
        """
        Metodo que cancela la cuenta si el NIP es correcto.

        :param nip: El NIP de la cuenta.
        :type nip: str

        Si el NIP proporcionado coincide con el de la cuenta, la cuenta se cancela.
        """
        if nip != self.__nip:
            print("NIP incorrecto")
            return
        self.__numero_cuenta = None
        self.__nombre_cliente = None
        self.__saldo = 0
        self.__nip = None
        print("Cuenta cancelada")

    def cambiar_nip(self, nuevo_nip, antiguo_nip):
        """
        Cambia el NIP de la cuenta si el NIP antiguo es correcto.

        :param nuevo_nip: El nuevo NIP de la cuenta.
        :type nuevo_nip: str
        :param antiguo_nip: El NIP antiguo de la cuenta.
        :type antiguo_nip: str
        Si el NIP antiguo proporcionado coincide con el de la cuenta, se cambia el NIP.
        """
        if antiguo_nip != self.__nip:
            print("NIP anterior incorrecto")
            return
        self.__nip = nuevo_nip
        print("NIP cambiado exitosamente")

    def __str__(self):
        """
        Método que permite imprimir una CuentaDebito en formato cadena.
        :return: La cadena en formato str
        :rtype: str
        """
        cadena = "Número de cuenta: {}\nCliente: {}\nSaldo: ${}".format(self.numero_cuenta, self.nombre_cliente, self.saldo)
        return cadena


