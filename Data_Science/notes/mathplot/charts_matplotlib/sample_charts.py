#ESTO ES UN MÓDULO, puedes ejecutarlo en main.py

import matplotlib.pyplot as plt #Uso de pyplot

def generate_PIE_chart():
    labels = ["A","B","C"]
    values = [100,200,200]
    #Paara generar la grafica, obtenemos la figura y las cordenadas desde matplotlib
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels)
    #Podriamos indicarle despues que queremos ver la gráfica con plt.show() pero en esta ocasion queremos generar una imagen para que el programa no se atore aqui simplemente 

    #Le indicaremos que genere la grafica y lo guarde en un archivo como tal
    plt.savefig("pie.png")
    plt.close()

def generate_bar_chart(labels, values):
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    plt.savefig('bar.png')
    plt.close()
