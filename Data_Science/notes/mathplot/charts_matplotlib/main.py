#Script que prueba el modulo "Sample" para generar charts
import sample_charts as sch

def run():
    """
    ACTUALIZA o CREA en formato png la gráfica de pastel o circular al ejecutarse
    """
    sch.generate_PIE_chart()

if __name__ == "__main__":
    run()

    labels = ["A", "B", "C"]
    values = [1,2,3]
    sch.generate_bar_chart(labels, values)