"""
Visualizador de Campos Electromagnéticos en Tiempo Real
Aplicación basada en la Ley de Faraday
"""

import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow


def main():
    # Iniciar la aplicación PyQt
    app = QApplication(sys.argv)

    # Crear y mostrar la ventana principal
    window = MainWindow()
    window.show()

    # Ejecutar el bucle principal de la aplicación
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()