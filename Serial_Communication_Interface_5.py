import sys
import csv
import time
import serial
import serial.tools.list_ports
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QComboBox, QTextEdit, QWidget, QFileDialog, QLineEdit, QMessageBox
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class SerialThread(QThread):
    data_received = pyqtSignal(str)

    def __init__(self, serial_port):
        super().__init__()
        self.serial_port = serial_port
        self._running = True

    def run(self):
        while self._running and self.serial_port.is_open:
            try:
                line = self.serial_port.readline().decode('utf-8').strip()
                if line:
                    self.data_received.emit(line)
            except Exception as e:
                print(f"Error reading data: {e}")

    def stop(self):
        self._running = False
        self.serial_port.close()


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)


class SerialApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Serial Communication Interface')
        self.setGeometry(100, 100, 900, 600)

        self.serial_port = None
        self.serial_thread = None
        self.data = []
        self.scanning = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.send_scan_command)

        self.combobox = QComboBox()
        self.refresh_button = QPushButton('Refresh COM Ports')
        self.connect_button = QPushButton('Connect')
        self.start_button = QPushButton('START')
        self.pause_button = QPushButton('PAUSE')
        self.record_button = QPushButton('RECORD')
        self.save_button = QPushButton('SAVE TO CSV')
        self.send_button = QPushButton('SEND COMMAND')
        self.command_textbox = QTextEdit()
        self.response_textbox = QTextEdit()
        self.plot_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.start_freq = QLineEdit('1600000000')
        self.stop_freq = QLineEdit('3000000000')
        self.points = QLineEdit('201')

        self.layout_setup()
        self.signal_setup()
        self.refresh_com_ports()

    def layout_setup(self):
        main_layout = QVBoxLayout()
        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel('Select COM Port:'))
        top_layout.addWidget(self.combobox)
        top_layout.addWidget(self.refresh_button)
        top_layout.addWidget(self.connect_button)

        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel('Start frequency (Hz):'))
        freq_layout.addWidget(self.start_freq)
        freq_layout.addWidget(QLabel('Stop frequency (Hz):'))
        freq_layout.addWidget(self.stop_freq)
        freq_layout.addWidget(QLabel('Points:'))
        freq_layout.addWidget(self.points)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.record_button)
        button_layout.addWidget(self.save_button)

        command_layout = QVBoxLayout()
        command_layout.addWidget(QLabel('Command:'))
        command_layout.addWidget(self.command_textbox)
        command_layout.addWidget(self.send_button)

        main_layout.addLayout(top_layout)
        main_layout.addLayout(freq_layout)
        main_layout.addLayout(button_layout)
        main_layout.addLayout(command_layout)
        main_layout.addWidget(QLabel('Response:'))
        main_layout.addWidget(self.response_textbox)
        main_layout.addWidget(self.plot_canvas)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def signal_setup(self):
        self.refresh_button.clicked.connect(self.refresh_com_ports)
        self.connect_button.clicked.connect(self.connect_serial_port)
        self.start_button.clicked.connect(self.start_scan)
        self.pause_button.clicked.connect(self.pause_scan)
        self.record_button.clicked.connect(self.record_data)
        self.save_button.clicked.connect(self.save_data)
        self.send_button.clicked.connect(self.send_command)

    def refresh_com_ports(self):
        self.combobox.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.combobox.addItem(port.device)

    def connect_serial_port(self):
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()

        selected_port = self.combobox.currentText()
        if selected_port:
            try:
                self.serial_port = serial.Serial(selected_port, 9600, timeout=1)
                self.serial_thread = SerialThread(self.serial_port)
                self.serial_thread.data_received.connect(self.display_response)
                self.serial_thread.start()
                self.response_textbox.append(f"Connected to {selected_port}")
            except serial.SerialException as e:
                self.response_textbox.append(f"Error: {e}")

    def send_scan_command(self):
        try:
            start_freq = int(self.start_freq.text())
            stop_freq = int(self.stop_freq.text())
            points = int(self.points.text())

            # Validaciones manuales
            if not (11 <= points <= 201):
                QMessageBox.warning(self, "Error", "El número de puntos debe estar entre 11 y 201.")
                return

            if not (50000 <= start_freq <= 3000000000):
                QMessageBox.warning(self, "Error", "La frecuencia de inicio debe estar entre 50kHz y 3GHz.")
                return

            if not (50000 <= stop_freq <= 3000000000):
                QMessageBox.warning(self, "Error", "La frecuencia de detención debe estar entre 50kHz y 3GHz.")
                return

            if start_freq >= stop_freq:
                QMessageBox.warning(self, "Error", "La frecuencia de inicio debe ser menor que la de detención.")
                return

            if self.serial_port and self.serial_port.is_open:
                command = f"scan {self.start_freq.text()} {self.stop_freq.text()} {self.points.text()} 5"
                self.data = [] # Clear data before starting a new scan
                self.serial_port.write((command + '\n').encode('utf-8'))
        except ValueError:
                QMessageBox.warning(self, "Error", "Por favor, ingrese valores numéricos válidos.")
    
    def start_scan(self):
        self.data = []
        self.scanning = True
        self.timer.start(22000)

    def pause_scan(self):
        self.scanning = False
        self.timer.stop()

    def record_data(self):
        if self.data:
            self.response_textbox.append("Data recorded.")

    def save_data(self):
        if not self.data:
            self.response_textbox.append("No data to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Data", "", "CSV Files (*.csv)")
        if file_path:
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Frequency (Hz)', 'Real Part', 'Imaginary Part'])
                writer.writerows(self.data)
            self.response_textbox.append(f"Data saved to {file_path}")

    def send_command(self):
        if self.serial_port and self.serial_port.is_open:
            command = self.command_textbox.toPlainText()
            self.data = []
            self.serial_port.write((command + '\n').encode('utf-8'))

    def display_response(self, data):
        self.response_textbox.append(data)
        fields = data.split()
        if len(fields) == 3:
            timestamp, real_part, imag_part = fields
            try:
                timestamp = float(timestamp)
                real_part = float(real_part)
                imag_part = float(imag_part)

                # Store data for plotting and saving
                self.data.append((timestamp, real_part, imag_part))

                # Plot the absolute value of the complex number
                # self.update_plot()
            except ValueError as e:
                print(f"Error processing data: {e}")
        # Comprobar si hemos recibido todos los datos esperados
        if len(self.data) >= int(self.points.text()):  # Esperamos tantos puntos como el usuario especificó
            self.update_plot()  # Graficamos solo cuando hemos recibido todos los datos

    def clear_plot(self):
        """Clear the plot and reset data."""
        self.data = []  # Clear stored data
        self.plot_canvas.ax.clear()  # Clear plot
        self.plot_canvas.ax.set_xlabel('Freq (MHz)')
        self.plot_canvas.ax.set_ylabel('|Complex Value|')
        self.plot_canvas.ax.set_title('Complex Magnitude')
        self.plot_canvas.figure.tight_layout()
        self.plot_canvas.draw()
        # self.response_textbox.append("Plot cleared for new 'scan' command.")

    def update_plot(self):
        """Update the plot with new data."""
        timestamps = [row[0]/1e6 for row in self.data]
        abs_values = [20*np.log10(np.abs(complex(row[1], row[2]))) for row in self.data]

        self.plot_canvas.ax.clear()
        self.plot_canvas.ax.plot(timestamps, abs_values, 'b-')
        self.plot_canvas.ax.set_xlabel('Freq (MHz)')
        self.plot_canvas.ax.set_ylabel('|Complex Value|')
        self.plot_canvas.ax.set_title('Complex Magnitude')
        self.plot_canvas.figure.tight_layout()
        self.plot_canvas.draw()
    
    def closeEvent(self, event):
        if self.serial_thread:
            self.serial_thread.stop()
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SerialApp()
    window.show()
    sys.exit(app.exec_())
