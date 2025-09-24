import os
import re
import mysql.connector
import pandas as pd
from datetime import datetime

# Configuración de la base de datos MySQL
DB_CONFIG = {
    "host": "localhost",
    "user": "root",  # Cambia esto si tienes otro usuario
    "password": "DepartamentUBEB",  # Coloca tu contraseña
    "database": "Vials_orina"  # Cambia esto si tu base de datos tiene otro nombre
}

# Ruta donde están los archivos CSV (ajusta la ruta a tu carpeta)
FOLDER_PATH = r"C:\UPC Enginyeria Biomèdica\4t curs\TFG\90 vials orina"  # Modifica según tu escritorio

# Expresión regular para extraer datos del nombre del archivo
FILENAME_REGEX = re.compile(r"r(\d+)_c([\d\.]+)_s(\d+).csv")

# Conectar a la base de datos MySQL
conn = mysql.connector.connect(**DB_CONFIG)
cursor = conn.cursor()

# Recorrer todos los archivos CSV en la carpeta
for file in os.listdir(FOLDER_PATH):
    if file.endswith(".csv"):
        match = FILENAME_REGEX.match(file)
        if match:
            round_num, concentration, sample = match.groups()
            round_num = int(round_num)
            concentration = float(concentration)
            sample = int(sample)

            # Obtener fecha y hora real de creación del archivo
            file_path = os.path.join(FOLDER_PATH, file)
            creation_timestamp = os.path.getctime(file_path)
            creation_datetime = datetime.fromtimestamp(creation_timestamp).strftime('%Y-%m-%d %H:%M:%S')

            # Buscar si ya existe la muestra en Samples
            cursor.execute("SELECT id FROM Samples WHERE name = %s", (file,))
            result = cursor.fetchone()

            if result:
                sample_id = result[0]
                # 🗑 Eliminar mediciones antiguas asociadas a esta muestra
                cursor.execute("DELETE FROM Measurements WHERE sample_id = %s", (sample_id,))
                # Actualizar la fecha de la muestra en Samples
                cursor.execute("UPDATE Samples SET date = %s WHERE id = %s", (creation_datetime, sample_id))
                print(f"Datos de {file} actualizados correctamente.")
            else:
                # Insertar nueva muestra si no existe
                cursor.execute("""
                    INSERT INTO Samples (name, round, concentration, sample, date)
                    VALUES (%s, %s, %s, %s, %s)
                """, (file, round_num, concentration, sample, creation_datetime))
                sample_id = cursor.lastrowid
                print(f"Nueva muestra {file} insertada.")

            # Leer datos del CSV con pandas
            df = pd.read_csv(file_path)

            # Insertar nuevas mediciones
            for _, row in df.iterrows():
                cursor.execute("""
                    INSERT INTO Measurements (sample_id, frequency, real_part, imaginary_part)
                    VALUES (%s, %s, %s, %s);
                """, (sample_id, row["Frequency (Hz)"], row["Real Part"], row["Imaginary Part"]))

# Confirmar cambios y cerrar conexión
conn.commit()
cursor.close()
conn.close()

print("Importación finalizada con éxito.")
