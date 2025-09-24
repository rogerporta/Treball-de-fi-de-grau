import mysql.connector
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import cm, colormaps
from matplotlib.colors import Normalize
from scipy.interpolate import interp1d


# Connexió a la base de dades
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='DepartamentUBEB',
    database='Piruvat2',
)

# -----------------------------
# Carrega de baseline mitjana
# -----------------------------
baseline_path = 'C:/UPC Enginyeria Biomèdica/4t curs/TFG/Orina_Etanol/cowSpectrum_baseline.csv'
baseline_df = pd.read_csv(baseline_path)
avg_magnitude_baseline = baseline_df.iloc[0].values.astype(float)

cursor = conn.cursor()

# Consulta per les mostres dels diferents rounds i concentracions
# query = """
#     SELECT s.round, s.concentration, s.sample, m.frequency, m.real_part, m.imaginary_part
#     FROM Measurements m
#     JOIN Samples s ON m.sample_id = s.id
#     WHERE s.round IN (1) 
#     AND (s.concentration = 90 OR s.concentration = 30 OR s.concentration = 80)
# """

# query = """
#     SELECT s.round, s.concentration, s.sample, m.frequency, m.real_part, m.imaginary_part
#     FROM Measurements m
#     JOIN Samples s ON m.sample_id = s.id
#     WHERE s.round IN (1, 2, 3, 4, 5)
#     AND s.concentration IN (10, 20, 30, 40, 50, 60, 70, 80, 90)
# """

# query = """
#     SELECT s.round, s.concentration, s.sample, m.frequency, m.real_part, m.imaginary_part
#     FROM Measurements m
#     JOIN Samples s ON m.sample_id = s.id
#     WHERE s.round IN (1)
#     AND s.concentration IN (10, 50, 100, 200, 300, 600, 1000) AND s.sample IN (0)
# """

query = """
    SELECT s.round, s.concentration, s.sample, m.frequency, m.real_part, m.imaginary_part
    FROM Measurements m
    JOIN Samples s ON m.sample_id = s.id
    WHERE s.round IN (1, 2, 3, 4, 5)
    AND s.concentration IN (10, 50, 100, 200, 300, 600, 1000, 1500, 2000)
"""
cursor.execute(query)
data = cursor.fetchall()

cursor.close()
conn.close()

if not data:
    raise ValueError("No s'han trobat dades per les concentracions indicades.")

# Reestructurar dades: round → concentració → mostra
concentrations = {}

for row in data:
    round_n = row[0]
    concentration = row[1]
    sample = row[2]
    frequency = row[3]
    real_part = row[4]
    imaginary_part = row[5]

    if round_n not in concentrations:
        concentrations[round_n] = {}
    if concentration not in concentrations[round_n]:
        concentrations[round_n][concentration] = {}
    if sample not in concentrations[round_n][concentration]:
        concentrations[round_n][concentration][sample] = {
            'frequency': [], 'real_part': [], 'imaginary_part': []
        }

    concentrations[round_n][concentration][sample]['frequency'].append(frequency)
    concentrations[round_n][concentration][sample]['real_part'].append(real_part)
    concentrations[round_n][concentration][sample]['imaginary_part'].append(imaginary_part)

# Convertir llistes a arrays
for round_n in concentrations:
    for conc in concentrations[round_n]:
        for sample in concentrations[round_n][conc]:
            concentrations[round_n][conc][sample]['frequency'] = np.array(concentrations[round_n][conc][sample]['frequency'])
            concentrations[round_n][conc][sample]['real_part'] = np.array(concentrations[round_n][conc][sample]['real_part'])
            concentrations[round_n][conc][sample]['imaginary_part'] = np.array(concentrations[round_n][conc][sample]['imaginary_part'])

# # Colors
# colors = {10: 'blue', 20: 'yellow', 30: 'red', 40: 'green', 50: 'orange', 60: 'gold', 70: 'purple', 80: 'brown', 90: 'skyblue'}
# Definir el colormap i la normalització segons les concentracions
colormap = colormaps['jet']
norm = Normalize(vmin=0, vmax=1800)

# Agrupem correctament: per cada concentració, acumulem totes les mostres de tots els rounds
grouped_by_conc = {}

for round_n, concs in concentrations.items():
    for conc, samples in concs.items():
        if conc not in grouped_by_conc:
            grouped_by_conc[conc] = {}
        for sample_name, sample_data in samples.items():
            # Generem un identificador únic per evitar sobreescriure mostres amb el mateix nom entre rounds
            unique_sample_name = f"{round_n}_{sample_name}"
            grouped_by_conc[conc][unique_sample_name] = sample_data

#  **Diagrama de Smith**
fig, ax = plt.subplots(figsize=(6, 6))
for conc, samples in grouped_by_conc.items():
    for sample, data in samples.items():
        ax.plot(data['real_part'], data['imaginary_part'], marker='o', linestyle='None', label=f'Concentració {conc}' if sample == list(samples.keys())[0] else "", color=colormap(norm(conc)), markersize=3)

ax.set_xlabel('Part Real')
ax.set_ylabel('Part Imaginària')
ax.set_title('Carta de Smith')
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)
ax.grid(True)

# Barra de color
sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Concentració")
plt.tight_layout()
plt.show()
plt.close()




mean_complex_by_conc = {}

for conc, samples in grouped_by_conc.items():
    real_parts = []
    imag_parts = []

    for sample, data in samples.items():
        real_parts.append(data['real_part'])     # shape (n_freq,)
        imag_parts.append(data['imaginary_part'])

    # Convertir a arrays de shape (n_samples, n_freq)
    real_parts = np.array(real_parts)
    imag_parts = np.array(imag_parts)

    # Calcular mitjanes per columna (freqüència)
    mean_real = np.mean(real_parts, axis=0)
    mean_imag = np.mean(imag_parts, axis=0)

    # Número complex mitjà
    mean_complex_by_conc[conc] = mean_real + 1j * mean_imag
fig, ax = plt.subplots(figsize=(6, 6))

for conc, s21_mean in mean_complex_by_conc.items():
    ax.plot(np.real(s21_mean), np.imag(s21_mean), label=f'Conc. {conc}',
            color=colormap(norm(conc)), linewidth=2)

ax.set_xlabel('Part Real')
ax.set_ylabel('Part Imaginària')
ax.set_title('Carta de Smith - Corbes Mitjanes per Concentració')
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)
ax.grid(True)

# Barra de color
sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Concentració")
plt.tight_layout()
plt.show()
plt.close()












#  **Magnitud de S21 vs. freqüència**
fig, ax = plt.subplots()
for conc, samples in grouped_by_conc.items():
    for sample, data in samples.items():
        frequency = np.array(data['frequency'])
        real_part = np.array(data['real_part'])
        imaginary_part = np.array(data['imaginary_part'])

        # Convertir a S21 (número complejo)
        S21_sample = real_part + 1j * imaginary_part
        
        # Calcular la magnitud en dB
        S21_magnitude = 20 * np.log10(np.abs(S21_sample))

        # Aplicar la corrección por división
        S21_magnitude_corrected = S21_magnitude
    
        ax.plot(data['frequency'], S21_magnitude_corrected, marker='o', linestyle='-', markersize=0.5, label=f'Concentració {conc}' if sample == list(samples.keys())[0] else "", color=colormap(norm(conc)))

ax.set_xlabel('Freqüència (Hz)')
ax.set_ylabel('20*log10(|S21|)')
ax.set_title('Magnitud S21 vs Freqüència')
ax.grid(True)
# Barra de color
sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Concentració")
plt.tight_layout()
plt.show()
plt.close()

# Función para encontrar la frecuencia de resonancia (mínimo de S21)
def find_resonance_frequency(frequency, S21_magnitude):
    min_index = np.argmin(S21_magnitude)  # Índice del mínimo
    return frequency[min_index], S21_magnitude[min_index]

# Diccionario para almacenar frecuencias de resonancia por concentración
resonance_frequencies = {}

fig, ax = plt.subplots()
for conc, samples in grouped_by_conc.items():
    all_magnitudes = []
    frequency = None
    
    for sample, data in samples.items():
        if frequency is None:
            frequency = np.array(data['frequency'])
        real_part = np.array(data['real_part'])
        imaginary_part = np.array(data['imaginary_part'])
        
        # Calcular magnitud en dB aplicando la baseline
        S21_sample = real_part + 1j * imaginary_part
        S21_magnitude = 20 * np.log10(np.abs(S21_sample))
        S21_magnitude_corrected = S21_magnitude
        
        all_magnitudes.append(S21_magnitude_corrected)
    
    # Calcular la media de las curvas de la misma concentración
    avg_magnitude = np.mean(all_magnitudes, axis=0)
    
    # Guardar la frecuencia de resonancia promedio
    f_res, S21_res = find_resonance_frequency(frequency, avg_magnitude)
    resonance_frequencies[conc] = f_res
    
    # Graficar la curva promedio
    ax.plot(frequency, avg_magnitude, linestyle='-', label=f'Concentració {conc}', color=colormap(norm(conc)))
    

# Mostrar el gráfico
ax.set_xlabel('Freqüència (Hz)')
ax.set_ylabel('20*log10(|S21|)')
ax.set_title('Mitjana magnitud S21 vs Freqüència')
ax.grid(True)
# Barra de color
sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Concentració")
plt.tight_layout()
plt.show()
plt.close()












#  **Magnitud de S21 vs. freqüència**
fig, ax = plt.subplots()
for conc, samples in grouped_by_conc.items():
    for sample, data in samples.items():
        frequency = np.array(data['frequency'])
        real_part = np.array(data['real_part'])
        imaginary_part = np.array(data['imaginary_part'])

        # Convertir a S21 (número complejo)
        S21_sample = real_part + 1j * imaginary_part
        
        # Calcular la magnitud en dB
        S21_magnitude = 20 * np.log10(np.abs(S21_sample))

        # Aplicar la corrección por división
        S21_magnitude_corrected = S21_magnitude - avg_magnitude_baseline
    
        ax.plot(data['frequency'], S21_magnitude_corrected, marker='o', linestyle='-', markersize=0.5, label=f'Concentració {conc}' if sample == list(samples.keys())[0] else "", color=colormap(norm(conc)))

ax.set_xlabel('Freqüència (Hz)')
ax.set_ylabel('20*log10(|S21|)')
ax.set_title('Magnitud S21 corregida vs Freqüència')
ax.grid(True)
# Barra de color
sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Concentració")
plt.tight_layout()
plt.show()
plt.close()

# Función para encontrar la frecuencia de resonancia (mínimo de S21)
def find_resonance_frequency(frequency, S21_magnitude):
    min_index = np.argmin(S21_magnitude)  # Índice del mínimo
    return frequency[min_index], S21_magnitude[min_index]

# Diccionario para almacenar frecuencias de resonancia por concentración
resonance_frequencies = {}

fig, ax = plt.subplots()
for conc, samples in grouped_by_conc.items():
    all_magnitudes = []
    frequency = None
    
    for sample, data in samples.items():
        if frequency is None:
            frequency = np.array(data['frequency'])
        real_part = np.array(data['real_part'])
        imaginary_part = np.array(data['imaginary_part'])
        
        # Calcular magnitud en dB aplicando la baseline
        S21_sample = real_part + 1j * imaginary_part
        S21_magnitude = 20 * np.log10(np.abs(S21_sample))
        S21_magnitude_corrected = S21_magnitude - avg_magnitude_baseline
        
        all_magnitudes.append(S21_magnitude_corrected)
    
    # Calcular la media de las curvas de la misma concentración
    avg_magnitude = np.mean(all_magnitudes, axis=0)
    
    # Guardar la frecuencia de resonancia promedio
    f_res, S21_res = find_resonance_frequency(frequency, avg_magnitude)
    resonance_frequencies[conc] = f_res
    
    # Graficar la curva promedio
    ax.plot(frequency, avg_magnitude, linestyle='-', label=f'Concentració {conc}', color=colormap(norm(conc)))
    

# Mostrar el gráfico
ax.set_xlabel('Freqüència (Hz)')
ax.set_ylabel('20*log10(|S21|)')
ax.set_title('Mitjana magnitud S21 corregida vs Freqüència')
ax.grid(True)
# Barra de color
sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Concentració")
plt.tight_layout()
plt.show()
plt.close()

# Imprimir frecuencias de resonancia promedio
for conc, f_avg in resonance_frequencies.items():
    print(f'Concentració {conc}: Freqüència de resonància mitjana = {f_avg:.2f} Hz')

# Diccionari per emmagatzemar resultats de mitjana i desviació
mean_std = {}

fig, ax = plt.subplots()
for conc, samples in grouped_by_conc.items():
    all_magnitudes = []
    freqs = None  # Per assegurar que totes les mostres tenen les mateixes freqüències
    
    for sample, data in samples.items():
        frequency = np.array(data['frequency'])
        real_part = np.array(data['real_part'])
        imaginary_part = np.array(data['imaginary_part'])
        
        S21_sample = real_part + 1j * imaginary_part
        S21_magnitude = 20 * np.log10(np.abs(S21_sample))
        S21_magnitude_corrected = S21_magnitude - avg_magnitude_baseline
        
        all_magnitudes.append(S21_magnitude_corrected)
        
        if freqs is None:
            freqs = frequency
    
    # Convertir llista de magnituds en array de NumPy
    all_magnitudes = np.array(all_magnitudes)
    
    # Calcular la mitjana i la desviació estàndard
    mean_magnitude = np.mean(all_magnitudes, axis=0)
    std_magnitude = np.std(all_magnitudes, axis=0)
    
    # Guardar resultats
    mean_std[conc] = {'frequencies': freqs, 'mean': mean_magnitude, 'std': std_magnitude}
    
    # Representar la mitjana amb bandes d'error
    ax.plot(freqs, mean_magnitude, linestyle='-', linewidth=2, label=f'Concentració {conc}', color=colormap(norm(conc)))
    ax.fill_between(freqs, mean_magnitude - std_magnitude, mean_magnitude + std_magnitude, color=colormap(norm(conc)), alpha=0.2)
    
ax.set_xlabel('Freqüència (Hz)')
ax.set_ylabel('20*log10(|S21|)')
ax.set_title('Mitjana i Desviació de magnitud S21 corregida vs Freqüència')
ax.grid(True)
# Barra de color
sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Concentració")
plt.tight_layout()
plt.show()
plt.close()



# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------



# --- Ajustament de les corbes per concentració ---

# --- Funció per trobar el punt mínim d'una corba ---
def find_min_point(freq, mag):
    idx = np.argmin(mag)  # Índex del valor mínim de magnitud
    return freq[idx], mag[idx]  # Retorna la freqüència i magnitud del mínim

min_points_by_conc = {}   # Diccionari per emmagatzemar el punt mínim mitjà per concentració
adjusted_curves = {}      # Diccionari per guardar les corbes ajustades (alineades)

for conc, samples in grouped_by_conc.items():
    min_points = []              # Llista per guardar els mínims de cada mostra
    adjusted_curves[conc] = {}  # Diccionari per guardar les corbes ajustades d'aquesta concentració

    for sample, data in samples.items():
        # Converteix dades a arrays NumPy
        freq = np.array(data['frequency'])
        real = np.array(data['real_part'])
        imag = np.array(data['imaginary_part'])

        # Calcula el S21 i la seva magnitud en dB
        s21 = real + 1j * imag
        mag_db = 20 * np.log10(np.abs(s21)) - avg_magnitude_baseline

        # Troba el punt mínim de la corba
        f_min, mag_min = find_min_point(freq, mag_db)
        min_points.append((f_min, mag_min))  # Desa el punt mínim

    # Calcula el punt mínim mitjà (freq i mag) per la concentració
    min_points = np.array(min_points)
    f_ref = np.mean(min_points[:, 0])
    mag_ref = np.mean(min_points[:, 1])
    min_points_by_conc[conc] = (f_ref, mag_ref)

    # Ajusta cada mostra perquè el seu mínim coincideixi amb la mitjana
    for sample, data in samples.items():
        freq = np.array(data['frequency'])
        s21 = np.array(data['real_part']) + 1j * np.array(data['imaginary_part'])
        mag_db = 20 * np.log10(np.abs(s21)) - avg_magnitude_baseline

        f_min, _ = find_min_point(freq, mag_db)
        shift = f_ref - f_min                       # Quantitat a desplaçar
        shifted_freq = freq + shift                # Nova freqüència ajustada
        interp_mag = interp1d(shifted_freq, mag_db, kind='linear', fill_value='extrapolate')
        adjusted_mag = interp_mag(freq)            # Reinterpola per mantenir la malla original

        # Desa la corba ajustada
        adjusted_curves[conc][sample] = {'frequency': freq, 'magnitude': adjusted_mag}

fig, ax = plt.subplots()
for conc, samples in adjusted_curves.items():
    for idx, (sample, data) in enumerate(samples.items()):
        # Mostra la llegenda només una vegada per concentració
        label = f'Concentració {conc}' if idx == 0 else ""
        ax.plot(data['frequency'], data['magnitude'], color=colormap(norm(conc)), alpha=0.7, label=label)

ax.set_xlabel('Freqüència (Hz)')
ax.set_ylabel('20*log10(|S21|)')
ax.set_title('Corbes ajustades per concentració')
ax.grid(True)
# Barra de color
sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Concentració")
plt.tight_layout()
plt.show()
plt.close()

mean_curves = {}  # Diccionari per guardar la mitjana per cada concentració

for conc, samples in adjusted_curves.items():
    freqs = [d['frequency'] for d in samples.values()]
    mags = [d['magnitude'] for d in samples.values()]
    
    # Defineix una malla comuna per interpolar totes les corbes
    freq_common = np.linspace(min(freqs[0]), max(freqs[0]), len(freqs[0]))
    interp_mags = np.array([np.interp(freq_common, f, m) for f, m in zip(freqs, mags)])

    # Calcula la mitjana de totes les magnituds interpolades
    mean_curves[conc] = {'frequency': freq_common, 'magnitude': np.mean(interp_mags, axis=0)}

# Gràfic de les mitjanes
fig, ax = plt.subplots()
for conc, data in mean_curves.items():
    ax.plot(data['frequency'], data['magnitude'], color=colormap(norm(conc)), linewidth=2, label=f'Mitjana {conc}')

ax.set_xlabel('Freqüència (Hz)')
ax.set_ylabel('20*log10(|S21|)')
ax.set_title('Mitjana corbes ajustades')
ax.grid(True)
# Barra de color
sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Concentració")
plt.tight_layout()
plt.show()
plt.close()

std_vals = []  # Llista amb la desviació estàndard mitjana per concentració
concs = []     # Llista de concentracions

for conc, samples in adjusted_curves.items():
    freqs = [d['frequency'] for d in samples.values()]
    mags = [d['magnitude'] for d in samples.values()]
    
    # Malla comuna per interpolar les magnituds
    freq_common = np.linspace(min(freqs[0]), max(freqs[0]), len(freqs[0]))
    interp_mags = np.array([np.interp(freq_common, f, m) for f, m in zip(freqs, mags)])

    # Calcula la desviació estàndard mitjana
    std_vals.append(np.mean(np.std(interp_mags, axis=0)))
    concs.append(conc)

# Converteix concentracions a float si és possible (per ordenar-les)
try:
    x_vals = list(map(float, concs))
except ValueError:
    x_vals = concs

# Gràfic de la desviació estàndard mitjana per concentració
plt.figure(figsize=(8, 5))
plt.plot(x_vals, std_vals, marker='o', linestyle='-', color='teal')
plt.xlabel('Concentració (mg/dL)')
plt.ylabel('Desviació Estàndard Mitjana')
plt.title('Variabilitat per Concentració')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.close()


# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------


# --- PCA ---

# Crear matriu de dades per al PCA (totes les mostres)
concentration_labels = []
data_matrix = []

for conc, samples in grouped_by_conc.items():
    for sample, values in samples.items():
        # Obtenim la magnitud corregida de cada mostra
        frequency = np.array(values['frequency'])
        real_part = np.array(values['real_part'])
        imaginary_part = np.array(values['imaginary_part'])
        S21_sample = real_part + 1j * imaginary_part
        S21_magnitude = 20 * np.log10(np.abs(S21_sample))
        S21_magnitude_corrected = S21_magnitude - avg_magnitude_baseline  # Aplicar baseline

        data_matrix.append(S21_magnitude_corrected)
        concentration_labels.append(conc)  # Guardar la concentració associada

# Convertir la matriu a array NumPy
data_matrix = np.array(data_matrix)

# Normalitzar les dades (standard scaling)
scaler = StandardScaler()
data_matrix_scaled = scaler.fit_transform(data_matrix)


# Aplicar PCA amb 3 components
pca = PCA(n_components=4)
principal_components = pca.fit_transform(data_matrix_scaled)

# Figura per a PC1 vs PC2
fig, ax = plt.subplots(figsize=(8, 6))
for i, conc in enumerate(concentration_labels):
    ax.scatter(principal_components[i, 0], principal_components[i, 1], 
               color=colormap(norm(conc)), label=f'Concentració {conc}', s=50, alpha=0.7)

ax.set_xlabel('Primera Component Principal (PC1)')
ax.set_ylabel('Segona Component Principal (PC2)')
ax.set_title('PCA: PC1 vs PC2')
ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
ax.grid(True)

sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Concentració')
plt.show()
plt.close()

# Figura per a PC1 vs PC3
fig, ax = plt.subplots(figsize=(8, 6))
for i, conc in enumerate(concentration_labels):
    ax.scatter(principal_components[i, 0], principal_components[i, 2], 
               color=colormap(norm(conc)), label=f'Concentració {conc}', s=50, alpha=0.7)

ax.set_xlabel('Primera Component Principal (PC1)')
ax.set_ylabel('Tercera Component Principal (PC3)')
ax.set_title('PCA: PC1 vs PC3')
ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
ax.grid(True)

sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Concentració')
plt.show()
plt.close()


# Figura per a PC1 vs PC4
fig, ax = plt.subplots(figsize=(8, 6))
for i, conc in enumerate(concentration_labels):
    ax.scatter(principal_components[i, 0], principal_components[i, 3], 
               color=colormap(norm(conc)), label=f'Concentració {conc}', s=50, alpha=0.7)

ax.set_xlabel('Primera Component Principal (PC1)')
ax.set_ylabel('Quarta Component Principal (PC4)')
ax.set_title('PCA: PC1 vs PC4')
ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
ax.grid(True)

sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Concentració')
plt.show()
plt.close()


# Representació en 3D de les tres primeres components principals
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

for i, conc in enumerate(concentration_labels):
    ax.scatter(principal_components[i, 0], principal_components[i, 1], principal_components[i, 3], 
               color=colormap(norm(conc)), label=f'Concentració {conc}', s=50, alpha=0.7)

ax.set_xlabel('Primera Component Principal')
ax.set_ylabel('Segona Component Principal')
ax.set_zlabel('Tercera Component Principal')
ax.set_title('PCA de totes les mostres S21 (3D)')
sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Concentració')

#plt.legend()  # Evitem la llegenda per evitar redundància de valors
plt.show()

# Explicar quanta variància explica cada component
print(f'Variància explicada per la PC1: {pca.explained_variance_ratio_[0]*100:.2f}%')
print(f'Variància explicada per la PC2: {pca.explained_variance_ratio_[1]*100:.2f}%')
print(f'Variància explicada per la PC3: {pca.explained_variance_ratio_[2]*100:.2f}%')
print(f'Variància explicada per la PC4: {pca.explained_variance_ratio_[3]*100:.2f}%')
# print(f'Variància explicada per la PC5: {pca.explained_variance_ratio_[4]*100:.2f}%')
# print(f'Variància explicada per la PC6: {pca.explained_variance_ratio_[5]*100:.2f}%')

# Normalitzar les dades (standard scaling)
scaler = StandardScaler()
data_matrix_scaled = scaler.fit_transform(data_matrix)

# Aplicar PCA amb 3 components
pca = PCA()
principal_components = pca.fit_transform(data_matrix_scaled)

# Variància explicada per cada component
explained_variance_ratio = pca.explained_variance_ratio_ * 100

# Crear el scree plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='-', color='b')
plt.xlabel('Número de Component Principal')
plt.ylabel('Variància Explicada (%)')
plt.title('Scree Plot - Variància Explicada per PC')
plt.grid(True)
plt.show()
plt.close()

# Calcular la variància acumulada
cumulative_variance = np.cumsum(pca.explained_variance_ratio_) * 100

# Representació del gràfic
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', color='g')
plt.xlabel('Número de Component Principal')
plt.ylabel('Variància Acumulada (%)')
plt.title('Variància explicada acumulada')
plt.axhline(y=90, color='r', linestyle='--', label='90% Variance')  # Línia de referència al 90%
plt.grid(True)
plt.legend()
plt.show()
plt.close()


