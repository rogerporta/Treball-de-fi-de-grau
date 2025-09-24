import mysql.connector
import numpy as np
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
    database='Piruvat',
)
cursor = conn.cursor()

# Baseline (Round 1, Concentration 0, Sample 0)
query_baseline = """
    SELECT m.frequency, m.real_part, m.imaginary_part
    FROM Measurements m
    JOIN Samples s ON m.sample_id = s.id
    WHERE s.round = 1 AND s.concentration = 00 AND s.sample = 0
"""
cursor.execute(query_baseline)
baseline_data = cursor.fetchall()

if not baseline_data:
    raise ValueError("No es van trobar dades de baseline.")

baseline_freq = np.array([row[0] for row in baseline_data])
baseline_real = np.array([row[1] for row in baseline_data])
baseline_imag = np.array([row[2] for row in baseline_data])
S21_baseline = baseline_real + 1j * baseline_imag

# Definir el colormap i la normalització segons les concentracions
colormap = colormaps['jet']
norm = Normalize(vmin=0, vmax=350)

# Calcular magnitud en dB
mag_db = 20 * np.log10(np.abs(S21_baseline))

# Ordenar per freqüència (per si les dades arriben desordenades)
ord_idx = np.argsort(baseline_freq)
freq_sorted = baseline_freq[ord_idx] / 1e9  # passem a GHz per claredat
mag_db_sorted = mag_db[ord_idx]

# Gràfic
plt.figure(figsize=(9,5))
plt.plot(freq_sorted, mag_db_sorted, lw=1.5, color=colormap(norm(0)))
plt.xlabel("Freqüència (Hz)")
plt.ylabel("|S21| (dB)")
plt.title("Baseline magnitud S21 vs Freqüència")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()




# Connexió a la base de dades
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='DepartamentUBEB',
    database='Vials_orina',
)
cursor = conn.cursor()

# Baseline (Round 1, Concentration 0, Sample 0)
query = """
    SELECT s.round, s.concentration, s.sample, m.frequency, m.real_part, m.imaginary_part
    FROM Measurements m
    JOIN Samples s ON m.sample_id = s.id
    WHERE s.round IN (0)
    AND s.concentration IN (00)
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


# Definir el colormap i la normalització segons les concentracions
colormap = colormaps['jet']
norm = Normalize(vmin=0, vmax=100)

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


# Definir el colormap i la normalització segons les concentracions
colormap = colormaps['jet']
norm = Normalize(vmin=0, vmax=100)

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

        ax.plot(data['frequency'], S21_magnitude, marker='o', linestyle='-', markersize=0.5, label=f'Concentració {conc}' if sample == list(samples.keys())[0] else "", color=colormap(norm(conc)))

ax.set_xlabel('Freqüència (Hz)')
ax.set_ylabel('20*log10(|S21|)')
ax.set_title('Baseline magnitud S21 vs Freqüència')
ax.grid(True)
# Barra de color
sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Concentració")
plt.tight_layout()
plt.show()
plt.close()




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Carrega de baseline mitjana
# -----------------------------
baseline_path = 'C:/UPC Enginyeria Biomèdica/4t curs/TFG/Orina_Etanol/cowSpectrum_baseline.csv'
baseline_df = pd.read_csv(baseline_path)
avg_magnitude_baseline = baseline_df.iloc[0].values.astype(float)

# -----------------------------
# Vector de freqüències (201 punts de 1.6 a 3 GHz)
# -----------------------------
freq = np.linspace(1.6, 3.0, len(avg_magnitude_baseline))  # GHz

# -----------------------------
# Gràfic magnitud vs freq
# -----------------------------
fig, ax = plt.subplots()
ax.plot(freq, avg_magnitude_baseline, lw=1.5, color=colormap(norm(0)))
ax.set_xlabel('Freqüència (Hz)')
ax.set_ylabel('20*log10(|S21|)')  # suposem que ja està en dB
ax.set_title("Baseline mitjana Magnitud S21 AsLS vs Freqüència")
ax.grid(True, alpha=0.3)

# Barra de color
sm = cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Concentració")
plt.tight_layout()
plt.show()
plt.close()

