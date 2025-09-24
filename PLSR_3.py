import mysql.connector
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from sklearn.utils import shuffle

# ----------------------------------
# Connexió a la base de dades MySQL
# ----------------------------------
conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='DepartamentUBEB',
    database='orina_etanol',
)
cursor = conn.cursor()

query = """
    SELECT s.round, s.concentration, s.sample, m.frequency, m.real_part, m.imaginary_part
    FROM Measurements m
    JOIN Samples s ON m.sample_id = s.id
    WHERE s.round IN (1, 2, 3, 4, 5)
    AND s.concentration IN (10, 20, 30, 40, 50, 60, 70, 80, 90)
"""
cursor.execute(query)
data = cursor.fetchall()
cursor.close()
conn.close()

# -----------------------------
# Estructura de dades i neteja
# -----------------------------
concentrations = {}
for row in data:
    round_n, conc, sample, freq, real, imag = row
    key = (round_n, conc, sample)
    if key not in concentrations:
        concentrations[key] = {'frequency': [], 'real': [], 'imag': []}
    concentrations[key]['frequency'].append(freq)
    concentrations[key]['real'].append(real)
    concentrations[key]['imag'].append(imag)

# Convertir a arrays
for key in concentrations:
    for k in ['frequency', 'real', 'imag']:
        concentrations[key][k] = np.array(concentrations[key][k])

# -----------------------------
# Carrega de baseline mitjana
# -----------------------------
baseline_path = 'C:/UPC Enginyeria Biomèdica/4t curs/TFG/Orina_Etanol/cowSpectrum_baseline.csv'
baseline_df = pd.read_csv(baseline_path)
avg_magnitude_baseline = baseline_df.iloc[0].values.astype(float)

# -----------------------------
# Creació de la matriu de dades
# -----------------------------
X, y, labels = [], [], []
freq_axis_ref = None  # <- NOVETAT: guardarem l'eix de freq. real

for (round_n, conc, sample), values in concentrations.items():
    freq = values['frequency']
    real = values['real']
    imag = values['imag']
    S21 = real + 1j * imag
    mag_db = 20 * np.log10(np.abs(S21))
    mag_corr = mag_db - avg_magnitude_baseline

    # coherència de la graella de freqüències (NOVETAT)
    if freq_axis_ref is None:
        freq_axis_ref = freq.copy()
    else:
        if len(freq) != len(freq_axis_ref) or not np.allclose(freq, freq_axis_ref, atol=1e-6):
            raise ValueError("Les freqüències no coincideixen entre mostres.")

    X.append(mag_corr)
    y.append(conc)
    labels.append((round_n, conc, sample))

X = np.array(X)
y = np.array(y)
labels = np.array(labels)

# -----------------------------
# Separació training/test per concentració
# -----------------------------
train_concs = [10, 20, 30, 50, 70, 90]
test_concs = [40, 60, 80]

train_idx = np.isin(y, train_concs)
test_idx = np.isin(y, test_concs)

X_train_raw = X[train_idx];  y_train = y[train_idx]
X_test_raw  = X[test_idx];   y_test  = y[test_idx]
labels_test = labels[test_idx]

# -----------------------------
# Autoscaling usant training
# -----------------------------
mean_train = X_train_raw.mean(axis=0)
std_train  = X_train_raw.std(axis=0)
std_train[std_train == 0] = 1.0   # <- NOVETAT: evitar divisió per zero

X_train = (X_train_raw - mean_train) / std_train
X_test  = (X_test_raw  - mean_train) / std_train

# -----------------------------
# Entrenament amb LOGO-CV
# -----------------------------
max_components = 31
mse_cv = []
vip_scores_all = []
unique_concs = np.unique(y_train)

for n_comp in range(1, max_components + 1):
    fold_mse = []
    fold_vips = []
    gkf = GroupKFold(n_splits=len(unique_concs))

    for train_idx2, test_idx2 in gkf.split(X_train, y_train, y_train):
        X_t, X_v = X_train[train_idx2], X_train[test_idx2]
        y_t, y_v = y_train[train_idx2], y_train[test_idx2]

        pls = PLSRegression(n_components=n_comp)
        pls.fit(X_t, y_t)
        y_pred = pls.predict(X_v).ravel()

        mse = np.sqrt(mean_squared_error(y_v, y_pred))
        fold_mse.append(mse)

        T = pls.x_scores_; W = pls.x_weights_; Q = pls.y_loadings_
        p, h = W.shape
        s = np.sum(T ** 2, axis=0) * np.sum(Q ** 2, axis=0)
        vip = np.sqrt(p * (W ** 2 @ s.reshape(-1, 1)) / np.sum(s)).ravel()
        fold_vips.append(vip)

    mse_cv.append(np.mean(fold_mse))
    vip_scores_all.append(np.mean(fold_vips, axis=0))

plt.figure()
plt.plot(range(1, max_components + 1), mse_cv, marker='o')  # <- CORRECCIÓ: sense sqrt ni +1
plt.xlabel('Nombre de components')
plt.ylabel('RMSECV')
plt.title('Error quadràtic mitjà - LOGO CV')
plt.grid(True)
plt.show()

optimal_components = np.argmin(mse_cv) + 1
vip_optimal = vip_scores_all[optimal_components - 1]

# VIP AMB FREQ. REALS (mateix gràfic de sempre però amb eix correcte)
plt.figure(figsize=(8, 4))
freq_axis = freq_axis_ref  # <- NOVETAT
markerline, stemlines, baseline = plt.stem(freq_axis, vip_optimal, basefmt=" ")
plt.setp(markerline, 'markerfacecolor', 'skyblue')
plt.setp(stemlines, 'color', 'blue')
plt.axhline(1, color='black', linewidth=1)

vip_mask = vip_optimal > 1
highlight_regions = []
start = None
for i, v in enumerate(vip_mask):
    if v and start is None:
        start = i
    elif not v and start is not None:
        highlight_regions.append((start, i - 1))
        start = None
if start is not None:
    highlight_regions.append((start, len(vip_mask) - 1))

for start, end in highlight_regions:
    plt.axvspan(freq_axis[start], freq_axis[end], color='red', alpha=0.4)

plt.xlabel('Freqüència (Hz)')
plt.ylabel('Average VIP Score')
plt.title('Averaged VIP Scores')
plt.xlim([min(freq_axis), max(freq_axis)])
plt.ylim([0, 2])
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# MODEL FINAL I RESULTATS — AMB TOTS ELS PUNTS
# -----------------------------
pls_final = PLSRegression(n_components=optimal_components)
pls_final.fit(X_train, y_train)
y_pred_test = pls_final.predict(X_test).ravel()

rmsep = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2 = r2_score(y_test, y_pred_test)
print("\n=== RESULTATS (AMB TOTS ELS PUNTS) ===")
print(f"RMSEP (test): {rmsep:.2f}%")
print(f"R² (test): {r2:.2f}")

bias = np.mean(y_pred_test - y_test)
print(f"Biaix (pred − real): {bias:.2f}%")
loa = 1.96 * np.std(y_pred_test - y_test, ddof=1)
upper_loa = bias + loa
lower_loa = bias - loa

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_test, color='b', label='Predicted vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k-', label='Perfect Agreement')
plt.plot([min(y_test), max(y_test)], [min(y_test)+bias, max(y_test)+bias], 'r-', label='Bias')
plt.plot([min(y_test), max(y_test)], [min(y_test)+upper_loa, max(y_test)+upper_loa], 'k--', label=f'LoA = {upper_loa:.2f} (+1.96SD)')
plt.plot([min(y_test), max(y_test)], [min(y_test)+lower_loa, max(y_test)+lower_loa], 'k--', label=f'LoA = {lower_loa:.2f} (-1.96SD)')
plt.xlabel('Actual Eth. Concentration (Test) (%)')
plt.ylabel('Predicted Eth. Concentration (%)')
plt.title('Predicted vs. Actual in External Validation — TOTS')
plt.legend(loc='best'); plt.grid(True); plt.tight_layout(); 
ax = plt.gca()
x0, x1 = float(np.min(y_test)), float(np.max(y_test))
y0, y1 = float(np.min(y_pred_test)), float(np.max(y_pred_test))
ax.text(
    x0 + 0.02*(x1 - x0),
    y1 - 0.12*(y1 - y0),
    f"RMSEP: {rmsep:.2f}%\nR²: {r2:.2f}",
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
)
plt.show()


# -----------------------------
# RESULTATS — SENSE OUTLIERS (Hampel en residus del test, sense reentrenar)
# -----------------------------
residuals = y_test - y_pred_test
med = np.median(residuals)
mad = np.median(np.abs(residuals - med)) + 1e-12
z = 0.6745 * (residuals - med) / mad
keep_mask = np.abs(z) <= 3.5   # criteri Hampel (|z|>3.5 → outlier)

n_out = int(np.sum(~keep_mask))
print("\n=== RESULTATS (SENSE OUTLIERS DE TEST) ===")
print(f"Outliers detectats i exclosos del test: {n_out}")
if n_out > 0:
    print("Mostres excloses (round, conc, sample):")
    for lab in labels_test[~keep_mask]:
        print("  ", lab)

y_test_no = y_test[keep_mask]
y_pred_no = y_pred_test[keep_mask]

rmsep_no = np.sqrt(mean_squared_error(y_test_no, y_pred_no))
r2_no    = r2_score(y_test_no, y_pred_no)
print(f"RMSEP (test, sense outliers): {rmsep_no:.2f}%")
print(f"R² (test, sense outliers): {r2_no:.2f}")

bias_no = np.mean(y_pred_no - y_test_no)
print(f"Biaix (pred − real): {bias:.2f}%")
loa_no  = 1.96 * np.std(y_pred_no - y_test_no, ddof=1)
upper_loa_no = bias_no + loa_no
lower_loa_no = bias_no - loa_no

plt.figure(figsize=(6, 6))
plt.scatter(y_test_no, y_pred_no, color='b', label='Predicted vs Actual')
plt.plot([min(y_test_no), max(y_test_no)], [min(y_test_no), max(y_test_no)], 'k-', label='Perfect Agreement')
plt.plot([min(y_test_no), max(y_test_no)], [min(y_test_no)+bias_no, max(y_test_no)+bias_no], 'r-', label='Bias')
plt.plot([min(y_test_no), max(y_test_no)], [min(y_test_no)+upper_loa_no, max(y_test_no)+upper_loa_no], 'k--', label=f'LoA = {upper_loa_no:.2f} (+1.96SD)')
plt.plot([min(y_test_no), max(y_test_no)], [min(y_test_no)+lower_loa_no, max(y_test_no)+lower_loa_no], 'k--', label=f'LoA = {lower_loa_no:.2f} (-1.96SD)')
plt.xlabel('Actual Eth. Concentration (Test) (%)')
plt.ylabel('Predicted Eth. Concentration (%)')
plt.title('Predicted vs. Actual in External Validation — SENSE OUTLIERS')
plt.legend(loc='best'); 
plt.grid(True); 
plt.tight_layout(); 
ax = plt.gca()
x0, x1 = float(np.min(y_test_no)), float(np.max(y_test_no))
y0, y1 = float(np.min(y_pred_no)), float(np.max(y_pred_no))
ax.text(
    x0 + 0.02*(x1 - x0),
    y1 - 0.12*(y1 - y0),
    f"RMSEP: {rmsep_no:.2f}%\nR²: {r2_no:.2f}",
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
)
plt.show()
