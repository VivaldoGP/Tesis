import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from matplotlib.dates import DateFormatter, MonthLocator
from scipy.optimize import curve_fit


data = pd.read_csv(r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\evapotranspiration\et_parcela_5.csv")
data['num_index'] = date2num(data['Fecha'])
data['Fecha'] = pd.to_datetime(data['Fecha'])

transformada = np.fft.fft(data['Evapotranspiration'])

total_dias = len(data)
frecuencias = np.fft.fftfreq(total_dias)
amplitudes = np.abs(transformada)
frecuencias_dominantes = frecuencias[np.argsort(amplitudes)[::-1]]

umbral_frecuencia = 0.1  # Umbral de frecuencia para filtrar
transformada_filtrada = transformada.copy()
transformada_filtrada[np.abs(frecuencias) > umbral_frecuencia] = 0

senal_reconstruida = np.fft.ifft(transformada_filtrada)


def sinusoidal(t, A, omega, phi, offset):
    return A * np.sin(omega * t + phi) + offset


# Parámetros iniciales para el ajuste
amplitud_inicial = max(data['Evapotranspiration']) - min(data['Evapotranspiration'])
omega_inicial = 2 * np.pi / 365  # Frecuencia anual (365 días)
phi_inicial = 0
offset_inicial = np.mean(data['Evapotranspiration'])
parametros_iniciales = [amplitud_inicial, omega_inicial, phi_inicial, offset_inicial]

# Realizar ajuste de curva
parametros_optimizados, covarianzas = curve_fit(sinusoidal, data['num_index'], senal_reconstruida.real, p0=parametros_iniciales)

# Obtener parámetros óptimos
amplitud_optima, omega_optima, phi_optima, offset_optimo = parametros_optimizados

# Crear función ajustada
temperatura_ajustada = sinusoidal(data['num_index'], amplitud_optima, omega_optima, phi_optima, offset_optimo)

from sklearn.metrics import mean_squared_error, r2_score

# Calcular el error cuadrático medio (MSE)
mse = mean_squared_error(data['Evapotranspiration'], temperatura_ajustada)

# Calcular el coeficiente de determinación (R-cuadrado)
r2 = r2_score(data['Evapotranspiration'], temperatura_ajustada)

print("Error cuadrático medio (MSE):", mse)
print("Coeficiente de determinación (R^2):", r2)
print(data)


fig, ax = plt.subplots(figsize=(10, 6))
ax.xaxis.set_major_locator(MonthLocator(interval=4))
ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
plt.plot(data['Fecha'], data['Evapotranspiration'], label='Señal Original', color='firebrick', linewidth=1)
plt.plot(data['Fecha'], senal_reconstruida.real, label='Señal Reconstruida', linestyle='-.', color='gold',
         linewidth=1)
plt.plot(data['Fecha'], temperatura_ajustada, label='Señal Ajustada', linestyle='-.', color='navy',
         linewidth=1)
box_text = f'$R^2$: {r2:.3f}\nMSE: {mse:.3f}'
ax.text(0.92, 0.8, box_text, transform=ax.transAxes,
        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'), fontsize=8)
plt.xlabel('Fecha')
plt.ylabel('Evapotranspiración de referencia (mm/día)')
plt.title('Análisis de Fourier de Evapotranspiración de referencia')
plt.legend()
plt.grid(True)
plt.show()