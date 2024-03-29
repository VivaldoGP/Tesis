import numpy as np
from pandas import DataFrame, Series
from scipy.optimize import curve_fit


def fft(data: DataFrame | Series, freq_threshold: float,
        column_data: list):
    transformada = np.fft.fft(data[column_data[1]])

    n = len(data)
    freqs = np.fft.fftfreq(n)
    amplitudes = np.abs(transformada)
    freqs_dominantes = freqs[np.argsort(amplitudes)[::-1]]

    umbral_frecuencia = freq_threshold
    transformada_filtrada = transformada.copy()
    transformada_filtrada[np.abs(freqs) > umbral_frecuencia] = 0

    senal_reconstruida = np.fft.ifft(transformada_filtrada)

    fft_results = {
        'transformada': transformada,
        'freqs': freqs,
        'amplitudes': amplitudes,
        'freqs_dominantes': freqs_dominantes,
        'transformada_filtrada': transformada_filtrada,
        'senal_reconstruida': senal_reconstruida
    }

    return fft_results


def sinusoidal(t, A, omega, phi, offset):
    return A * np.sin(omega * t + phi) + offset


def adjust_fft_curve(data: DataFrame | Series, column_data: list,
                     fft_results: dict):
    amplitude_initial = max(data[column_data[1]]) - min(data[column_data[1]])
    omega_initial = 2 * np.pi / 365
    phi_initial = 0
    offset_initial = np.mean(data[column_data[1]])
    initial_parameters = [amplitude_initial, omega_initial, phi_initial, offset_initial]

    parametros_optimizados, covarianzas = curve_fit(sinusoidal, data[column_data[2]], fft_results['senal_reconstruida'].real, p0=initial_parameters)

    return parametros_optimizados, covarianzas
