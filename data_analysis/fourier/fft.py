import os
import pandas as pd
import json
from sklearn.metrics import mean_squared_error, r2_score

from stats_utils.fourier import fft, adjust_fft_curve, sinusoidal

data_dir = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\evapotranspiration"
dest_dir = r"C:\Users\Isai\Documents\Tesis\code\data_analysis\datos\fourier\evapotranspiration"
params_dir = r"C:\Users\Isai\Documents\Tesis\code\data_analysis\fourier\parametros\evapotranspiration"

for i in os.listdir(data_dir):
    print(i)
    data = pd.read_csv(os.path.join(data_dir, i))
    data['Fecha'] = pd.to_datetime(data['Fecha'])
    data['Dia'] = (data['Fecha'] - data['Fecha'].min()).dt.days + 1
    fft_results = fft(data, 0.1, ['Fecha', 'Evapotranspiration'])
    data['Reconstruida'] = fft_results['senal_reconstruida'].real
    parametros_optimizados, covarianzas = adjust_fft_curve(data, ['Fecha', 'Evapotranspiration', 'Dia'], fft_results)
    model_data = {
        'A': parametros_optimizados[0],
        'omega': parametros_optimizados[1],
        'phi': parametros_optimizados[2],
        'offset': parametros_optimizados[3]
    }

    ajustados = sinusoidal(data['Dia'], *parametros_optimizados)
    data['Ajustados'] = ajustados
    mse = mean_squared_error(data['Evapotranspiration'], ajustados)
    r2 = r2_score(data['Evapotranspiration'], ajustados)

    model_data['mse'] = mse
    model_data['r2'] = r2

    data.to_csv(os.path.join(dest_dir, i), index=False)
    with open(os.path.join(params_dir, i.split('.')[0] + '.json'), 'w') as f:
        json.dump(model_data, f)
