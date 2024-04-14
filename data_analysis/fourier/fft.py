import os
import pandas as pd
import json
from sklearn.metrics import mean_squared_error, r2_score

from stats_utils.fourier import fft, adjust_fft_curve, sinusoidal

dirs_name = str(input('Ingrese el nombre de la carpeta donde se encuentran los datos: '))
var_name = str(input('Ingrese el nombre de la variable a analizar: '))

data_dir = rf"C:\Users\Isai\Documents\Tesis\code\datos\agroclimate\{dirs_name}"
dest_dir = fr"C:\Users\Isai\Documents\Tesis\code\data_analysis\datos\fourier\{dirs_name}"
params_dir = fr"C:\Users\Isai\Documents\Tesis\code\data_analysis\fourier\parametros\{dirs_name}"

for i in os.listdir(data_dir):
    print(i)
    data = pd.read_csv(os.path.join(data_dir, i))
    data['Fecha'] = pd.to_datetime(data['Fecha'])
    data['dia'] = (data['Fecha'] - data['Fecha'].min()).dt.days + 1
    fft_results = fft(data, 0.1, ['Fecha', f'{var_name}'])
    data[f'reconstruida_{var_name}'] = fft_results['senal_reconstruida'].real
    parametros_optimizados, covarianzas = adjust_fft_curve(data, ['Fecha', f'{var_name}', 'dia'], fft_results)
    model_data = {
        'A': parametros_optimizados[0],
        'omega': parametros_optimizados[1],
        'phi': parametros_optimizados[2],
        'offset': parametros_optimizados[3]
    }

    ajustados = sinusoidal(data['dia'], *parametros_optimizados)
    data[f'ajustados_{var_name}'] = ajustados
    mse = mean_squared_error(data[f'{var_name}'], ajustados)
    r2 = r2_score(data[f'{var_name}'], ajustados)

    model_data['mse'] = mse
    model_data['r2'] = r2

    data.to_csv(os.path.join(dest_dir, i), index=False)
    with open(os.path.join(params_dir, i.split('.')[0] + '.json'), 'w') as f:
        json.dump(model_data, f, indent=4)
