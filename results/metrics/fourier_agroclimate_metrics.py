import os
from pathlib import PurePath
import json
import pandas as pd
import math

coefs_path = r"C:\Users\Isai\Documents\Tesis\code\data_analysis\fourier\parametros"

df_structure = {
    'variable': [],
    'parcela_id': [],
    'A': [],
    'omega': [],
    'phi': [],
    'offset': [],
    'r2': [],
    'mse': []
}

for coef in os.listdir(coefs_path):
    coef_vi_path = os.path.join(coefs_path, coef)
    for coefs in os.listdir(coef_vi_path):
        parcela_id = coefs.split('.')[0].split('_')[1]
        coef_file = os.path.join(coef_vi_path, coefs)
        with open(coef_file, encoding='utf-8') as f:
            coef_data = json.load(f)
            print(coef_data, parcela_id, coef)
            df_structure['variable'].append(coef)
            df_structure['parcela_id'].append(parcela_id)
            df_structure['A'].append(coef_data['A'])
            df_structure['omega'].append(coef_data['omega'])
            df_structure['phi'].append(coef_data['phi'])
            df_structure['offset'].append(coef_data['offset'])
            df_structure['r2'].append(coef_data['r2'])
            df_structure['mse'].append(coef_data['mse'])

df = pd.DataFrame(df_structure)
df['rmse'] = df['mse'].apply(math.sqrt)
df.dropna(inplace=True)
df.to_csv(PurePath(r'C:\Users\Isai\Documents\Tesis\code\results\metrics', f'fourier_agroclimate_metrics.csv'), index=False)