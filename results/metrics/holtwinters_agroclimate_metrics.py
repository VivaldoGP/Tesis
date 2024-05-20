import os
from pathlib import PurePath
import json
import pandas as pd


coefs_path = r"C:\Users\Isai\Documents\Tesis\code\data_analysis\holtwinters\model_metadata"

df_structure = {
    'variable': [],
    'parcela_id': [],
    'r2': [],
    'mse': [],
    'rmse': [],
    'aic': []
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
            df_structure['r2'].append(coef_data['r2'])
            df_structure['mse'].append(coef_data['mse'])
            df_structure['aic'].append(coef_data['aic'])
            df_structure['rmse'].append(coef_data['rmse'])

df = pd.DataFrame(df_structure)
df.dropna(inplace=True)
df.to_csv(PurePath(r'C:\Users\Isai\Documents\Tesis\code\results\metrics', f'holtwinters_agroclimate_metrics.csv'), index=False)