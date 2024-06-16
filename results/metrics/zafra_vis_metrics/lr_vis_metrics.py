import os
from pathlib import PurePath
import json
import pandas as pd
import math

zafra = int(input('Zafra: '))

coefs_path = rf"C:\Users\Isai\Documents\Tesis\code\data_analysis\linear_reg\coeficientes\zafra{zafra}"

df_structure = {
    'indice': [],
    'parcela_id': [],
    'a': [],
    'b': [],
    'c': [],
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
            # print(coef_data, parcela_id, coef)
            df_structure['indice'].append(coef)
            df_structure['parcela_id'].append(parcela_id)
            df_structure['a'].append(coef_data[1]['params'][2])
            df_structure['b'].append(coef_data[1]['params'][1])
            df_structure['c'].append(coef_data[1]['params'][0])
            df_structure['r2'].append(coef_data[1]['rsquared'])
            df_structure['mse'].append(coef_data[1]['mse'])

df = pd.DataFrame(df_structure)
df['rmse'] = df['mse'].apply(math.sqrt)
df.dropna(inplace=True)
df.to_csv(PurePath(r'C:\Users\Isai\Documents\Tesis\code\results\metrics\zafra_vis_metrics', f'zafra{zafra}_vis_metrics.csv'), index=False)
