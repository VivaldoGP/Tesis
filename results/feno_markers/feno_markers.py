import os
import pandas as pd
from pathlib import PurePath

vi_var = str(input('Variable: '))
zafra = int(input('Zafra: '))

reales = rf"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\ready_to_analyze\zafra{zafra}"
modelados = rf"C:\Users\Isai\Documents\Tesis\code\data_analysis\all_vars\zafra{zafra}"

modelados_files = [(pd.read_csv(PurePath(modelados, file), parse_dates=True), int(file.split('.')[0].split('_')[1])
                    ) for file in os.listdir(modelados) if file.endswith(".csv")]
reales_files = [(pd.read_csv(PurePath(reales, file), parse_dates=True), int(file.split('.')[0].split('_')[1]))
                for file in os.listdir(reales) if file.endswith(".csv")]


zafra_df = pd.DataFrame(columns=[
    'parcela',
    'real_mean',
    'model_mean',
    'real_pos',
    'model_pos',
    'real_pos_date',
    'model_pos_date',
    'real_min',
    'model_min',
    'sos_modelo',
    'sos_real',
    'eos_modelo',
    'eos_real',
    'season_length_modelo',
    'season_length_real'
])

for i in modelados_files:
    for j in reales_files:
        if i[1] == j[1]:
            i[0]['Fecha'] = pd.to_datetime(i[0]['Fecha'])
            j[0]['Fecha'] = pd.to_datetime(j[0]['Fecha'])
            model_var = i[0][vi_var]
            real_var = j[0][f"{vi_var}_mean"]
            pos_model = model_var.max()
            pos_real = real_var.max()
            model_pos_date_loc = model_var.idxmax()
            model_pos_date = i[0].loc[model_pos_date_loc, 'Fecha']
            real_pos_date_loc = real_var.idxmax()
            real_pos_date = j[0].loc[real_pos_date_loc, 'Fecha']
            model_min = model_var.min()
            real_min = real_var.min()
            sos_modelo_loc = model_var[model_var > 0.2].idxmin()
            sos_modelo_date = i[0].loc[sos_modelo_loc, 'Fecha']
            sos_real_loc = real_var[real_var > 0.2].idxmin()
            sos_real = j[0].loc[sos_real_loc, 'Fecha']
            eos_modelo_loc = i[0]['Fecha'].idxmax()
            eos_modelo_date = i[0].loc[eos_modelo_loc, 'Fecha']
            eos_real_loc = j[0]['Fecha'].idxmax()
            eos_real_date = j[0].loc[eos_real_loc, 'Fecha']
            season_length_modelo = (eos_modelo_date - sos_modelo_date).days
            season_length_real = (eos_real_date - sos_real).days
            zafra_df = zafra_df._append({
                'parcela': i[1],
                'real_mean': real_var.mean(),
                'model_mean': model_var.mean(),
                'real_pos': pos_real,
                'model_pos': pos_model,
                'real_pos_date': real_pos_date,
                'model_pos_date': model_pos_date,
                'real_min': real_min,
                'model_min': model_min,
                'sos_modelo': sos_modelo_date,
                'eos_modelo': eos_modelo_date,
                'sos_real': sos_real,
                'eos_real': eos_real_date,
                'season_length_real': season_length_real,
                'season_length_modelo': season_length_modelo
            }, ignore_index=True)

zafra_df.to_csv(PurePath(rf"C:\Users\Isai\Documents\Tesis\code\results\feno_markers\zafra{zafra}_{vi_var}.csv"), index=False)
