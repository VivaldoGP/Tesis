import os
import pandas as pd
import json
from stats_utils.holtwinters import holtwinters
from statsmodels.tools.eval_measures import rmse, mse
from sklearn.metrics import r2_score

parcela = int(input('Ingrese el id de la parcela: '))
variable = str(input('Ingrese la variable a analizar: '))
export_metadata = str(input('¿Desea exportar los metadatos? (y/n): '))
export_predictions = str(input('¿Desea exportar las predicciones? (y/n): '))
folder = 'relative_humidity'

data_dir = rf"C:\Users\Isai\Documents\Tesis\code\datos\agroclimate\{folder}\parcela_{parcela}.csv"
des_dir = rf"C:\Users\Isai\Documents\Tesis\code\data_analysis\datos\holtwinters\{folder}"
model_metadata = rf"C:\Users\Isai\Documents\Tesis\code\data_analysis\holtwinters\model_metadata\{folder}"

if not os.path.exists(des_dir):
    os.makedirs(des_dir)

if not os.path.exists(model_metadata):
    os.makedirs(model_metadata)

parcela_id = data_dir.split('_')[-1].split('.')[0]

df = pd.read_csv(data_dir, index_col='Fecha', parse_dates=True)

model, model_fit = holtwinters(df[variable], trend='add', seasonal='add', seasonal_periods=365)

metrics = {
    'aic': model_fit.aic,
    'bic': model_fit.bic,
    'rmse': rmse(df[variable], model_fit.fittedvalues),
    'mse': mse(df[variable], model_fit.fittedvalues),
    'r2': r2_score(df[variable], model_fit.fittedvalues)
}

if export_metadata == 'y':
    with open(os.path.join(model_metadata, f'parcela_{parcela_id}.json'), 'w') as f:
        f.write(json.dumps(metrics))
elif export_metadata == 'n':
    pass

df['holtwinters_predicts'] = model_fit.fittedvalues

if export_predictions == 'y':
    df.to_csv(os.path.join(des_dir, f'parcela_{parcela_id}.csv'))
elif export_predictions == 'n':
    pass
