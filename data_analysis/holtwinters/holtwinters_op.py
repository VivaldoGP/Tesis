import os
import pandas as pd
import json
from stats_utils.holtwinters import holtwinters
from statsmodels.tools.eval_measures import rmse, mse
from sklearn.metrics import r2_score

dirs_name = str(input('Ingrese el nombre de la carpeta donde se encuentran los results: '))
var_name = str(input('Ingrese el nombre de la variable a analizar: '))

data_dir = rf"C:\Users\Isai\Documents\Tesis\code\datos\agroclimate\{dirs_name}"
dest_dir = fr"C:\Users\Isai\Documents\Tesis\code\data_analysis\results\holtwinters\{dirs_name}"
model_metadata = fr"C:\Users\Isai\Documents\Tesis\code\data_analysis\holtwinters\model_metadata\{dirs_name}"

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

if not os.path.exists(model_metadata):
    os.makedirs(model_metadata)

for i in os.listdir(data_dir):
    df = pd.read_csv(os.path.join(data_dir, i), index_col='Fecha', parse_dates=True)
    model, model_fit = holtwinters(df[var_name], trend='add', seasonal='add', seasonal_periods=365)

    metrics = {
        'aic': model_fit.aic,
        'bic': model_fit.bic,
        'rmse': rmse(df[var_name], model_fit.fittedvalues),
        'mse': mse(df[var_name], model_fit.fittedvalues),
        'r2': r2_score(df[var_name], model_fit.fittedvalues)
    }

    with open(os.path.join(model_metadata, i.split('.')[0] + '.json'), 'w') as f:
        f.write(json.dumps(metrics, indent=4))

    df[f'{var_name}_holtwin_predicts'] = model_fit.fittedvalues
    df.to_csv(os.path.join(dest_dir, i))
