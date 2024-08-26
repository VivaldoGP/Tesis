from pathlib import PurePath
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import math
import itertools
import json
from os import makedirs

var_type = input('Tipo de variable (max o mean): ')
var_num = int(input('Número de variables: '))

ds = PurePath(r'data/zafra2122_2223_r.csv')
df = pd.read_csv(ds)

max_cols = []
mean_cols = []

for i in df.columns:
    if 'max' in i:
        max_cols.append(i)
    elif 'mean' in i:
        mean_cols.append(i)
print(len(max_cols), len(mean_cols))

# Combinaciones de variables
max_comb = list(itertools.combinations(max_cols, var_num))
mean_comb = list(itertools.combinations(mean_cols, var_num))

if var_type == 'max':
    i = 1
    for comb in max_comb:
        x = df[list(comb)]
        y = df['rendimiento']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        model = sm.OLS(y_train, sm.add_constant(x_train)).fit()
        # print(f'Comb {i}: {model.summary()}')
        # noinspection PyTypeChecker
        summary_dict = {
            "coefs": model.params.to_dict(),
            "p_values": model.pvalues.to_dict(),
            "rsquared": model.rsquared,
            "aic": model.aic,
            "fvalue": model.fvalue,
            "f_pvalue": model.f_pvalue,
            "mse_model": model.mse_model,
            "rmse_model": math.sqrt(model.mse_model)
        }
        summary_json = json.dumps(summary_dict, indent=4)
        filepath_score = PurePath(r'score_mlr', ds.stem, var_type, str(var_num))
        makedirs(filepath_score, exist_ok=True)
        with open(PurePath(filepath_score, f'comb_{i}.json'), 'w') as f:
            f.write(summary_json)

        # predicciones
        y_pred = model.predict(sm.add_constant(x))
        df['pred'] = y_pred
        filepath_predict = PurePath(r'predicts_mlr', ds.stem, var_type, str(var_num))
        makedirs(filepath_predict, exist_ok=True)
        df.to_csv(PurePath(filepath_predict, f'comb_{i}.csv'), index=False)

        i += 1
elif var_type == 'mean':
    i = 1
    for comb in mean_comb:
        x = df[list(comb)]
        y = df['rendimiento']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        model = sm.OLS(y_train, sm.add_constant(x_train)).fit()
        # print(f'Comb {i}: {model.summary()}')
        # noinspection PyTypeChecker
        summary_dict = {
            "coefs": model.params.to_dict(),
            "p_values": model.pvalues.to_dict(),
            "rsquared": model.rsquared,
            "aic": model.aic,
            "fvalue": model.fvalue,
            "f_pvalue": model.f_pvalue,
            "mse_model": model.mse_model,
            "rmse_model": math.sqrt(model.mse_model)
        }
        summary_json = json.dumps(summary_dict, indent=4)
        filepath = PurePath(r'score_mlr', ds.stem, var_type, str(var_num))
        makedirs(filepath, exist_ok=True)
        with open(PurePath(filepath, f'comb_{i}.json'), 'w') as f:
            f.write(summary_json)

            # predicciones
            y_pred = model.predict(sm.add_constant(x))
            df['pred'] = y_pred
            filepath_predict = PurePath(r'predicts_mlr', ds.stem, var_type, str(var_num))
            makedirs(filepath_predict, exist_ok=True)
            df.to_csv(PurePath(filepath_predict, f'comb_{i}.csv'), index=False)
        i += 1
