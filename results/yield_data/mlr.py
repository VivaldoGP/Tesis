from pathlib import PurePath
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import math

ds = PurePath(r'C:\Users\Isai\Documents\Tesis\code\results\yield_data\zafras_modelado.csv')

exit_file = ds.name.split('.')[0] + '_mlr.csv'

df = pd.read_csv(ds)

var1 = str(input('Variable 1: '))
var2 = str(input('Variable 2: '))

model_metadata = pd.DataFrame(columns=['vars','aic', 'rsquared', 'mse', 'rmse'])

x = df[[var1, var2]]
y = df['rendimiento']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = sm.OLS(y_train, sm.add_constant(x_train)).fit()
y_pred = model.predict(sm.add_constant(x))
df[f'{var1}&{var2}_pred'] = y_pred
model_aic = model.aic
params = model.params
p_vals = model.pvalues
r2 = model.rsquared

model_metadata = model_metadata._append({'vars': f'{var1}&{var2}',
                                         'aic': model_aic,
                                         'rsquared': r2,
                                         'mse': model.mse_model,
                                         'rmse': math.sqrt(model.mse_model)},
                                        ignore_index=True)
print(model_metadata)
model_metadata.to_csv(PurePath(r'C:\Users\Isai\Documents\Tesis\code\results\yield_data\mlr\metrics', f"{var1}&{var2}_{exit_file}"), index=False)
export_df = df[['parcela', 'rendimiento', f'{var1}',f'{var2}', f'{var1}&{var2}_pred']]
export_df.to_csv(PurePath(r'C:\Users\Isai\Documents\Tesis\code\results\yield_data\mlr\predicts', f"{var1}&{var2}_{exit_file}"), index=False)
