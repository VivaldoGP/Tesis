import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm
import math

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

main_path = r"data/zafras_modelado.csv"

ds = pd.read_csv(main_path)

var_name = []
r2 = []
p_values = []
mse = []

bye_cols = ['ndvi_max_date', 'gndvi_max_date']
for column in bye_cols:
    if column in ds.columns:
        ds.drop(columns=[column], inplace=True)

y = ds['rendimiento'].values
for col in ds.drop(columns=['parcela', 'rendimiento']).columns:
    var_name.append(col)
    # print(f'Analizando {col}')
    if col == 'msi_mean':
        x = ds[col].values.reshape(-1, 1)
        poly = PolynomialFeatures(degree=2)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        x_train = poly.fit_transform(x_train)
        model = sm.OLS(y_train, sm.add_constant(x_train)).fit()
        y_pred = model.predict(sm.add_constant(poly.fit_transform(x)))
        ds[f'{col}_pred'] = y_pred
        r2.append(model.rsquared)
        p_values.append(model.pvalues)
        mse.append(model.mse_model)
        # print(model.summary(), model.pvalues)
    else:
        x = ds[col].values.reshape(-1, 1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model = sm.OLS(y_train, sm.add_constant(x_train)).fit()
        y_pred = model.predict(sm.add_constant(x))
        ds[f'{col}_pred'] = y_pred
        r2.append(model.rsquared)
        p_values.append(model.pvalues)
        mse.append(model.mse_model)
        # print(model.summary(), model.pvalues)

results = pd.DataFrame({'Variable': var_name, 'R2': r2, 'P_values': p_values, 'MSE': mse})
results['RMSE'] = results['MSE'].apply(lambda b: math.sqrt(b))
print(results.sort_values(by='R2', ascending=False))
results.to_csv(r"score/zafras_modelado.csv", index=False)
ds.to_csv(r"predicts/zafras_modelado.csv", index=False)