import os
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import statsmodels.api as sm

from plot_utils.charts import poly_degree_aic

import matplotlib.pyplot as plt


def calc_degree(data: DataFrame, columns: list, max_degree: int = 10):
    x = data[columns[0]].values.reshape(-1, 1)
    y = data[columns[1]].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    degrees = np.arange(1, max_degree)

    df = pd.DataFrame(columns=['degree', 'params', 'aic', 'rsquared', 'mse'])
    modelos = {}

    for degree in degrees:
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        x_poly_train = poly_features.fit_transform(x_train)
        x_poly_test = poly_features.transform(x_test)
        x_poly = poly_features.transform(x)

        model = sm.OLS(y_train, sm.add_constant(x_poly_train)).fit()
        modelos[degree] = (model, poly_features)
        y_pred = model.predict(sm.add_constant(x_poly))
        # print(y_pred)
        #print(model.summary())

        df = df._append({'degree': degree,
                         'params': model.params,
                         'aic': model.aic,
                         'rsquared': model.rsquared,
                         'mse': model.mse_model}, ignore_index=True)

    return df, modelos


data_test = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\ready_to_analyze\parcela_16.csv"
parcela_id = data_test.split('_')[-1].split('.')[0]
df = pd.read_csv(data_test)

datos, modelos = calc_degree(df, ['Dia', 'ndvi_mean'])
print(datos)
print(modelos.keys())

"""
poly_degree_aic(datos, 'degree', ['aic', 'rsquared'], 'AIC vs Grado del polinomio',
                'Grado del polinomio', ['AIC', '$R^2$'], subtitle=f'Parcela {parcela_id}',
                export=True,
                export_path=r"C:/Users\Isai\Documents\Tesis\code\imagenes_tesis/aic_vs_degrees\parcela_16.jpg")
"""

'''
polys = PolynomialFeatures(degree=2, include_bias=False)
x = df['Dia'].values.reshape(-1, 1)
x_poly = polys.fit_transform(x)
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(df['Dia'], df['ndvi_mean'], color='r')
ax.plot(x, modelos[2].predict(sm.add_constant(x_poly)), color='g')
plt.tight_layout()
plt.show()
'''

x = df['Dia'].values.reshape(-1, 1)
y = df['ndvi_mean'].values

# Crear subfiguras
fig, ax = plt.subplots(3, 3, figsize=(12, 12))
ax = ax.flatten()

# Asegurarse de que haya al menos 9 modelos en 'modelos'
for degree in range(1,10):
    if degree not in modelos:
        continue

    model_info = modelos[degree]
    poly_degree = model_info[1]
    model = model_info[0]

    x_poly = poly_degree.fit_transform(x)
    y_pred = model.predict(sm.add_constant(x_poly))

    ax[degree-1].scatter(df['Dia'], df['ndvi_mean'], color='lightcoral', label='Datos reales', marker='D', s=3)
    ax[degree-1].plot(x, y_pred, label=f'Modelo (grado {degree})', linestyle='--', linewidth=0.7, color='royalblue')
    ax[degree-1].set_title(f'Grado {degree}')
    ax[degree-1].set_xlabel('Día')
    ax[degree-1].set_ylabel('ndvi_mean')
    ax[degree-1].legend(loc='lower right')

    box_text = f'$R^2$:{model.rsquared:.3f}\nAIC: {model.aic:.3f}\nMSE: {model.mse_model:.3f}'
    ax[degree - 1].text(0.9, 0.9, box_text, transform=ax[degree - 1].transAxes,
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'), fontsize=8)

plt.subplots_adjust(wspace=1, hspace=1.2)
plt.tight_layout()
plt.savefig(r"C:\Users\Isai\Documents\Tesis\code\imagenes_tesis\polys_vs_data\parcela_16.jpg", dpi=600,
            bbox_inches='tight')
plt.show()
