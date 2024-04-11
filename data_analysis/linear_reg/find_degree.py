import os

from json import loads, dumps
import pandas as pd
import statsmodels.api as sm

from plot_utils.charts import poly_degree_aic

import matplotlib.pyplot as plt

from stats_utils.regression_models import linear_reg_model

pd.set_option('display.max_rows', None)  # Muestra todas las filas
pd.set_option('display.max_columns', None)

parcela = int(input('Ingrese el id de la parcela: '))
variable = str(input('Ingrese la variable a analizar: '))
zafra = int(input('Ingrese el año de la zafra: '))

data_test = rf"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\ready_to_analyze\zafra{zafra}\parcela_{parcela}.csv"

coef_folder = rf"C:\Users\Isai\Documents\Tesis\code\data_analysis\linear_reg\coeficientes\zafra{zafra}"
variable_folder_coef = os.path.join(coef_folder, variable)
if not os.path.exists(variable_folder_coef):
    os.makedirs(variable_folder_coef)

aic_img_folder = rf"C:\Users\Isai\Documents\Tesis\code\imagenes_tesis\aic_vs_degrees\zafra{zafra}"
variable_folder_aic = os.path.join(aic_img_folder, variable)
if not os.path.exists(variable_folder_aic):
    os.makedirs(variable_folder_aic)

model_predicts_folder = fr"C:\Users\Isai\Documents\Tesis\code\data_analysis\datos\model_predicts\zafra{zafra}"
variable_folder_predicts = os.path.join(model_predicts_folder, variable)
if not os.path.exists(variable_folder_predicts):
    os.makedirs(variable_folder_predicts)

polys_img_folder = fr"C:\Users\Isai\Documents\Tesis\code\imagenes_tesis\polys_vs_data\zafra{zafra}"
variable_folder_polys = os.path.join(polys_img_folder, variable)
if not os.path.exists(variable_folder_polys):
    os.makedirs(variable_folder_polys)

parcela_id = data_test.split('_')[-1].split('.')[0]
df = pd.read_csv(data_test)

metadata, modelos = linear_reg_model(df, ['dia', variable], max_degree=7)
print(metadata)
meta_json = loads(metadata.to_json(orient='records'))
meta_to_export = dumps(meta_json, indent=4)

with open(os.path.join(variable_folder_coef, rf"parcela_{parcela}.json"), 'w') as f:
    f.write(meta_to_export)
print(modelos.keys())

poly_degree_aic(metadata, 'degree', ['aic', 'rsquared'], 'AIC vs Grado del polinomio',
                'Grado del polinomio', ['AIC', '$R^2$'], subtitle=f'Parcela {parcela_id}',
                export=True,
                export_path=os.path.join(variable_folder_aic, rf"parcela_{parcela}.jpg"))

x = df['dia'].values.reshape(-1, 1)
y = df[variable].values

# Crear subfiguras
fig, ax = plt.subplots(3, 2, figsize=(18, 12))
fig.suptitle(f'Parcela {parcela_id}', fontsize=16)
ax = ax.flatten()

# Asegurarse de que haya al menos 9 modelos en 'modelos'
for degree in range(1, 7):
    if degree not in modelos:
        continue

    model_info = modelos[degree]
    poly_degree = model_info[1]
    model = model_info[0]
    print(model, poly_degree)

    x_poly = poly_degree.fit_transform(x)
    y_pred = model.predict(sm.add_constant(x_poly))
    df[f'{variable}_pred_{degree}'] = y_pred

    ax[degree - 1].scatter(df['dia'], df[variable], color='lightcoral', label='Datos reales', marker='D', s=3)
    ax[degree - 1].plot(x, y_pred, label=f'Modelo (grado {degree})', linestyle='--', linewidth=0.7, color='royalblue')
    ax[degree - 1].set_title(f'Grado {degree}')
    ax[degree - 1].set_xlabel('Día')
    ax[degree - 1].set_ylabel(variable.split('_')[0].upper())
    ax[degree - 1].legend(loc='lower right')

    box_text = f'$R^2$: {model.rsquared:.3f}\nAIC: {model.aic:.3f}\nMSE: {model.mse_model:.3f}'
    ax[degree - 1].text(0.9, 0.9, box_text, transform=ax[degree - 1].transAxes,
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'), fontsize=8)

print(df)
df.to_csv(os.path.join(variable_folder_predicts, rf"parcela_{parcela}.csv"), index=False)
plt.subplots_adjust(hspace=1.2, wspace=1)

plt.tight_layout()
# plt.suptitle(f'Parcela {parcela_id}', fontsize=10)
plt.savefig(os.path.join(variable_folder_polys, rf"parcela_{parcela}.jpg"), dpi=600,
            bbox_inches='tight')
plt.show()
