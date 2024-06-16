import os

from json import dumps, loads

import pandas as pd
import numpy as np

import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

from plot_utils.charts import poly_degree_aic
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

focus_vars = ['ndvi_mean', 'gndvi_mean', 'msi_mean', 'ndmi_mean', 'cire_mean', 'ndre1_mean']

zafra = int(input('Ingrese el año de la zafra: '))
export_metadata = bool(input('Exportar metadata? (True/False): '))

data_folder = rf"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\ready_to_analyze\zafra{zafra}"

coef_folder = rf"C:\Users\Isai\Documents\Tesis\code\data_analysis\linear_reg\coeficientes\zafra{zafra}"
aic_img_folder = rf"C:\Users\Isai\Documents\Tesis\code\tesis_img\aic_vs_degrees\zafra{zafra}"
model_predicts_folder = fr"C:\Users\Isai\Documents\Tesis\code\data_analysis\results\model_predicts\zafra{zafra}"
polys_img_folder = fr"C:\Users\Isai\Documents\Tesis\code\tesis_img\polys_vs_data\zafra{zafra}"

for j in os.listdir(data_folder):
    ds = pd.read_csv(os.path.join(data_folder, j), parse_dates=['Fecha'])
    parcela_id = j.split('_')[-1].split('.')[0]
    for variable in ds.columns:
        if variable in focus_vars:
            # print(f'Parcela {parcela_id} - Variable {variable}')
            variable_folder_coef = os.path.join(coef_folder, variable)
            if not os.path.exists(variable_folder_coef):
                os.makedirs(variable_folder_coef)

            variable_folder_aic = os.path.join(aic_img_folder, variable)
            if not os.path.exists(variable_folder_aic):
                os.makedirs(variable_folder_aic)

            variable_folder_predicts = os.path.join(model_predicts_folder, variable)
            if not os.path.exists(variable_folder_predicts):
                os.makedirs(variable_folder_predicts)

            variable_folder_polys = os.path.join(polys_img_folder, variable)
            if not os.path.exists(variable_folder_polys):
                os.makedirs(variable_folder_polys)

            # Aquí empieza el modelo
            x_data = ds['dia'].values.reshape(-1, 1)
            y_data = ds[variable].values

            x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

            degrees = np.arange(1, 7)

            model_metadata = []
            models = {}

            for degree in degrees:
                poly_features = PolynomialFeatures(degree=degree, include_bias=False)
                x_poly_train = poly_features.fit_transform(x_train)
                x_poly_test = poly_features.transform(x_test)
                x_poly = poly_features.transform(x_data)

                model = sm.OLS(y_train, sm.add_constant(x_poly_train)).fit()
                models[degree] = (model, poly_features)
                y_pred = model.predict(sm.add_constant(x_poly))

                model_metadata.append({'degree': degree,
                                       'params': model.params,
                                       'aic': model.aic,
                                       'rsquared': model.rsquared,
                                       'mse': model.mse_model})
            model_metadata_df = pd.DataFrame(model_metadata)

            print(f"parcela:{parcela_id} variable:{variable}")
            if export_metadata:
                meta_json = model_metadata_df.to_json(orient='records')
                meta_to_export = dumps(loads(meta_json), indent=4)

                with open(os.path.join(variable_folder_coef, rf"parcela_{parcela_id}.json"), 'w') as f:
                    f.write(meta_to_export)

            poly_degree_aic(model_metadata_df, 'degree', ['aic', 'rsquared'], f'Parcela {parcela_id}',
                            'Grado del polinomio', ['AIC', '$R^2$'],
                            export=True,
                            export_path=os.path.join(variable_folder_aic, rf"parcela_{parcela_id}.pdf"))

            fig, ax = plt.subplots(3, 2, figsize=(18, 12))
            fig.suptitle(f'Parcela {parcela_id}', fontsize=16)
            ax = ax.flatten()

            for degree in range(1, 7):
                if degree not in models:
                    continue

                model_info = models[degree]
                poly_degree = model_info[1]
                model = model_info[0]

                x_poly = poly_degree.fit_transform(x_data)
                y_pred = model.predict(sm.add_constant(x_poly))
                ds[f'{variable}_pred_{degree}'] = y_pred

                ax[degree - 1].plot(ds['dia'], ds[variable], label='Real', color='royalblue')
                ax[degree - 1].plot(ds['dia'], ds[f'{variable}_pred_{degree}'], label=f'Grado {degree}',
                                    color='coral')
                ax[degree - 1].set_title(f'Grado {degree}')
                ax[degree - 1].legend(loc='lower right')

                box_text = f'$R^2$: {model.rsquared:.3f}\nAIC: {model.aic:.3f}\nMSE: {model.mse_model:.3f}'
                ax[degree - 1].text(0.05, 0.95, box_text, transform=ax[degree - 1].transAxes, fontsize=12,
                                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ds.to_csv(os.path.join(variable_folder_predicts, f'parcela_{parcela_id}.csv'), index=False)
            plt.subplots_adjust(hspace=0.5, wspace=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(variable_folder_polys, f'parcela_{parcela_id}.pdf'), dpi=100)
