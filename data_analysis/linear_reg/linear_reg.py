import os
import json
import pandas as pd
import statsmodels.api as sm

from stats_utils.regression_models import polinomial_reg

from plot_utils.charts import poly_degree_aic
import matplotlib.pyplot as plt


zafra = 2021
export_metadata = True
main_vars = ['ndvi_mean', 'gndvi_mean', 'msi_mean', 'ndmi_mean', 'cire_mean', 'ndre1_mean']

coef_folder = rf"coeficientes/zafra{zafra}"
aic_img_folder = rf"../../tesis_img/aic_vs_degrees/zafra{zafra}"
model_predicts_folder = fr"../datos/model_predicts/zafra{zafra}"
polys_img_folder = fr"../../tesis_img/polys_vs_data/zafra{zafra}"

with os.scandir(rf'../../datos/parcelas/ready_to_analyze/zafra{zafra}') as files:
    for file in files:
        df = pd.read_csv(file.path, parse_dates=['Fecha'])
        parcela_id = file.name.split('_')[-1].split('.')[0]
        for var in df.columns:
            if var in main_vars:
                var_folder_coef = os.path.join(coef_folder, var)
                if not os.path.exists(var_folder_coef):
                    os.makedirs(var_folder_coef)
                    print(f'Creating folder {var_folder_coef}')
                print(f'Parcela {parcela_id} - Variable {var}')

                var_folder_aic = os.path.join(aic_img_folder, var)
                if not os.path.exists(var_folder_aic):
                    os.makedirs(var_folder_aic)
                    print(f'Creating folder {var_folder_aic}')

                var_folder_predicts = os.path.join(model_predicts_folder, var)
                if not os.path.exists(var_folder_predicts):
                    os.makedirs(var_folder_predicts)
                    print(f'Creating folder {var_folder_predicts}')

                var_folder_polys = os.path.join(polys_img_folder, var)
                if not os.path.exists(var_folder_polys):
                    os.makedirs(var_folder_polys)
                    print(f'Creating folder {var_folder_polys}')

                model_metadata, models = polinomial_reg(df, ['dia', var], 7)

                model_metadata_df = pd.DataFrame(model_metadata)

                if export_metadata:
                    meta_json = model_metadata_df.to_json(orient='records')
                    meta_to_export = json.dumps(json.loads(meta_json), indent=4)

                    with open(os.path.join(var_folder_coef, rf'parcela_{parcela_id}.json'), 'w') as f:
                        f.write(meta_to_export)

                poly_degree_aic(model_metadata_df, 'degree', ['aic', 'rsquared'],
                                f'Parcela {parcela_id}',
                                'Grado del polinomio', ['AIC', '$R^2$'],
                                True, os.path.join(var_folder_aic, f'parcela_{parcela_id}.pdf'))

                fig, ax = plt.subplots(3, 2, figsize=(18, 12))
                fig.suptitle(f'Parcela {parcela_id} - Variable {var}', fontsize=16)
                ax = ax.flatten()

                for degree in range(1, 7):
                    if degree not in models:
                        continue
                    model_info = models[degree]
                    poly_degree = model_info[1]
                    model = model_info[0]

                    x_poly = poly_degree.transform(df['dia'].values.reshape(-1, 1))
                    y_pred = model.predict(sm.add_constant(x_poly))
                    df[f'{var}_pred'] = y_pred

                    ax[degree-1].plot(df['dia'], df[var], label='Real', color=(150/255, 0, 24/255))
                    ax[degree-1].plot(df['dia'], df[f'{var}_pred'], label='Predicción', color=(105/255, 255/255, 231/255))
                    ax[degree-1].set_title(f'Grado {degree}')
                    ax[degree-1].legend(loc='best')
                    box_text = f'$R^2$: {model.rsquared:.3f}\nAIC: {model.aic:.3f}\nMSE: {model.mse_model:.3f}'
                    ax[degree - 1].text(0.05, 0.95, box_text, transform=ax[degree - 1].transAxes, fontsize=12,
                                        verticalalignment='top',
                                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                df.to_csv(os.path.join(var_folder_predicts, f'parcela_{parcela_id}.csv'), index=False)
                plt.subplots_adjust(wspace=0.5, hspace=0.5)
                plt.tight_layout()
                plt.savefig(os.path.join(var_folder_polys, f'parcela_{parcela_id}.pdf'), dpi=100)
                plt.close()
