import unittest
import pandas as pd
from pandas import DataFrame
from plot_utils.charts import simple_line_plot, two_line_plot, precipitation_plot, vi_vs_climate_plot, poly_degree_plot
from some_utils.cleanning_data import harvest_dates
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime
import geopandas as gpd
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import statsmodels.api as sm

from stats_utils.fourier import fft, adjust_fft_curve, sinusoidal
import seaborn as sns
from stats_utils.regression_models import linear_reg_model
from mpl_toolkits.mplot3d import Axes3D

ds_path = r"C:\Users\Isai\Documents\Tesis\code\data_analysis\datos\fourier\relativity_humidity\parcela_5.csv"
ds_path_2 = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\indices_stats_cleaned\parcela_16.csv"
ds_path_3 = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\merged\parcela_5.csv"
ds_path_4 = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\ready_to_analyze\parcela_5.csv"
ds_path_5 = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\precipitation_harvest_dates\parcela_5.csv"
ds_path_6 = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\temperature_harvest_dates\parcela_5.csv"
ds_path_7 = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\evapo_harvest_dates\parcela_5.csv"
ds_path_8 = r"C:\Users\Isai\Documents\Tesis\code\data_analysis\datos\model_predicts\ndvi_mean"
ds_path_9 = r"C:\Users\Isai\Documents\Tesis\code\data_analysis\datos\model_predicts\parcela_5.csv"
temperatura = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\temperature\parcela_5.csv"
evapo = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\evapotranspiration\parcela_5.csv"
rh = rf"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\relative_humidity\parcela_5.csv"
rf = r"C:\Users\Isai\Documents\Tesis\code\data_analysis\datos\fourier\solar_radiation\parcela_1.csv"
rf_ref = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\radiation_flux\parcela_1.csv"

pd.set_option('display.max_rows', None)  # Muestra todas las filas
pd.set_option('display.max_columns', None)


class MyTestCase(unittest.TestCase):
    def test_plotting(self):
        ds = pd.read_csv(r"C:\Users\Isai\Documents\Tesis\code\data_analysis\all_vars\zafra2021\parcela_1.csv")
        ds['Fecha'] = pd.to_datetime(ds['Fecha'])
        print(ds)
        simple_line_plot(ds, 'Fecha', 'ndvi', 'Parcela 1', 'Fecha', 'ndvi', export=False)
        self.assertEqual(True, True)  # add assertion here

    def test_plotting_2(self):
        ds = pd.read_csv(r"C:\Users\Isai\Documents\Tesis\code\data_analysis\all_vars\zafra2021\parcela_10.csv")
        ds['Kc'] = 1.15 * ds['ndvi'] + 0.17
        ds['ETc'] = ds['Kc'] * ds['Evapotranspiration']
        ds2 = pd.read_csv(r"C:\Users\Isai\Documents\Tesis\code\data_analysis\datos\model_predicts\zafra2021\ndvi_mean\parcela_10.csv")
        ds['Fecha'] = pd.to_datetime(ds['Fecha'])
        ds2['Fecha'] = pd.to_datetime(ds2['Fecha'])
        two_line_plot(ds, ds2, 'Fecha', ['ndvi', 'ndvi_mean_pred_2'], 'Parcela 6', 'Fecha', 'ndvi', export=False)
        self.assertEqual(True, True)

    def test_poly(self):
        ds = pd.read_csv(ds_path_4)
        ds['Fecha'] = pd.to_datetime(ds['Fecha'])
        ds['Dia'] = (ds['Fecha'] - ds['Fecha'].min()).dt.days

        coef = np.polyfit(ds['Dia'], ds['ndvi_mean'], 1)
        poly = np.poly1d(coef)

        ds['y_fit'] = poly(ds['Dia'])

        print(ds, coef)
        print(poly)
        print(r2_score(ds['ndvi_mean'], ds['y_fit']))
        plt.scatter(ds['Dia'], ds['ndvi_mean'], label='Real', color='r')
        plt.plot(ds['Dia'], ds['y_fit'], label='Fit')
        plt.legend()
        plt.show()

        self.assertEqual(True, True)

    def test_plotting_all(self):
        root = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\precipitation_harvest_dates"
        csv_files = [file for file in os.listdir(root) if file.endswith('.csv')]

        fig, ax = plt.subplots()

        for file in csv_files:
            full_path = os.path.join(root, file)

            df = pd.read_csv(full_path)
            df['Fecha'] = pd.to_datetime(df['Fecha'])
            df['Dia'] = (df['Fecha'] - df['Fecha'].min()).dt.days
            df['acum_precip'] = df['Precipitation'].cumsum()
            ax.plot(df['Dia'], df['Precipitation'], label=file[:-4].split('_')[1], linestyle='--', linewidth=0.7)

        ax.set_xlabel('Días')
        ax.set_ylabel('Temp (°K)')
        ax.set_title('Serie de tiempo de temperatura')
        ax.legend(title='Parcela')

        plt.show()

    def test_plot_polys(self):
        root = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\ready_to_analyze"
        csv_files = [file for file in os.listdir(root) if file.endswith('.csv')]
        fig, ax = plt.subplots()

        for file in csv_files:
            full_path = os.path.join(root, file)
            df = pd.read_csv(full_path)
            df['Fecha'] = pd.to_datetime(df['Fecha'])
            df['Dia'] = (df['Fecha'] - df['Fecha'].min()).dt.days
            coef = np.polyfit(df['Dia'], df['ndvi_mean'], 2)
            print(f'{file[:-4]}: {coef}')
            poly = np.poly1d(coef)
            df['y_fit'] = poly(df['Dia'])
            print(df)
            print(file[:-4].split('_')[1])
            # ax.scatter(df['Dia'], df['ndvi_mean'], label=file[:-4], color='r')
            ax.plot(df['Dia'], df['y_fit'], label=file[:-4].split('_')[1], linestyle='--', linewidth=0.7)

        ax.set_xlabel('Días')
        ax.set_ylabel('Temp (°K)')
        ax.set_title('Poli de sexto grado')
        ax.legend(title='Parcela')
        plt.show()

    def test_precip(self):
        ds = pd.read_csv(ds_path_5)
        ds['Fecha'] = pd.to_datetime(ds['Fecha'])
        precipitation_plot(ds, 'Fecha', 'Precipitation', 'Parcela 5', 'Fecha', 'Precipitación', export=False)
        self.assertEqual(True, True)

    def test_vi_climate(self):
        ds = pd.read_csv(r"C:\Users\Isai\Documents\Tesis\code\data_analysis\all_vars\zafra2021\parcela_1.csv",
                         parse_dates=True)
        ds['Fecha'] = pd.to_datetime(ds['Fecha'])
        ds['Kc'] = 1.15 * ds['ndvi'] + 0.17
        ds['ETc'] = ds['Kc'] * ds['Ajustados']
        ds['ETc_acum'] = ds['ETc'].cumsum()
        ds2 = pd.read_csv(r"C:\Users\Isai\Documents\Tesis\code\data_analysis\all_vars\parcela_16.csv")
        print(ds.corr())
        cor = ds.corr()
        # cor.columns = ['Fecha', 'dias', 'ndvi', 'Temperature', 'Ajustados_x', 'SRF', 'Ajustados_y', 'Precipitation', 'acum', 'Evapotranspiration', 'Ajustados',
        # 'acum_ref', 'acum_ajustados', 'Kc', 'ETc', 'ETc_acum', 'RH12', 'holtwinters_predicts']
        ds2['Fecha'] = pd.to_datetime(ds2['Fecha'])
        vi_vs_climate_plot(ds, ds2, ['Fecha', 'ndvi'], ['Fecha', 'Precipitation'],
                           'Parcela 16', 'Fecha', ['NDVI', 'Preci mm/dia'],
                           x_label_type='dias', export=False)
        self.assertEqual(True, True)

    def test_multi_poly(self):
        ds = pd.read_csv(ds_path_4)

        days = ds['Dia'].values.reshape(-1, 1)
        ndvi = ds['ndvi_mean'].values
        print(days)

        x_train, x_test, y_train, y_test = train_test_split(days, ndvi, test_size=0.2, random_state=42)
        grados = np.arange(1, 10)
        first_degree = grados[0]
        last_degree = grados[-1]

        for grado in grados:
            poly_features = PolynomialFeatures(degree=grado, include_bias=False)
            x_poly_train = poly_features.fit_transform(x_train)
            x_poly_test = poly_features.transform(x_test)

            model = LinearRegression()
            model.fit(x_poly_train, y_train)
            print(model.coef_, model.intercept_)

            y_pred = model.predict(x_poly_test)
            print(len(x_poly_test), len(y_pred))

            mse_test = mean_squared_error(y_test, model.predict(x_poly_test))
            model_applid = model.predict(x_poly_train)
            print(x_poly_train, x_poly_train.size)
            print(model_applid)

    def test_polidata(self):

        def calc_degree(data: DataFrame, columns: list, max_degree: int = 10):
            x = data[columns[0]].values.reshape(-1, 1)
            y = data[columns[1]].values

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

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
                # print(model.summary())

                df = df._append({'degree': degree,
                                 'params': model.params,
                                 'aic': model.aic,
                                 'rsquared': model.rsquared,
                                 'mse': model.mse_model}, ignore_index=True)

            return df, modelos

        dfs = []
        for i in os.listdir(ds_path_8):
            df = pd.read_csv(os.path.join(ds_path_8, i))
            dfs.append(df)

        df_tot = pd.concat(dfs, axis=0, ignore_index=True)
        # print(df)

        days = df_tot['Dia'].values.reshape(-1, 1)
        ndvi_modelo = df_tot['y_pred_4'].values
        ndvi_mean = df_tot['ndvi_mean'].values

        datos, modelos = calc_degree(df_tot, ['Dia', 'y_pred_4'])
        print(modelos.keys())
        model_info = modelos[4]
        poly_degree = model_info[1]
        model = model_info[0]

        x_poly = poly_degree.transform(days)
        y_pred = model.predict(sm.add_constant(x_poly))
        print(y_pred)
        fig, ax = plt.subplots()
        plt.style.use('_mpl-gallery')
        ax.scatter(days, y_pred, color='coral', label='modelo de grado 2', s=10, alpha=1, marker='v')
        ax.scatter(days, ndvi_modelo, marker='o', color='teal', label='datos modelados',
                   s=8, alpha=0.7)

        ax.scatter(days, ndvi_mean, color='olive', label='datos reales', s=8, alpha=0.7, marker='D')
        ax.set_title('Modelo polinomial de grado 4')
        ax.set_xlabel('Días')
        ax.set_ylabel('NDVI')
        ax.legend(loc='upper left')
        box_text = f'$R^2$:{model.rsquared:.3f}\nAIC: {model.aic:.3f}\nMSE: {model.mse_model:.3f}'
        ax.text(0.9, 0.9, box_text, transform=ax.transAxes,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'), fontsize=8)

        plt.show()

        pass

    def test_climate_multicolumn(self):
        fig, ax = plt.subplots(2, 2, figsize=(12, 12))

        ax[0, 0].plot(pd.read_csv(ds_path_9)['Dia'], pd.read_csv(ds_path_9)['ndvi_mean'], color='yellowgreen',
                      label='Datos reales', linestyle='--', linewidth=0.7)
        ax[0, 0].plot(pd.read_csv(ds_path_9)['Dia'], pd.read_csv(ds_path_9)['y_pred_2'], color='forestgreen',
                      label='Datos modelados', linestyle='--', linewidth=0.7)
        ax[0, 0].set_title('NDVI promedio')
        ax[0, 0].set_xlabel('Días')
        ax[0, 0].set_ylabel('NDVI')

        ax[0, 1].plot(pd.read_csv(ds_path_5)['Dia'], pd.read_csv(ds_path_5)['Precipitation'], color='cornflowerblue',
                      label='Datos reales', linestyle='--', linewidth=0.7)
        ax[0, 1].set_title('Precipitación')
        ax[0, 1].set_xlabel('Días')
        ax[0, 1].set_ylabel('mm por día')

        ax[1, 0].plot(pd.read_csv(ds_path_6)['Dia'], pd.read_csv(ds_path_6)['Temperature'], color='firebrick',
                      label='Datos reales', linestyle='--', linewidth=0.7)
        ax[1, 0].set_title('Temperatura')
        ax[1, 0].set_xlabel('Días')
        ax[1, 0].set_ylabel('°K')

        ax[1, 1].plot(pd.read_csv(ds_path_7)['Dia'], pd.read_csv(ds_path_7)['Evapotranspiration'], color='goldenrod',
                      label='Datos reales', linestyle='--', linewidth=0.7)
        ax[1, 1].set_title('Evapotranspiración')
        ax[1, 1].set_xlabel('Días')
        ax[1, 1].set_ylabel('mm por día')
        plt.suptitle('Parcela 5')
        plt.show()

        pass

    def test_fft(self):

        temp_data = pd.read_csv(rh)
        temp_data['Fecha'] = pd.to_datetime(temp_data['Fecha'])
        temp_data['Dia'] = (temp_data['Fecha'] - temp_data['Fecha'].min()).dt.days + 1

        results = fft(temp_data, 0.1, ['Fecha', 'RH12', 'Dia'])
        temp_data['reconstructed'] = results['senal_reconstruida'].real

        optimized, cov = adjust_fft_curve(temp_data, ['Fecha', 'RH12', 'Dia'], results)
        print(optimized, np.linalg.cond(cov))

        adjusted = sinusoidal(temp_data['Dia'], *optimized)
        temp_data['ajustado'] = adjusted
        print(temp_data)
        plt.plot(pd.to_datetime(temp_data['Fecha']), temp_data['RH12'], label='Señal Original',
                 color='firebrick', linewidth=1, alpha=0.7)
        plt.plot(pd.to_datetime(temp_data['Fecha']), results['senal_reconstruida'].real, label='Señal Ajustada',
                 linestyle='-.',
                 color='navy',
                 linewidth=1)
        plt.plot(pd.to_datetime(temp_data['Fecha']), adjusted, label='Señal Reconstruida', linestyle='-.', color='gold')
        plt.show()

    def test_poly_deg(self):
        ds = pd.read_csv(ds_path_9)

        poly_degree_plot(ds, 'ndvi_mean')

    def test_holt_plot(self):
        ds = pd.read_csv(
            rf"C:\Users\Isai\Documents\Tesis\code\data_analysis\datos\holtwinters\relative_humidity\parcela_1.csv")
        ds['Fecha'] = pd.to_datetime(ds['Fecha'])
        simple_line_plot(ds, 'Fecha', 'holtwinters_predicts', 'Parcela 1', 'Fecha', 'RH', export=False)
        pass

    def test_climate_data_models(self):
        varible = 'evapotranspiration'
        varible_df = 'Evapotranspiration'

        arima = pd.read_csv(rf"C:\Users\Isai\Documents\Tesis\code\data_analysis\datos\arima\{varible}\parcela_1.csv",
                            parse_dates=True)
        arima = arima[arima['arima_predicts'] > 0]
        holt = pd.read_csv(
            rf"C:\Users\Isai\Documents\Tesis\code\data_analysis\datos\holtwinters\{varible}\parcela_1.csv",
            parse_dates=True)
        fourier = pd.read_csv(
            rf"C:\Users\Isai\Documents\Tesis\code\data_analysis\datos\fourier\{varible}\parcela_1.csv",
            parse_dates=True)
        original_data = pd.read_csv(rf"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\{varible}\parcela_1.csv",
                                    parse_dates=True)

        fig, ax = plt.subplots(3, 1, figsize=(10, 12))

        fig.suptitle('Comparación de modelos de datos climáticos')
        ax[0].plot(pd.to_datetime(original_data['Fecha']), original_data[varible_df], label='Datos reales',
                   color='royalblue', linestyle='-', linewidth=1)
        ax[0].plot(pd.to_datetime(arima['Fecha']), arima['arima_predicts'], label='ARIMA', color='orangered',
                   linestyle='-', linewidth=1, alpha=0.7)
        ax[0].set_xlabel('Fecha')
        ax[0].set_ylabel('Evapotranspiración (mm/día)')
        ax[0].set_title('ARIMA')
        ax[0].legend()

        ax[1].plot(pd.to_datetime(original_data['Fecha']), original_data[varible_df], label='Datos reales',
                   color='royalblue', linestyle='-', linewidth=1)
        ax[1].plot(pd.to_datetime(holt['Fecha']), holt['holtwinters_predicts'], label='Holt-Winters', color='orangered',
                   linestyle='-', linewidth=1, alpha=0.7)
        ax[1].set_xlabel('Fecha')
        ax[1].set_ylabel('Evapotranspiración (mm/día)')
        ax[1].set_title('Holt-Winters')
        ax[1].legend()

        ax[2].plot(pd.to_datetime(original_data['Fecha']), original_data[varible_df], label='Datos reales',
                   color='royalblue', linestyle='-', linewidth=1)
        ax[2].plot(pd.to_datetime(fourier['Fecha']), fourier['Ajustados'], label='Fourier', color='orangered',
                   linestyle='-', linewidth=1, alpha=0.7)

        ax[2].set_xlabel('Fecha')
        ax[2].set_ylabel('Evapotranspiración (mm/día)')
        ax[2].set_title('Fourier')
        ax[2].legend()

        plt.subplots_adjust(hspace=1.2, wspace=0.3)
        plt.tight_layout()
        #plt.savefig(rf"C:\Users\Isai\Documents\Tesis\code\imagenes_tesis\{varible}_models.jpg", dpi=300)

        plt.show()
        pass

    def test_vars_vs_vars(self):
        data = pd.read_csv(r"C:\Users\Isai\Documents\Tesis\code\data_analysis\all_vars\zafra2021\parcela_1.csv",
                           parse_dates=True)

        fig, ax = plt.subplots(5, 1, figsize=(12, 12))
        ax[0].scatter(data['ndvi'], data['Temperature'], color='royalblue', label='NDVI vs Temp medida', alpha=0.7)
        ax[0].scatter(data['ndvi'], data['Ajustados_x'], color='orangered', label='NDVI vs Temp ajustada', alpha=0.7)

        ax[1].scatter(data['ndvi'], data['SRF'], color='royalblue', label='NDVI vs Temp medida', s=8,
                      alpha=0.7)
        ax[1].scatter(data['ndvi'], data['Ajustados_y'], color='orangered', label='NDVI vs Temp ajustada', s=8,
                      alpha=0.7)

        ax[2].scatter(data['ndvi'], data['Evapotranspiration'], color='royalblue', label='NDVI vs Temp medida', s=8,
                      alpha=0.7)
        ax[2].scatter(data['ndvi'], data['Ajustados'], color='orangered', label='NDVI vs Temp ajustada', s=8,
                      alpha=0.7)

        ax[3].scatter(data['ndvi'], data['Precipitation'], color='royalblue', label='NDVI vs Temp medida', s=8,
                      alpha=0.7)
        ax[3].scatter(data['ndvi'], data['acum'], color='orangered', label='NDVI vs Temp ajustada', s=8,
                      alpha=0.7)

        ax[4].scatter(data['ndvi'], data['RH12'], color='royalblue', label='NDVI vs Temp medida', s=8,
                      alpha=0.7)
        ax[4].scatter(data['ndvi'], data['holtwinters_predicts'], color='orangered', label='NDVI vs Temp ajustada', s=8,
                      alpha=0.7)
        plt.legend()

        plt.tight_layout()
        plt.show()
        pass

    def test_Seaborn(self):
        data = pd.read_csv(r"C:\Users\Isai\Documents\Tesis\code\data_analysis\all_vars\zafra2021\parcela_1.csv",
                           parse_dates=True)
        sns.pairplot(data, kind='scatter')
        plt.show()
        pass

    def test_corr(self):
        data = pd.read_csv(r"C:\Users\Isai\Documents\Tesis\code\data_analysis\all_vars\zafra2021\parcela_5.csv",
                           parse_dates=True)
        print(data[['ndvi', 'Evapotranspiration', 'acum_ref', 'Ajustados', 'acum_ajustados']].corr())
        plt.scatter(data['ndvi'], data['Evapotranspiration'], marker='o', color='royalblue',
                    label='NDVI vs Temp modelada', s=8)
        plt.scatter(data['ndvi'], data['acum_ref'], marker='o', color='orangered', label='NDVI vs Temp medida', s=8)
        plt.scatter(data['ndvi'], data['Ajustados'], marker='o', color='gold', label='NDVI vs Temp ajustada', s=8)
        plt.scatter(data['ndvi'], data['acum_ajustados'], marker='o', color='teal',
                    label='NDVI vs Temp ajustada acumulada', s=8)
        plt.legend()
        plt.show()
        pass

    def test_rec_poly(self):
        data = pd.DataFrame({'x': np.linspace(1, 395, 394), 'Fecha': pd.date_range(start='2022-03-22', periods=394)})
        data['y'] = -1.2686e-05 * data['x'] ** 2 + 0.0058008862 * data['x'] + 0.1160076519
        data['Fecha'] = pd.to_datetime(data['Fecha'])

        wheather = pd.read_csv(
            r"C:\Users\Isai\Documents\Tesis\code\data_analysis\datos\fourier\evapotranspiration\parcela_5.csv",
            parse_dates=True)
        wheather['Fecha'] = pd.to_datetime(wheather['Fecha'])
        wheather = wheather[(wheather['Fecha'] >= '2022-03-22') & (wheather['Fecha'] <= '2023-04-21')]

        print(wheather.dtypes)

        data = pd.merge(data, wheather, on='Fecha', how='outer')
        sns.pairplot(data, kind='scatter')
        print(data.corr())

        fig, ax = plt.subplots()
        ax.scatter(data['Fecha'], data['y'])
        ax1 = ax.twinx()
        ax1.scatter(wheather['Fecha'], wheather['Ajustados'])
        plt.show()

    def test_vis(self):
        data = r"C:\Users\Isai\Documents\Tesis\code\data_analysis\all_vars\parcela_16.csv"
        data2 = r"C:\Users\Isai\Documents\Tesis\code\data_analysis\datos\model_predicts\parcela_16.csv"
        data = pd.read_csv(data, parse_dates=True)
        data2 = pd.read_csv(data2, parse_dates=True)
        data['Fecha'] = pd.to_datetime(data['Fecha'])
        data['Kc'] = 1.15 * data['ndvi'] + 0.17
        data['ETc'] = data['Kc'] * data['Evapotranspiration']
        print(data.corr())

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.plot(data2['Dia'], data2['msi_mean'], label='NDVI', color='royalblue', linestyle='-', linewidth=1)
        ax1 = ax.twinx()
        ax1.plot(data['dias'], data['msi'], label='kc', color='orangered', linestyle='-', linewidth=1)
        plt.legend()
        plt.show()

    def test_all(self):
        dfs = []
        dfs2 = []
        for i in os.listdir(r"C:\Users\Isai\Documents\Tesis\code\data_analysis\all_vars\zafra2021"):
            df = pd.read_csv(os.path.join(r"C:\Users\Isai\Documents\Tesis\code\data_analysis\all_vars\zafra2021", i))
            dfs.append(df)

        for i in os.listdir(r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\ready_to_analyze\zafra2021"):
            df2 = pd.read_csv(
                os.path.join(r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\ready_to_analyze\zafra2021", i))
            dfs2.append(df2)

        df_tot = pd.concat(dfs, axis=0, ignore_index=True)
        df_tot2 = pd.concat(dfs2, axis=0, ignore_index=True)
        # print(df)

        days = df_tot['dias'].values.reshape(-1, 1)
        days2 = df_tot2['dia'].values.reshape(-1, 1)
        ndvi_modelo = df_tot['ndvi'].values
        ndvi_mean = df_tot2['ndvi_mean'].values
        lluvia = df_tot['Precipitation'].values

        datos, modelos = linear_reg_model(df_tot, ['dias', 'ndvi'])
        print(modelos.keys())
        model_info = modelos[2]
        poly_degree = model_info[1]
        model = model_info[0]

        x_poly = poly_degree.transform(days)
        y_pred = model.predict(sm.add_constant(x_poly))
        print(y_pred)
        fig, ax = plt.subplots()
        plt.style.use('_mpl-gallery')
        ax.scatter(days, y_pred, color='coral', label='modelo de grado 2', s=10, alpha=1, marker='v')
        ax.scatter(days, ndvi_modelo, marker='o', color='teal', label='datos modelados',
                   s=8, alpha=0.7)

        ax.scatter(days2, ndvi_mean, color='orangered', label='datos reales', s=8, alpha=0.7, marker='D')
        ax.set_title('Modelo polinomial de grado 4')
        ax.set_xlabel('Días')
        ax.set_ylabel('NDVI')
        ax.legend(loc='upper left')
        box_text = f'$R^2$:{model.rsquared:.3f}\nAIC: {model.aic:.3f}\nMSE: {model.mse_model:.3f}'
        ax.text(0.9, 0.9, box_text, transform=ax.transAxes,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'), fontsize=8)

        plt.show()
        pass

    def test_mean_vis(self):
        from plot_utils.charts import column_adjust_plot
        data = pd.read_csv(r'C:\Users\Isai\Documents\Tesis\code\garbage_tests\parcela_4.csv', parse_dates=True)
        data['Fecha'] = pd.to_datetime(data['Fecha'])
        data = data.sort_values(by=['Fecha'], ascending=True)
        print(data)
        columnas = ['ndvi_mean', 'ndmi_mean', 'msi_mean', 'gndvi_mean', 'evi_mean', 'ndwi_mean']
        for i in range(1, 15, 1):
            print(i)
        column_adjust_plot(data, columnas, 'Parcela 3')
        pass

    def test_uno(self):
        ds = pd.read_csv(r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\indices_stats\parcela_7.csv")
        ds['Fecha'] = pd.to_datetime(ds['Fecha'])
        ds = ds.sort_values(by=['Fecha'], ascending=True)
        plt.plot(ds['Fecha'], ds['ndvi_mean'])
        plt.show()
        pass

    def test_read_geojson(self):
        gpf = gpd.read_file(r"C:\Users\Isai\Documents\Tesis\code\Parcelas\poligonos_parcelas.geojson", driver='GeoJSON')
        print(gpf)

    def test_get_max_value(self):
        ndvi_root = r"C:\Users\Isai\Documents\Tesis\code\data_analysis\all_vars\zafra2022"
        ndvi_root_21 = r"C:\Users\Isai\Documents\Tesis\code\data_analysis\all_vars\zafra2021"

        gpf = gpd.read_file(r"C:\Users\Isai\Documents\Tesis\code\Parcelas\poligonos_parcelas.geojson", driver='GeoJSON')
        promedios = gpf.groupby('Productor')['Rendimiento'].mean()
        gpf['promedios'] = gpf['Productor'].map(promedios)
        gpf = gpf.drop_duplicates(subset='Productor')

        empty_df = pd.DataFrame(columns=['Parcela', 'Maximo', 'lluvia', 'evapo', 'evapo_ajus', 'dias', 'kc', 'Etc'])
        for i in os.listdir(ndvi_root):
            parcela_id = int(i.split('_')[1].split('.')[0])
            ds = pd.read_csv(os.path.join(ndvi_root, i))
            max_value = ds['ndvi'].max()
            lluvia = ds['Precipitation'].sum()
            evapo_ref = ds['Evapotranspiration'].sum()
            evapo_ajustados = ds['Ajustados'].sum()
            dias = ds['dias'].max()
            kc = 1.15 * max_value + 0.17
            Etc = (ds['Evapotranspiration'] * 1.15 * ds['ndvi'] + 0.17).sum()


            empty_df = empty_df._append({'Parcela': parcela_id, 'Maximo': max_value, 'lluvia': lluvia, 'evapo': evapo_ref,
                                         'evapo_ajus': evapo_ajustados, 'dias': dias, 'kc': kc, 'Etc': Etc}, ignore_index=True)

        empty_df_2 = pd.DataFrame(columns=['Parcela', 'Maximo', 'lluvia', 'evapo', 'evapo_ajus', 'dias', 'kc', 'Etc'])
        for i in os.listdir(ndvi_root_21):
            parcela_id = int(i.split('_')[1].split('.')[0])
            ds = pd.read_csv(os.path.join(ndvi_root_21, i))
            max_value = ds['ndvi'].max()
            lluvia = ds['Precipitation'].sum()
            evapo_ref = ds['Evapotranspiration'].sum()
            evapo_ajustados = ds['Ajustados'].sum()
            dias = ds['dias'].max()
            kc = 1.15 * max_value + 0.17
            Etc = (ds['Evapotranspiration'] * 1.15 * ds['ndvi'] + 0.17).sum()

            empty_df_2 = empty_df_2._append({'Parcela': parcela_id, 'Maximo': max_value, 'lluvia': lluvia, 'evapo': evapo_ref,
                                             'evapo_ajus': evapo_ajustados, 'dias': dias, 'kc': kc, 'Etc': Etc}, ignore_index=True)

        empty_df = empty_df.drop(empty_df.loc[empty_df['Parcela'].isin([6, 7, 8, 9, 10, 11])].index)
        full_df = pd.merge(gpf, empty_df, left_on='Id', right_on='Parcela', how='outer')
        data_22 = full_df[['Parcela', 'Maximo', 'Rendimiento', 'lluvia', 'evapo', 'evapo_ajus', 'dias', 'kc', 'Etc']]
        # empty_df_2['Rendimiento'] = pd.Series(np.random.randint(25, 75, size=15))
        empty_df_2['Rendimiento'] = pd.Series([50, 70, 72, 35, 56, 63, 40, 63, 43, 54, 70, 72, 71, 68, 74])

        ultimate_df = pd.concat([data_22, empty_df_2], ignore_index=True)
        # ultimate_df.drop(24, inplace=True)
        print(ultimate_df)

        print('2022', data_22[['Maximo', 'Rendimiento', 'lluvia', 'evapo', 'evapo_ajus', 'dias', 'kc', 'Etc']].corr())

        metadata, modelos = linear_reg_model(ultimate_df, ['Etc', 'Rendimiento'])
        meta_2, modelos_2 = linear_reg_model(data_22, ['Maximo', 'Rendimiento'])
        x_3 = data_22[['Maximo', 'Etc']]
        x_3 = sm.add_constant(x_3)
        y_3 = data_22['Rendimiento']
        modelo_3 = sm.OLS(y_3, x_3).fit()
        predictions = modelo_3.predict(x_3)
        print(f"mse: {modelo_3.mse_model}")
        print(f"rsquared: {modelo_3.rsquared}")
        print(modelo_3.summary())

        model_info_2 = modelos_2[1]
        poly_degree_2 = model_info_2[1]
        model_2 = model_info_2[0]
        print(meta_2)
        print(model_2.summary())
        x_1 = data_22['Etc'].values.reshape(-1, 1)
        x = ultimate_df['Etc'].values.reshape(-1, 1)
        y = ultimate_df['Rendimiento'].values
        y_2 = data_22['Rendimiento'].values



        x_poly_2 = poly_degree_2.transform(x_1)
        y_pred_2 = model_2.predict(sm.add_constant(x_poly_2))

        x1_vals = np.linspace(data_22['Maximo'].min(), data_22['Maximo'].max(), 10)
        x2_vals = np.linspace(data_22['Etc'].min(), data_22['Etc'].max(), 10)
        x1_mesh, x2_mesh = np.meshgrid(x1_vals, x2_vals)
        X_mesh = np.column_stack([x1_mesh.ravel(), x2_mesh.ravel()])
        X_mesh = sm.add_constant(X_mesh)
        y_pred_mesh = modelo_3.predict(X_mesh)
        y_pred_mesh = y_pred_mesh.reshape(x1_mesh.shape)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Graficar los puntos de datos
        ax.scatter(data_22['Maximo'], data_22['Etc'], data_22['Rendimiento'], color='blue', label='Datos reales')

        # Graficar el plano de regresión
        ax.plot_surface(x1_mesh, x2_mesh, y_pred_mesh, alpha=0.5, color='red', label='Plano de regresión')

        # Etiquetas y título
        ax.set_xlabel('Maximo')
        ax.set_ylabel('lluvia')
        ax.set_zlabel('Rendimiento')
        ax.set_title('Regresión lineal múltiple en 3D')

        # Mostrar leyenda
        ax.legend()

        # Mostrar el gráfico
        plt.show()
        '''
        plt.scatter(y_2, y_pred_2, color='coral')
        plt.scatter(y_3, predictions, color='royalblue')
        plt.plot(y_3, y_3, color='black')
        #plt.scatter(data_22['Etc'], data_22['Rendimiento'], color='gold', label='2021')
        '''
        plt.show()

    def test_rainbow(self):
        root = r"C:\Users\Isai\Documents\Tesis\code\data_analysis\all_vars\zafra2022"

        archivos_csv = [archivo for archivo in os.listdir(root) if archivo.endswith('.csv')]
        cmap = get_cmap('Paired')  # Puedes cambiar 'viridis' por cualquier otra paleta de colores

        # Normalizar los índices de los archivos para asignar colores
        norm = Normalize(vmin=0, vmax=len(archivos_csv) - 1)

        # Graficar cada archivo CSV
        for i, archivo in enumerate(archivos_csv):
            # Leer el archivo CSV
            datos = pd.read_csv(os.path.join(root, archivo))

            # Extraer columnas de datos
            datos['Fecha'] = pd.to_datetime(datos['Fecha'])
            datos = datos.sort_values(by=['Fecha'], ascending=True)


            # Obtener el color correspondiente de la rampa
            color = cmap(norm(i))

            # Graficar los datos con el color correspondiente
            plt.plot(datos['Fecha'], datos['ndvi'], color=color, label=archivo.split('.')[0].split('_')[1])

        # Agregar leyenda
        plt.legend()

        # Mostrar el gráfico
        plt.show()
        pass


    def test_promedios(self):
        gpf = gpd.read_file(r"C:\Users\Isai\Documents\Tesis\code\Parcelas\poligonos_parcelas.geojson", driver='GeoJSON')
        promedios = gpf.groupby('Productor')['Rendimiento'].mean()
        gpf['promedios'] = gpf['Productor'].map(promedios)
        gpf = gpf.drop_duplicates(subset='Productor')
        print(gpf)
        pass

if __name__ == '__main__':
    unittest.main()
