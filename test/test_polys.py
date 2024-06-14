import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm
from statsmodels.tools.eval_measures import rmse, mse
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from stats_utils.regression_models import linear_reg_model
from statsmodels.tsa.holtwinters import ExponentialSmoothing


root = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\ready_to_analyze"
parcela = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\ready_to_analyze\parcela_1.csv"

class MyTestCase(unittest.TestCase):
    def test_model_degree(self):
        best_degrees = []

        # Iterar sobre los archivos en el directorio
        for file_name in os.listdir(root):
            if file_name.endswith('.csv'):
                file_path = os.path.join(root, file_name)

                # Leer el archivo CSV
                df = pd.read_csv(file_path)

                # Extraer las columnas x e y
                X = df['Dia'].values.reshape(-1, 1)
                y = df['ndvi_mean'].values

                # Realizar validación cruzada para diferentes grados de polinomios
                degrees = np.arange(1, 10)
                cv_scores = []

                for degree in degrees:
                    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
                    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
                    cv_scores.append(-scores.mean())

                # Encontrar el grado que minimiza el error
                best_degree = degrees[np.argmin(cv_scores)]
                best_degrees.append(best_degree)

                # Visualizar los resultados solo si lo deseas
                model = make_pipeline(PolynomialFeatures(best_degree), LinearRegression())
                model.fit(X, y)

                # Visualizar datos y curva de regresión polinómica
                plt.scatter(X, y, label='Datos reales')
                x_fit = np.linspace(min(X), max(X), 100).reshape(-1, 1)
                y_fit = model.predict(x_fit)
                plt.plot(x_fit, y_fit, label=f'Regresión Polinómica (grado {best_degree})', color='red')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.title(f'Ajuste para {file_path}')
                plt.legend()
                plt.show()

        # Imprimir los grados óptimos para cada archivo
        print("Grados óptimos para cada archivo:")
        for file_name, degree in zip(os.listdir(root), best_degrees):
            if file_name.endswith('.csv'):
                print(f"Archivo {file_name}: Grado {degree}")

        self.assertEqual(True, True)  # add assertion here


    def test_aic(self):
        directory_path = root

        # Lista para almacenar los grados óptimos de cada archivo
        best_degrees_aic = []

        # Iterar sobre los archivos en el directorio
        for file_name in os.listdir(directory_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(directory_path, file_name)

                # Leer el archivo CSV
                df = pd.read_csv(file_path)

                # Extraer las columnas x e y
                X = df['Dia'].values.reshape(-1, 1)
                y = df['ndvi_mean'].values

                # Inicializar variables para el AIC mínimo y su correspondiente grado
                min_aic = float('inf')
                best_degree = None

                # Realizar validación cruzada para diferentes grados de polinomios
                degrees = np.arange(1, 10)

                for degree in degrees:
                    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
                    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
                    mse = -scores.mean()
                    n = len(X)

                    # Calcular AIC
                    aic = 2 * (degree + 1) + n * np.log(mse)

                    # Actualizar el grado óptimo si encontramos un AIC menor
                    if aic < min_aic:
                        min_aic = aic
                        best_degree = degree

                best_degrees_aic.append((best_degree, min_aic))

                # Visualizar los resultados solo si lo deseas
                model = make_pipeline(PolynomialFeatures(best_degree), LinearRegression())
                model.fit(X, y)

                # Visualizar datos y curva de regresión polinómica
                plt.scatter(X, y, label='Datos reales')
                x_fit = np.linspace(min(X), max(X), 100).reshape(-1, 1)
                y_fit = model.predict(x_fit)
                plt.plot(x_fit, y_fit, label=f'Regresión Polinómica (grado {best_degree})', color='red')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.title(f'Ajuste para {file_path}')
                plt.legend()
                plt.show()

        # Imprimir los grados óptimos para cada archivo
        print("Grados óptimos para cada archivo:")
        for file_name, (degree, aic) in zip(os.listdir(directory_path), best_degrees_aic):
            if file_name.endswith('.csv'):
                print(f"Archivo {file_name}: Grado {degree}, aic: {aic:2f}")

    def test_model(self):
        df = pd.read_csv(parcela)
        days = df['Dia'].values.reshape(-1, 1)
        ndvi = df['ndvi_mean'].values

        # Crear un modelo de regresión lineal
        x_poly = PolynomialFeatures(degree=3).fit_transform(days.reshape(-1, 1))
        model = LinearRegression()
        model.fit(x_poly, ndvi)
        y_pred = model.predict(x_poly)
        print(y_pred)
        print(model.coef_, model.intercept_)
        print(f"mse: {mean_squared_error(ndvi, y_pred)}, r2: {r2_score(ndvi, y_pred)}")

        plt.scatter(days, ndvi, label='Datos reales')
        plt.plot(days, y_pred, label='Regresión Polinómica', color='red')
        plt.xlabel('Días')
        plt.ylabel('NDVI')
        plt.title('Ajuste para parcela 5')
        plt.legend()
        plt.show()


    def test_akaike(self):
        df = pd.read_csv(parcela)
        x = df['Dia'].values.reshape(-1, 1)
        y = df['ndvi_mean'].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        print(x_train)
        grados = np.arange(1, 10)
        aic_values = []

        for grado in grados:
            poly_features = PolynomialFeatures(degree=grado, include_bias=False)
            x_poly_train = poly_features.fit_transform(x_train)
            print(x_poly_train.size)
            print(x_poly_train)
            x_poly_test = poly_features.transform(x_test)
            print(x_poly_test)

            model = sm.OLS(y_train, sm.add_constant(x_poly_train)).fit()
            y_pred = model.predict(sm.add_constant(x_poly_test))
            mse_statsmodel = mse(y_test, y_pred)

            mse_test = mean_squared_error(y_test, model.predict(sm.add_constant(x_poly_test)))
            r2_test = r2_score(y_test, model.predict(sm.add_constant(x_poly_test)))
            print(f"""Grado: {grado}, MSE: {mse_test}, R2: {r2_test, model.rsquared}""")
            num_params = x_poly_train.shape[1]

            print(f"aic: {model.aic}")
            print(f"params: {model.params}")
            print(f"rsquared: {model.rsquared}, mse: {mse_statsmodel}")
            print(model.summary())
            #aic_values.append(model.aic)
        ''''
        plt.plot(grados, aic_values, label='AIC')
        plt.xlabel('Grado del polinomio')
        plt.ylabel('AIC')
        plt.title('AIC vs Grado del polinomio')
        plt.legend()
        plt.show()
        '''
        #grado_optimo = grados[np.argmin(aic_values)]
        #print(aic_values)
        #print(f'Grado óptimo: {grado_optimo}')


    def test_akaike2(self):
        df = pd.read_csv(parcela)
        x = df['Dia'].values.reshape(-1, 1)
        y = df['ndvi_mean'].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        print(x_train)
        grados = np.arange(1, 10)
        aic_values = []

        for grado in grados:
            poly_features = PolynomialFeatures(degree=grado, include_bias=False)
            x_poly_train = poly_features.fit_transform(x_train)
            print(x_poly_train.size)
            print(x_poly_train)
            x_poly_test = poly_features.transform(x_test)

            model = LinearRegression()
            model.fit(x_poly_train, y_train)

            mse_test = mean_squared_error(y_test, model.predict(x_poly_test))
            r2_test = r2_score(y_test, model.predict(x_poly_test))
            print(f"""Grado: {grado}, MSE: {mse_test}, R2: {r2_test, model.score(x_poly_test, y_test)}""")
            print(f"coef: {model.coef_}, intercept: {model.intercept_}")
            num_params = x_poly_train.shape[1]

            aic = len(y_test) * np.log(mse_test) + 2 * num_params
            aic_values.append(aic)

        plt.plot(grados, aic_values, label='AIC')
        plt.xlabel('Grado del polinomio')
        plt.ylabel('AIC')
        plt.title('AIC vs Grado del polinomio')
        plt.legend()
        plt.show()

        grado_optimo = grados[np.argmin(aic_values)]
        print(aic_values)
        print(f'Grado óptimo: {grado_optimo}')


    def test_armonico(self):
        temp = pd.read_csv(r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\temperature\parcela_5.csv")
        temp = temp.set_index('Fecha')

        # Descomposición de la serie temporal
        result = seasonal_decompose(temp['Temperature'], model='additive', period=365)
        print(result, type(result))
        print(result.trend, type(result.trend))
        print(result.seasonal, type(result.seasonal))

        model = ARIMA(result.resid.dropna(), order=(1, 1, 1))
        model_fit = model.fit()
        forecast_resid = model_fit.forecast(steps=365)
        print(forecast_resid, type(forecast_resid))

        forecast = result.trend + result.seasonal + pd.Series(forecast_resid, index=result.resid.dropna().index)

        plt.figure(figsize=(12, 8))
        plt.subplot(511)
        plt.plot(temp['Temperature'], label='Original')
        plt.legend(loc='upper left')
        plt.subplot(512)
        plt.plot(result.trend, label='Tendencia')
        plt.legend(loc='upper left')
        plt.subplot(513)
        plt.plot(result.seasonal, label='Estacionalidad')
        plt.legend(loc='upper left')
        plt.subplot(514)
        plt.plot(result.resid, label='Residuos')
        plt.legend(loc='upper left')
        plt.subplot(515)
        plt.plot(forecast, label='Predicción')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()
        pass


    def test_poly_climate(self):
        data_dir = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\evapotranspiration\parcela_1.csv"
        dest_dir = r"C:\Users\Isai\Documents\Tesis\code\data_analysis\datos\model_predicts\evapotranspiration"
        params_dir = r"C:\Users\Isai\Documents\Tesis\code\data_analysis\fourier\parametros\evapotranspiration"

        df = pd.read_csv(data_dir)
        df['Fecha'] = pd.to_datetime(df['Fecha'])
        df['Dia'] = (df['Fecha'] - df['Fecha'].min()).dt.days

        metadata, modelos = linear_reg_model(df, ['Dia', 'Evapotranspiration'])
        print(metadata)

        x = df['Dia'].values.reshape(-1, 1)
        y = df['Evapotranspiration'].values

        # Crear subfiguras
        fig, ax = plt.subplots(3, 3, figsize=(18, 12))
        fig.suptitle(f'Parcela 1', fontsize=16)
        ax = ax.flatten()

        # Asegurarse de que haya al menos 9 modelos en 'modelos'
        for degree in range(1, 10):
            if degree not in modelos:
                continue

            model_info = modelos[degree]
            poly_degree = model_info[1]
            model = model_info[0]

            x_poly = poly_degree.fit_transform(x)
            y_pred = model.predict(sm.add_constant(x_poly))
            df[f'y_pred_{degree}'] = y_pred

            ax[degree - 1].scatter(df['Dia'], df['Evapotranspiration'], color='lightcoral', label='Datos reales', marker='D', s=3)
            ax[degree - 1].plot(x, y_pred, label=f'Modelo (grado {degree})', linestyle='--', linewidth=0.7,
                                color='royalblue')
            ax[degree - 1].set_title(f'Grado {degree}')
            ax[degree - 1].set_xlabel('Día')
            ax[degree - 1].set_ylabel('Evapotranspiración')
            ax[degree - 1].legend(loc='lower right')

            box_text = f'$R^2$:{model.rsquared:.3f}\nAIC: {model.aic:.3f}\nMSE: {model.mse_model:.3f}'
            ax[degree - 1].text(0.9, 0.9, box_text, transform=ax[degree - 1].transAxes,
                                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'), fontsize=8)

        print(df)
        # df.to_csv(os.path.join(variable_folder_predicts, rf"parcela_{parcela}.csv"), index=False)
        plt.subplots_adjust(hspace=1.2, wspace=1)

        plt.tight_layout()
        # plt.suptitle(f'Parcela {parcela_id}', fontsize=10)
        #plt.savefig(os.path.join(variable_folder_polys, rf"parcela_{parcela}.jpg"), dpi=600,
          #          bbox_inches='tight')
        plt.show()

        pass


    def test_arima_climate(self):
        data_dir = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\relative_humidity\parcela_5.csv"
        data = pd.read_csv(data_dir, index_col='Fecha', parse_dates=True)

        modelo = ARIMA(data['RH12'], order=(1, 1, 0), freq='D')
        modelo_fit = modelo.fit()
        pred = modelo_fit.predict(start='2021-01-01', end='2023-12-31', typ='levels', dynamic=False)
        print(modelo_fit.summary())
        print(modelo_fit.mse, modelo_fit.mae)
        print(f"R2: {r2_score(data['RH12'], modelo_fit.fittedvalues)}")
        print(f"MSE: {mse(data['RH12'], modelo_fit.fittedvalues)}")
        data['pred'] = modelo_fit.fittedvalues
        print(data)
        plt.plot(data['RH12'], label='Datos reales')
        plt.plot(modelo_fit.fittedvalues, label='Ajuste')
        #plt.plot(pred, label='Predicción')
        plt.legend()
        plt.show()

        pass

    def test_holtwinters(self):
        data_dir = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\relative_humidity\parcela_5.csv"
        data = pd.read_csv(data_dir, index_col='Fecha', parse_dates=True)

        model = ExponentialSmoothing(data['RH12'], seasonal='add', seasonal_periods=365)
        results = model.fit()

        # print(results.summary())
        print(f"R2: {r2_score(data['RH12'], results.fittedvalues)}")
        print(f"MSE: {mse(data['RH12'], results.fittedvalues)}")

        plt.plot(data['RH12'], label='Datos reales')
        plt.plot(results.fittedvalues, label='Ajuste')
        plt.legend()
        plt.show()
        pass

if __name__ == '__main__':
    unittest.main()
