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



if __name__ == '__main__':
    unittest.main()
