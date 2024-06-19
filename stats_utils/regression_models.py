from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

import statsmodels.api as sm

import numpy as np

from pandas import DataFrame


def linear_reg_model(data: DataFrame, columns: list, max_degree: int = 10):
    x_data = data[columns[0]].values.reshape(-1, 1)
    y_data = data[columns[1]].values

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    degrees = np.arange(1, max_degree)

    model_metadata = DataFrame(columns=['degree', 'params', 'aic', 'rsquared', 'mse'])
    models = {}

    for degree in degrees:
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        x_poly_train = poly_features.fit_transform(x_train)
        x_poly_test = poly_features.transform(x_test)
        x_poly = poly_features.transform(x_data)

        model = sm.OLS(y_train, sm.add_constant(x_poly_train)).fit()
        models[degree] = (model, poly_features)
        y_pred = model.predict(sm.add_constant(x_poly))

        model_metadata = model_metadata._append({'degree': degree,
                                                 'params': model.params,
                                                 'aic': model.aic,
                                                 'rsquared': model.rsquared,
                                                 'mse': model.mse_model}, ignore_index=True)

    return model_metadata, models


def polinomial_reg(data: DataFrame, columns: list, max_degree: int):
    x_data = data[columns[0]].values.reshape(-1, 1)
    y_data = data[columns[1]].values

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    degrees = np.arange(1, max_degree)

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
    model_metadata_df = DataFrame(model_metadata)

    return model_metadata_df, models


def simple_linear_reg(data: DataFrame, columns: list):
    x_data = data[columns[0]].values.reshape(-1, 1)
    y_data = data[columns[1]].values

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    model = sm.OLS(y_train, sm.add_constant(x_train)).fit()
    y_pred = model.predict(sm.add_constant(x_data))

    return model
