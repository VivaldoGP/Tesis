from sklearn.linear_model import LinearRegression


def simple_weight_regression(x_data, y_data, weight_data):
    """
    Realiza una regresión lineal ponderada con los datos de entrada.
    :param x_data: Datos de la variable independiente
    :param y_data: Datos de la variable dependiente
    :param weight_data: Datos de los pesos
    :return: Modelo de regresión lineal
    """
    model = LinearRegression()
    model.fit(x_data, y_data, sample_weight=weight_data)
    return model
