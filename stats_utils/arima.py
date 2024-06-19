from statsmodels.tsa.arima.model import ARIMA

from pandas import DataFrame


def arima_model(data: DataFrame, order: tuple):

    model = ARIMA(data, order=order, freq='D')
    model_fit = model.fit()

    return model, model_fit
