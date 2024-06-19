from statsmodels.tsa.holtwinters import ExponentialSmoothing

from pandas import DataFrame


def holtwinters(data: DataFrame, trend: str = 'add', seasonal: str = 'add', seasonal_periods: int = 365):
    model = ExponentialSmoothing(data, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods, freq='D')
    model_fit = model.fit()

    return model, model_fit
