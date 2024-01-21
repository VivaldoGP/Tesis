from pandas import DataFrame
import datetime
from pandas import DatetimeIndex


def harvest_dates(data: DataFrame, start_date: datetime.date, end_date: datetime.date):
    """Limita el dataframe a un rango de fechas establecido y lo ordena de forma ascendente.

    Args:
        data (DataFrame): Dataframe con los datos a limpiar.
        start_date (datetime.date): Fecha de inicio del rango.
        end_date (datetime.date): Fecha de fin del rango.
    """
    data['Fecha'] = data['Fecha'].astype('datetime64[ns]')
    data = data[(data['Fecha'] > start_date) & (data['Fecha'] <= end_date)]
    data = data.sort_values(by=['Fecha'], ascending=True)

    return data


def cloud_filter(data: DataFrame,  cloud_dates: DatetimeIndex):
    """Elimina los datos que coincidan con las fechas de nubes.

    Args:
        data (DataFrame): Dataframe con los datos a limpiar.
        cloud_dates (list): Lista de fechas con nubes.
    """
    data = data[~data['Fecha'].isin(cloud_dates)]

    return data


def treshold_data(data: DataFrame, treshold: int | float, treshold_column: str):
    """Descarta los datos menores al valor establecido del umbral.

    Args:
        data (DataFrame): Dataframe con los datos a limpiar.
        treshold (int): Valor umbral para eliminar los datos.
        treshold_column (str): Nombre de la columna a utilizar como umbral.
    """
    data = data[data[treshold_column] > treshold]

    return data
