import rasterio

import fiona
from shapely.geometry import shape

import pandas as pd

import os

import datetime

agera_path = r"C:\Users\DELL\Documents\Tesis_sugarCane\pruebas"
puntos_path = r"C:\Users\Isai\Documents\Tesis\Tesis\Parcelas\centroides\centroides.shp"

et_path = r"C:\Users\DELL\PycharmProjects\Tesis\dataframes\et"


def date_from_filename(filename: str):
    try:
        final_part = filename.split('_')[-1]
        date_part, extension = tuple(final_part.split('.'))
        year: int = int(date_part[:4])
        month: int = int(date_part[4:6])
        day: int = int(date_part[6:8])
        return datetime.date(year=year, month=month, day=day)
    except (TypeError, ValueError):
        return None


parcelas_df = {}

for file in os.listdir(agera_path):
    if file.endswith('.tif'):
        full_path = os.path.join(agera_path, file)
        fecha = date_from_filename(filename=full_path)

        with rasterio.open(full_path) as isrc:
            transform = isrc.transform

            with fiona.open(puntos_path, 'r') as src:
                for feature in src:
                    geom = shape(feature['geometry'])
                    x_coord, y_coord = geom.x, geom.y
                    row, col = isrc.index(x_coord, y_coord)
                    pixel_value = isrc.read(1, window=((row, row + 1), (col, col + 1)))

                    parcela_id = feature['properties']['Id']

                    if parcela_id not in parcelas_df:
                        parcelas_df[parcela_id] = pd.DataFrame(columns=['Fecha', 'Valor'])

                    parcelas_df[parcela_id] = pd.concat([
                        parcelas_df[parcela_id],
                        pd.DataFrame({'Fecha': [fecha], 'Valor': [pixel_value[0][0]]})
                    ])

for id_, df_ in parcelas_df.items():
    df_.to_csv(os.path.join(et_path, f"et_parcela_{id_}.csv"), index=False)
    print(id_, df_)