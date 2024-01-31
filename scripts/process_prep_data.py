import fiona
from shapely.geometry import shape
import xarray as xr

import os
import pandas as pd
import datetime
import geopandas as gpd

precipitation_path = r"C:\Users\Isai\Documents\Tesis\code\precipitation\precip"
puntos_path = r"C:\Users\Isai\Documents\Tesis\code\Parcelas\centroides\centroides.shp"
precipitation_data = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\precipitation"

gdf = gpd.read_file(puntos_path)

parcelas_df = {}

for file in os.listdir(precipitation_path):
    if file.endswith('.nc'):
        full_path = os.path.join(precipitation_path, file)
        ds = xr.open_dataset(full_path)
        fecha_xarray = ds['time'][0]
        fecha_numpy = fecha_xarray.values
        fecha = fecha_numpy.astype('datetime64[D]')
        #print(fecha)

        with fiona.open(puntos_path, 'r') as src:
            for feature in src:
                geom = shape(feature['geometry'])
                x_coord, y_coord = geom.x, geom.y
                pixel_value = ds.sel(lon=x_coord, lat=y_coord, method='nearest')
                parcela_id = feature['properties']['Id']
                #print(parcela_id, pixel_value['Precipitation_Flux'].values[0])

                if parcela_id not in parcelas_df:
                    parcelas_df[parcela_id] = pd.DataFrame(columns=['Fecha', 'Precipitation'])

                parcelas_df[parcela_id] = pd.concat([
                    parcelas_df[parcela_id],
                    pd.DataFrame({'Fecha': [fecha], 'Precipitation': [pixel_value['Precipitation_Flux'].values[0]]})
                ])

for id_, df_ in parcelas_df.items():
    df_.to_csv(os.path.join(precipitation_data, f"parcela_{id_}.csv"), index=False)
    print(id_, df_)
