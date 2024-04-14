import fiona
from shapely.geometry import shape
import xarray as xr

import os
import pandas as pd


folder = str(input("Enter the folder path: "))
var_name = str(input("Enter the variable name: "))
ds_var_name = str(input("Enter the variable name in the dataset: "))

source_data = rf"C:\Users\Isai\Documents\Tesis\code\agrometa_data\{folder}"
puntos_path = r"C:\Users\Isai\Documents\Tesis\code\Parcelas\centroides\centroides.shp"
destiny_path = rf"C:\Users\Isai\Documents\Tesis\code\datos\agroclimate\{folder}"

if not os.path.exists(destiny_path):
    os.makedirs(destiny_path)

parcelas_df = {}

with fiona.open(puntos_path, 'r') as src:
    for file in os.listdir(source_data):
        if file.endswith('.nc'):
            full_path = os.path.join(source_data, file)
            ds = xr.open_dataset(full_path)
            fecha_xarray = ds['time'][0]
            fecha_numpy = fecha_xarray.values
            fecha = fecha_numpy.astype('datetime64[D]')
            # print(fecha)
            for feature in src:
                geom = shape(feature['geometry'])
                x_coord, y_coord = geom.x, geom.y
                pixel_value = ds.sel(lon=x_coord, lat=y_coord, method='nearest')
                parcela_id = feature['properties']['Id']
                # print(parcela_id, pixel_value['Precipitation_Flux'].values[0])

                if parcela_id not in parcelas_df:
                    parcelas_df[parcela_id] = pd.DataFrame(columns=['Fecha', f'{var_name}'])

                parcelas_df[parcela_id] = pd.concat([
                    parcelas_df[parcela_id],
                    pd.DataFrame({'Fecha': [fecha],
                                  f'{var_name}': [pixel_value[f'{ds_var_name}'].values[0]]})
                ])

for id_, df_ in parcelas_df.items():
    df_.to_csv(os.path.join(destiny_path, f"parcela_{id_}.csv"), index=False)
