import os
import json
import pandas as pd

from some_utils.cleanning_data import harvest_dates, cloud_filter, treshold_data


raw_data_path = r"../datos/parcelas/indices_stats"
clouds_json = r"../fechas_claves/clouds.json"
harvest_json = r"../fechas_claves/harvest.json"
prepared_data_path = r"../datos/parcelas/indices_stats_cleaned"


for parcela in os.listdir(raw_data_path):
    filename, ext = parcela.split('.')
    name, parcela_id = filename.split('_')
    data = pd.read_csv(os.path.join(raw_data_path, parcela))
    data['Fecha'] = pd.to_datetime(data['Fecha'])
    print(data)

    with open(clouds_json, encoding='utf-8') as f:
        cloud_dates = json.load(f)
        for date in cloud_dates:
            to_delete = pd.to_datetime(date['Fechas'])
            parcel_id = date['Parcelas_id']
            if parcel_id == int(parcela_id):
                data = cloud_filter(data, to_delete)
                print(data)
    with open(harvest_json, encoding='utf-8') as f:
        harvest_data = json.load(f)
        for harvest in harvest_data:
            parcel_id = harvest['id']
            start_date = harvest['start']
            end_date = harvest['end']
            if parcel_id == int(parcela_id):
                data = harvest_dates(data, start_date, end_date)
                data = treshold_data(data, 0, 'ndvi_mean')
                # data['dia'] = (data['Fecha'] - data['Fecha'].min()).dt.days
                print(f'id: {parcela_id}', data)
                data.to_csv(os.path.join(prepared_data_path, f'parcela_{parcela_id}.csv'), index=False)
