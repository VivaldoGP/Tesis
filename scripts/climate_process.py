import pandas as pd
import os
import json
from some_utils.cleanning_data import harvest_dates


data_path = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\precipitation"
destiny_path = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\precipitation_harvest_dates"
harvest_path = r"C:\Users\Isai\Documents\Tesis\code\fechas_claves\harvest.json"


for parcela in os.listdir(data_path):
    filename, ext = parcela.split('.')
    parcela_id = filename.split('_')[-1]
    data = pd.read_csv(os.path.join(data_path, parcela))
    data['Fecha'] = pd.to_datetime(data['Fecha'])

    with open(harvest_path, encoding='utf-8') as f:
        harvest_data = json.load(f)
        for harvest in harvest_data:
            start_date = harvest['start']
            end_date = harvest['end']
            if int(parcela_id) == harvest['id']:
                data = harvest_dates(data, start_date, end_date)
                print(parcela_id, data)
                data.to_csv(os.path.join(destiny_path, f'parcela_{parcela_id}.csv'), index=False)
