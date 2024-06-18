import pandas as pd
import os
import json
from some_utils.cleanning_data import harvest_dates


root = r"../datos/parcelas/indices_stats_cleaned"
destiny_path = r"../datos/parcelas/ready_to_analyze"
harvest_json = r"../fechas_claves\harvest.json"


with open(harvest_json, encoding='utf-8') as f:
    harvest_data = json.load(f)

for parcela in os.listdir(root):
    filename, ext = parcela.split('.')
    name, parcela_id = filename.split('_')
    data = pd.read_csv(os.path.join(root, parcela))
    data['Fecha'] = pd.to_datetime(data['Fecha'])

    for harvest in harvest_data:
        parcel_id = harvest['id']
        start_date = harvest['start']
        mid_date = harvest['mid']
        end_date = harvest['end']
        if parcel_id == int(parcela_id):
            data_21 = harvest_dates(data, start_date, mid_date)
            data_22 = harvest_dates(data, mid_date, end_date)
            data_21['dia'] = (data_21['Fecha'] - data_21['Fecha'].min()).dt.days + 1
            data_22['dia'] = (data_22['Fecha'] - data_22['Fecha'].min()).dt.days + 1
            data_21.to_csv(os.path.join(os.path.join(destiny_path, 'zafra2021'),
                                        f'parcela_{parcela_id}.csv'), index=False)
            data_22.to_csv(os.path.join(os.path.join(destiny_path, 'zafra2022'),
                                        f'parcela_{parcela_id}.csv'), index=False)
