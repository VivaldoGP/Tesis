import unittest
import pandas as pd
import os
import json
import numpy as np
from datetime import datetime

from pandas import DataFrame

ndvi = pd.read_csv(r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\indices_stats_cleaned\parcela_5.csv")
ndvi_d2 = pd.read_csv(r"C:\Users\Isai\Documents\Tesis\code\data_analysis\datos\model_predicts\parcela_5.csv")

pd.set_option('display.max_rows', None)  # Muestra todas las filas
pd.set_option('display.max_columns', None)

class MyTestCase(unittest.TestCase):
    def test_something(self):

        id_max = ndvi['ndvi_mean'].idxmax()
        id_min = ndvi['ndvi_mean'].idxmin()
        id_max_2 = ndvi_d2['y_pred_2'].idxmax()
        id_min_2 = ndvi_d2['y_pred_2'].idxmin()
        max_date = ndvi.loc[id_max, 'Fecha']
        max_date_2 = ndvi_d2.loc[id_max_2, 'Fecha']
        min_date = ndvi.loc[id_min, 'Fecha']
        min_date_2 = ndvi_d2.loc[id_min_2, 'Fecha']
        max_value = ndvi.loc[id_max, 'ndvi_mean']
        max_value_2 = ndvi_d2.loc[id_max_2, 'y_pred_2']
        min_value = ndvi.loc[id_min, 'ndvi_mean']
        min_value_2 = ndvi_d2.loc[id_min_2, 'y_pred_2']
        print(f"El valor máximo de NDVI es: {max_value} en la fecha: {max_date}")
        print(f"El valor mínimo de NDVI es: {min_value} en la fecha: {min_date}")
        print(f"El valor máximo de NDVI_2 es: {max_value_2} en la fecha: {max_date_2}")
        print(f"El valor mínimo de NDVI_2 es: {min_value_2} en la fecha: {min_date_2}")
        print(f"El rango de valores de NDVI es: {max_value - min_value}")
        print(f"El rango de valores de NDVI_2 es: {max_value_2 - min_value_2}")
        print(f"El rango de fechas es: {(pd.to_datetime(max_date) - pd.to_datetime(min_date)).days}")
        print(f"El rango de fechas_2 es: {(pd.to_datetime(max_date_2) - pd.to_datetime(min_date_2)).days}")
        self.assertEqual(True, True)  # add assertion here

    def test_cor(self):
        parcela = int(input('Ingrese el id de la parcela: '))
        # export_data = str(input('¿Desea exportar los datos? (y/n): '))

        vi_data = rf"C:\Users\Isai\Documents\Tesis\code\data_analysis\datos\model_predicts\parcela_{parcela}.csv"
        temp = rf"C:\Users\Isai\Documents\Tesis\code\data_analysis\datos\fourier\temperature\parcela_{parcela}.csv"
        solar_rad = rf"C:\Users\Isai\Documents\Tesis\code\data_analysis\datos\fourier\solar_radiation\parcela_{parcela}.csv"
        precip = rf"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\precipitation\parcela_{parcela}.csv"
        evapo = rf"C:\Users\Isai\Documents\Tesis\code\data_analysis\datos\fourier\evapotranspiration\parcela_{parcela}.csv"
        rh = rf"C:\Users\Isai\Documents\Tesis\code\data_analysis\datos\holtwinters\relative_humidity\parcela_{parcela}.csv"

        vi = pd.read_csv(vi_data, parse_dates=['Fecha'])
        vi = vi[['Fecha', 'Parcela', 'ndvi_mean', 'y_pred_2']]

        inicio = vi['Fecha'].min()
        fin = vi['Fecha'].max()

        data = pd.DataFrame({'dias': np.linspace(1, 395, 395), 'Fecha': pd.date_range(start=inicio, end=fin, periods=395)})
        data['ndvi'] = -1.2686e-05 * data['dias'] ** 2 + 0.0058008862 * data['dias'] + 0.1160076519
        data['Fecha'] = data['Fecha'].dt.date

        temp = pd.read_csv(temp, parse_dates=['Fecha'])
        temp = temp[(temp['Fecha'] >= inicio) & (temp['Fecha'] <= fin)]

        solar_rad = pd.read_csv(solar_rad, parse_dates=['Fecha'])
        solar_rad = solar_rad[(solar_rad['Fecha'] >= inicio) & (solar_rad['Fecha'] <= fin)]

        precip = pd.read_csv(precip, parse_dates=['Fecha'])
        precip = precip[(precip['Fecha'] >= inicio) & (precip['Fecha'] <= fin)]
        precip['acum'] = precip['Precipitation'].cumsum()

        evapo = pd.read_csv(evapo, parse_dates=['Fecha'])
        evapo = evapo[(evapo['Fecha'] >= inicio) & (evapo['Fecha'] <= fin)]
        evapo['acum'] = evapo['Evapotranspiration'].cumsum()

        rh = pd.read_csv(rh, parse_dates=['Fecha'])
        rh = rh[(rh['Fecha'] >= inicio) & (rh['Fecha'] <= fin)]

        dfs = [temp, solar_rad, precip, evapo, rh]

        for df in dfs:
            data = pd.merge(data, df, on='Fecha', how='outer')

        data.to_csv(rf"C:\Users\Isai\Documents\Tesis\code\garbage_tests\parcela_{parcela}.csv", index=False)
        corr = data.corr()
        print(corr)
        # corr.to_csv(rf"C:\Users\Isai\Documents\Tesis\code\garbage_tests\corr_{parcela}.csv", index=False)

    def test_replicate_data(self):

        coefs_dir = r"C:\Users\Isai\Documents\Tesis\code\data_analysis\linear_reg\coeficientes\zafra2021\ndvi_mean"
        ndvi_data = r"C:\Users\Isai\Documents\Tesis\code\data_analysis\datos\model_predicts\zafra2021\ndvi_mean"

        with open(r"C:\Users\Isai\Documents\Tesis\code\fechas_claves\harvest.json", 'r') as file:
            harvest_dates = json.load(file)

        for parcela in os.listdir(coefs_dir):
            parcela_id = int(parcela.split('_')[1].split('.')[0])
            with open(os.path.join(coefs_dir, parcela), 'r') as file:
                coefs = json.load(file)
                print(f"Parcela: {parcela_id}")
                params = coefs[1]['params']
                c, b, a = params[0], params[1], params[2]
                print(f"Coeficientes: {a}, {b}, {c}")

            """
            for parcela_ in harvest_dates:
                if parcela_id == parcela_['id']:
                    start = datetime.strptime(parcela_['start'], '%Y-%m-%d')
                    end = datetime.strptime(parcela_['end'], '%Y-%m-%d')
                    dias = [start + pd.Timedelta(days=i) for i in range((end - start).days + 1)]
                    print(len(dias))
                    print(f"Start: {start}, End: {end} para la parcela {parcela_id}")
            """

            for file in os.listdir(ndvi_data):
                if int(file.split('_')[1].split('.')[0]) == parcela_id:
                    ndvi = pd.read_csv(os.path.join(ndvi_data, file), parse_dates=['Fecha'])
                    start_date = ndvi['Fecha'].min()
                    end_date = ndvi['Fecha'].max()
                    print(f"Start: {start_date}, End: {end_date} para la parcela {parcela_id}")
                    dias = [start_date + pd.Timedelta(days=i) for i in range((end_date - start_date).days + 1)]
                    dias_serie = pd.Series(dias)

                    df = pd.DataFrame({'Fecha': dias_serie, 'dias': np.linspace(1, len(dias), len(dias))})
                    df['ndvi'] = a * df['dias'] ** 2 + b * df['dias'] + c

                    for temp_file in os.listdir(r"C:\Users\Isai\Documents\Tesis\code\data_analysis\datos\fourier\temperature"):
                        if int(temp_file.split('_')[1].split('.')[0]) == parcela_id:
                            temp = pd.read_csv(os.path.join(r"C:\Users\Isai\Documents\Tesis\code\data_analysis\datos\fourier\temperature", temp_file), parse_dates=['Fecha'])
                            temp = temp[(temp['Fecha'] >= start_date) & (temp['Fecha'] <= end_date)]
                            df = pd.merge(df, temp, on='Fecha', how='outer')

                    for solar_file in os.listdir(r"C:\Users\Isai\Documents\Tesis\code\data_analysis\datos\fourier\solar_radiation"):
                        if int(solar_file.split('_')[1].split('.')[0]) == parcela_id:
                            solar = pd.read_csv(os.path.join(r"C:\Users\Isai\Documents\Tesis\code\data_analysis\datos\fourier\solar_radiation", solar_file), parse_dates=['Fecha'])
                            solar = solar[(solar['Fecha'] >= start_date) & (solar['Fecha'] <= end_date)]
                            df = pd.merge(df, solar, on='Fecha', how='outer')

                    for precip_file in os.listdir(r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\precipitation"):
                        if int(precip_file.split('_')[1].split('.')[0]) == parcela_id:
                            precip = pd.read_csv(os.path.join(r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\precipitation", precip_file), parse_dates=['Fecha'])
                            precip = precip[(precip['Fecha'] >= start_date) & (precip['Fecha'] <= end_date)]
                            precip['acum'] = precip['Precipitation'].cumsum()
                            df = pd.merge(df, precip, on='Fecha', how='outer')

                    for evapo_file in os.listdir(r"C:\Users\Isai\Documents\Tesis\code\data_analysis\datos\fourier\evapotranspiration"):
                        if int(evapo_file.split('_')[1].split('.')[0]) == parcela_id:
                            evapo = pd.read_csv(os.path.join(r"C:\Users\Isai\Documents\Tesis\code\data_analysis\datos\fourier\evapotranspiration", evapo_file), parse_dates=['Fecha'])
                            evapo = evapo[(evapo['Fecha'] >= start_date) & (evapo['Fecha'] <= end_date)]
                            evapo['acum_ref'] = evapo['Evapotranspiration'].cumsum()
                            evapo['acum_ajustados'] = evapo['Ajustados'].cumsum()

                            df = pd.merge(df, evapo, on='Fecha', how='outer')

                    for rh_file in os.listdir(r"C:\Users\Isai\Documents\Tesis\code\data_analysis\datos\holtwinters\relative_humidity"):
                        if int(rh_file.split('_')[1].split('.')[0]) == parcela_id:
                            rh = pd.read_csv(os.path.join(r"C:\Users\Isai\Documents\Tesis\code\data_analysis\datos\holtwinters\relative_humidity", rh_file), parse_dates=['Fecha'])
                            rh = rh[(rh['Fecha'] >= start_date) & (rh['Fecha'] <= end_date)]
                            df = pd.merge(df, rh, on='Fecha', how='outer')
                            print(df)
                            #df.to_csv(rf"C:\Users\Isai\Documents\Tesis\code\data_analysis\all_vars\zafra2021\parcela_{parcela_id}.csv", index=False)





if __name__ == '__main__':
    unittest.main()
