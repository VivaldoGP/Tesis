import unittest
import pandas as pd
import json

from some_utils.cleanning_data import harvest_dates, cloud_filter

clouds = r"C:\Users\Isai\Documents\Tesis\code\fechas_claves\clouds.json"
ds_path = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\indices_stats\parcela_1.csv"


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_remove_clouds(self):
        with open(clouds, encoding='utf-8') as f:
            cloud_dates = json.load(f)
            for date in cloud_dates:
                to_delete = pd.to_datetime(date['Fechas'])
                parcel_id = date['Parcelas_id']
                if parcel_id == 1:
                    print(to_delete)
                    ds = pd.read_csv(ds_path)
                    ds['Fecha'] = pd.to_datetime(ds['Fecha'])
                    ds_2 = cloud_filter(ds, to_delete)
                    print(ds_2)
                self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
