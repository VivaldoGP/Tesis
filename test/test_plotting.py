import unittest
import pandas as pd
from plot_utils.charts import simple_line_plot
from some_utils.cleanning_data import harvest_dates

ds_path = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\indices_stats\parcela_5.csv"
ds_path_2 = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\indices_stats_cleaned\parcela_1.csv"
ds_path_3 = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\merged\parcela_5.csv"
ds_path_4 = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\kc\parcela_3.csv"

class MyTestCase(unittest.TestCase):
    def test_plotting(self):
        ds = pd.read_csv(ds_path_4)
        ds['Fecha'] = pd.to_datetime(ds['Fecha'])
        simple_line_plot(ds, 'Fecha', 'ndvi_mean', 'Parcela 2', 'Fecha', 'ndvi', export=False)
        self.assertEqual(True, True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
