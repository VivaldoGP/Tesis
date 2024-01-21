import os
import pandas as pd

merged_path = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\merged"
kc_path = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\kc"

parcelas_df = {}

for i in os.listdir(merged_path):
    filename, ext = i.split('.')
    name, number = filename.split('_')
    parcelas_df[number] = pd.read_csv(os.path.join(merged_path, i))

for parcela_id, parcela_df in parcelas_df.items():
    parcela_df['Kc'] = 1.15 * parcela_df['ndvi_mean'] + 0.17
    parcela_df['ETc'] = parcela_df['Kc'] * parcela_df['Evapotranspiration']
    parcela_df.to_csv(os.path.join(kc_path, f"parcela_{parcela_id}.csv"), index=False)
    print(parcela_df)