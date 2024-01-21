import os
import pandas as pd

merged_path = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\merged"
evapo_path = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\evapotranspiration"

parcelas_df = {}

for file in os.listdir(merged_path):
    if file.endswith('.csv'):
        filename, ext = file.split('.')
        name, number = filename.split('_')
        parcelas_df[number] = pd.read_csv(os.path.join(merged_path, file))

for parcela_id, parcela_df in parcelas_df.items():
    parcela_df['Kc'] = 1.15 * parcela_df['ndvi_mean'] + 0.17
    parcela_df['ETc'] = parcela_df['Kc'] * parcela_df['Evapotranspiration']
    parcela_df.to_csv(os.path.join(evapo_path, f"parcela_{parcela_id}.csv"), index=False)
