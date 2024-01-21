import os
import pandas as pd

merged_path = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\merged"
evapo_path = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\evapotranspiration"
cleaned_path = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\indices_stats_cleaned"


parcelas_dict = {}
for cleaned in os.listdir(cleaned_path):
    filename, ext = cleaned.split(".")
    name, number = filename.split("_")
    parcelas_dict[number] = pd.read_csv(os.path.join(cleaned_path, cleaned))

et_dict = {}
for evapo in os.listdir(evapo_path):
    filename, ext = evapo.split(".")
    number_ = filename.split("_")[-1]
    et_dict[number_] = pd.read_csv(os.path.join(evapo_path, evapo))

for key, value in parcelas_dict.items():
    value['Fecha'] = value['Fecha'].astype('datetime64[ns]')
    for key_, value_ in et_dict.items():
        value_['Fecha'] = value_['Fecha'].astype('datetime64[ns]')
        if int(key) == int(key_):
            merged = pd.merge(value, value_, on='Fecha')
            # merged.drop(['Unnamed: 0'], axis=1, inplace=True)
            merged.to_csv(os.path.join(merged_path, f"parcela_{key}.csv"), index=False)
            print(key, merged)
