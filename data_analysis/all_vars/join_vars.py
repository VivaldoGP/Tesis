import os
import pandas as pd
import numpy as np
import json

zafra = int(input("Zafra: "))

ndvi_coefs_dir = rf"..\linear_reg\coeficientes\zafra{zafra}\ndvi_mean"
ndvi_dir = rf"..\datos\model_predicts\zafra{zafra}\ndvi_mean"

gndvi_coefs_dir = rf"..\linear_reg\coeficientes\zafra{zafra}\gndvi_mean"
gndvi_dir = rf"..\data_analysis\datos\model_predicts\zafra{zafra}\gndvi_mean"

cire_coefs_dir = rf"..\linear_reg\coeficientes\zafra{zafra}\cire_mean"
cire_dir = rf"..\data_analysis\datos\model_predicts\zafra{zafra}\cire_mean"

ndre1_coefs_dir = rf"..\linear_reg\coeficientes\zafra{zafra}\ndre1_mean"
ndre1_dir = rf"..\data_analysis\datos\model_predicts\zafra{zafra}\ndre1_mean"

ndmi_coefs_dir = rf"..\linear_reg\coeficientes\zafra{zafra}\ndmi_mean"
ndmi_dir = rf"..\data_analysis\datos\model_predicts\zafra{zafra}\ndmi_mean"

msi_coefs_dir = rf"..\linear_reg\coeficientes\zafra{zafra}\msi_mean"
msi_dir = rf"..\data_analysis\datos\model_predicts\zafra{zafra}\msi_mean"

for parcela in os.listdir(ndvi_coefs_dir):
    parcela_id = int(parcela.split('_')[1].split('.')[0])
    with open(os.path.join(ndvi_coefs_dir, parcela), 'r') as file:
        coefs = json.load(file)
        print(f"Parcela: {parcela_id}")
        params = coefs[1]['params']
        c_ndvi, b_ndvi, a_ndvi = params[0], params[1], params[2]
        print(f"Coeficientes: {a_ndvi}, {b_ndvi}, {c_ndvi}")

    with open(os.path.join(gndvi_coefs_dir, parcela), 'r') as file:
        coefs = json.load(file)
        print(f"Parcela: {parcela_id}")
        params = coefs[1]['params']
        c_gndvi, b_gndvi, a_gndvi = params[0], params[1], params[2]
        print(f"Coeficientes: {a_gndvi}, {b_gndvi}, {c_gndvi}")

    with open(os.path.join(cire_coefs_dir, parcela), 'r') as file:
        coefs = json.load(file)
        print(f"Parcela: {parcela_id}")
        params = coefs[1]['params']
        c_cire, b_cire, a_cire = params[0], params[1], params[2]
        print(f"Coeficientes: {a_cire}, {b_cire}, {c_cire}")

    with open(os.path.join(ndre1_coefs_dir, parcela), 'r') as file:
        coefs = json.load(file)
        print(f"Parcela: {parcela_id}")
        params = coefs[1]['params']
        c_ndre1, b_ndre1, a_ndre1 = params[0], params[1], params[2]
        print(f"Coeficientes: {a_ndre1}, {b_ndre1}, {c_ndre1}")

    with open(os.path.join(ndmi_coefs_dir, parcela), 'r') as file:
        coefs = json.load(file)
        print(f"Parcela: {parcela_id}")
        params = coefs[1]['params']
        c_ndmi, b_ndmi, a_ndmi = params[0], params[1], params[2]
        print(f"Coeficientes: {a_ndmi}, {b_ndmi}, {c_ndmi}")

    with open(os.path.join(msi_coefs_dir, parcela), 'r') as file:
        coefs = json.load(file)
        print(f"Parcela: {parcela_id}")
        params = coefs[1]['params']
        c_msi, b_msi, a_msi = params[0], params[1], params[2]
        print(f"Coeficientes: {a_msi}, {b_msi}, {c_msi}")

    for file in os.listdir(ndvi_dir):
        if int(file.split('_')[1].split('.')[0]) == parcela_id:
            ndvi = pd.read_csv(os.path.join(ndvi_dir, file), parse_dates=['Fecha'])
            start_date = ndvi['Fecha'].min()
            end_date = ndvi['Fecha'].max()
            print(f"Start: {start_date}, End: {end_date} para la parcela {parcela_id}")
            dias = [start_date + pd.Timedelta(days=i) for i in range((end_date - start_date).days + 1)]
            dias_serie = pd.Series(dias)
            df = pd.DataFrame({'Parcela': parcela_id, 'Fecha': dias_serie, 'dias': np.linspace(1, len(dias),
                                                                                               len(dias))})
            df['ndvi'] = a_ndvi * df['dias'] ** 2 + b_ndvi * df['dias'] + c_ndvi
            df['gndvi'] = a_gndvi * df['dias'] ** 2 + b_gndvi * df['dias'] + c_gndvi
            df['cire'] = a_cire * df['dias'] ** 2 + b_cire * df['dias'] + c_cire
            df['ndre1'] = a_ndre1 * df['dias'] ** 2 + b_ndre1 * df['dias'] + c_ndre1
            print(f"parcela: {parcela_id}, coefs: {a_ndvi}, {b_ndvi}, {c_ndvi}")
            df['ndmi'] = a_ndmi * df['dias'] ** 2 + b_ndmi * df['dias'] + c_ndmi
            print(f"parcela: {parcela_id}, coefs: {a_ndmi}, {b_ndmi}, {c_ndmi}")
            df['msi'] = a_msi * df['dias'] ** 2 + b_msi * df['dias'] + c_msi
            print(f"parcela: {parcela_id}, coefs: {a_msi}, {b_msi}, {c_msi}")

            for temp_file in os.listdir(r"..\datos\fourier\temperature"):
                if int(temp_file.split('_')[1].split('.')[0]) == parcela_id:
                    temp = pd.read_csv(
                        os.path.join(r"..\datos\fourier\temperature",
                                     temp_file), parse_dates=['Fecha'])
                    temp = temp[(temp['Fecha'] >= start_date) & (temp['Fecha'] <= end_date)]
                    df = pd.merge(df, temp, on='Fecha', how='outer')

            for solar_file in os.listdir(
                    r"..\datos\fourier\solar_radiation"):
                if int(solar_file.split('_')[1].split('.')[0]) == parcela_id:
                    solar = pd.read_csv(
                        os.path.join(r"..\datos\fourier\solar_radiation",
                                     solar_file), parse_dates=['Fecha'])
                    solar = solar[(solar['Fecha'] >= start_date) & (solar['Fecha'] <= end_date)]
                    df = pd.merge(df, solar, on='Fecha', how='outer')

            for precip_file in os.listdir(r"../..\datos\agroclimate\precipitation"):
                if int(precip_file.split('_')[1].split('.')[0]) == parcela_id:
                    precip = pd.read_csv(
                        os.path.join(r"../..\datos\agroclimate\precipitation", precip_file),
                        parse_dates=['Fecha'])
                    precip = precip[(precip['Fecha'] >= start_date) & (precip['Fecha'] <= end_date)]
                    precip['precip_acum'] = precip['precip'].cumsum()
                    df = pd.merge(df, precip, on='Fecha', how='outer')

            for evapo_file in os.listdir(
                    r"..\datos\fourier\evapotranspiration"):
                if int(evapo_file.split('_')[1].split('.')[0]) == parcela_id:
                    evapo = pd.read_csv(os.path.join(
                        r"..\datos\fourier\evapotranspiration",
                        evapo_file), parse_dates=['Fecha'])
                    evapo = evapo[(evapo['Fecha'] >= start_date) & (evapo['Fecha'] <= end_date)]
                    evapo['et_acum'] = evapo['et'].cumsum()
                    evapo['et_acum_ajustados'] = evapo['ajustados_et'].cumsum()

                    df = pd.merge(df, evapo, on='Fecha', how='outer')

            for rh_file in os.listdir(
                    r"..\datos\holtwinters\relative_humidity"):
                if int(rh_file.split('_')[1].split('.')[0]) == parcela_id:
                    rh = pd.read_csv(os.path.join(
                        r"..\datos\holtwinters\relative_humidity",
                        rh_file), parse_dates=['Fecha'])
                    rh = rh[(rh['Fecha'] >= start_date) & (rh['Fecha'] <= end_date)]
                    df = pd.merge(df, rh, on='Fecha', how='outer')
                    print(df.columns)

                    df.to_csv(
                        rf"zafra{zafra}/parcela_{parcela_id}.csv",
                        index=False)
