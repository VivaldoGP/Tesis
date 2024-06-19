import os
from pathlib import PurePath
import pandas as pd
import matplotlib.pyplot as plt

zafra = int(input('Zafra: '))
option = int(input('1. Reals\n2. Con modelo\n'))

focus_vars = ['ndvi', 'gndvi', 'msi', 'ndmi', 'cire', 'ndre1']

reales = rf"../../datos/parcelas/ready_to_analyze/zafra{zafra}"
modelados = rf"../../data_analysis/all_vars/zafra{zafra}"

modelados_files = [(pd.read_csv(PurePath(modelados, file), parse_dates=True), int(file.split('.')[0].split('_')[1])
                    ) for file in os.listdir(modelados) if file.endswith(".csv")]
reales_files = [(pd.read_csv(PurePath(reales, file), parse_dates=True), int(file.split('.')[0].split('_')[1]))
                for file in os.listdir(reales) if file.endswith(".csv")]

for i in modelados_files:
    for j in reales_files:
        if i[1] == j[1]:
            for var in focus_vars:
                if var in i[0].columns:
                    if option == 2:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(pd.to_datetime(i[0]['Fecha']), i[0][var], label='Modelado', color=(150/255, 0, 24/255))
                        ax.plot(pd.to_datetime(j[0]['Fecha']), j[0][f"{var}_mean"], label='Real', color='royalblue')
                        ax.set_title(f'Parcela {i[1]} - Índice {var.upper()}')
                        ax.set_ylabel(f'{var.upper()}')
                        ax.set_xlabel('Fecha')
                        ax.legend()
                        plt.tight_layout()
                        export_path = PurePath(rf"../../tesis_img/series_tiempo_real_modelo/zafra{zafra}/{var}")
                        if not os.path.exists(export_path):
                            os.makedirs(export_path)
                        plt.savefig(PurePath(export_path, f"parcela_{i[1]}_{var}.pdf"), dpi=100)
                    elif option == 1:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(pd.to_datetime(j[0]['Fecha']), j[0][f"{var}_mean"], label='Real', color='royalblue')
                        ax.set_title(f'Parcela {i[1]} - Índice {var.upper()}')
                        ax.set_ylabel(f'{var.upper()}')
                        ax.set_xlabel('Fecha')
                        ax.legend()
                        plt.tight_layout()
                        export_path = PurePath(rf"../../tesis_img/series_tiempo_real/zafra{zafra}/{var}")
                        if not os.path.exists(export_path):
                            os.makedirs(export_path)
                        plt.savefig(PurePath(export_path, f"parcela_{i[1]}_{var}.pdf"), dpi=100)
