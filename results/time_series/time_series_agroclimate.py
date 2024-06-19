import os
from pathlib import PurePath
import pandas as pd
import matplotlib.pyplot as plt

zafra = int(input('Zafra: '))
misma_escala = input('¿Misma escala? (s/n): ')
var1 = input('Variable 1: ')
var2 = input('Variable 2: ')

modelados = rf"../../data_analysis/all_vars/zafra{zafra}"

modelados_files = [(pd.read_csv(PurePath(modelados, file), parse_dates=True), int(file.split('.')[0].split('_')[1])
                    ) for file in os.listdir(modelados) if file.endswith(".csv")]

for i in modelados_files:
    if var1 in i[0].columns and var2 in i[0].columns:
        if misma_escala == 's':
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(pd.to_datetime(i[0]['Fecha']), i[0][var1], label=var1, color=(150/255, 0, 24/255), linewidth=0.7)
            ax.plot(pd.to_datetime(i[0]['Fecha']), i[0][var2], label=var2, color='g', linewidth=0.7)
            ax.set_title(f'Parcela {i[1]}')
            ax.set_ylabel(f'{var1}')
            ax.set_xlabel('Fecha')
            ax.legend()
            plt.tight_layout()
            export_path = PurePath(rf"../../tesis_img/series_tiempo_agroclimate/zafra{zafra}/{var1}_vs_{var2}")
            if not os.path.exists(export_path):
                os.makedirs(export_path)
            plt.savefig(PurePath(export_path, f"parcela_{i[1]}.pdf"), dpi=100)
            # plt.show()
        elif misma_escala == 'n':
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(pd.to_datetime(i[0]['Fecha']), i[0][var1], label=var1, color=(150/255, 0, 24/255), linewidth=0.7)
            ax1.set_title(f'Parcela {i[1]}')
            ax1.set_ylabel(f'{var1}')
            ax2 = ax1.twinx()
            ax2.plot(pd.to_datetime(i[0]['Fecha']), i[0][var2], label=var2, color='g', linewidth=0.7)
            ax2.set_ylabel(f'{var2}')
            ax1.set_xlabel('Fecha')
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='upper right')
            plt.tight_layout()
            export_path = PurePath(rf"../../tesis_img/series_tiempo_agroclimate/zafra{zafra}/{var1}_vs_{var2}")
            if not os.path.exists(export_path):
                os.makedirs(export_path)
            plt.savefig(PurePath(export_path, f"parcela_{i[1]}.pdf"), dpi=100)

            # plt.show()
