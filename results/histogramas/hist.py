import pandas as pd
import numpy as np
import os
from pathlib import PurePath

zafras = int(input('Zafra: '))
all_vars_files = fr'../../data_analysis/all_vars/zafra{zafras}'
all_vars = [(pd.read_csv(PurePath(all_vars_files, file)), int(file.split('.')[0].split('_')[1]))
            for file in os.listdir(all_vars_files) if file.endswith(".csv")]

bins = [0, 0.1, 25, 50, float('inf')]
for i in all_vars:
    hist, bin_edges = np.histogram(i[0]['precip'], bins=bins)
    df_hist = pd.DataFrame(
        {
            'Intervalo': ['0-0.1', '0.1-25', '25.1-50', '>50'],
            'Frecuencia': hist
        }
    )
    df_hist.to_csv(f'../../results/histogramas/data/zafra{zafras}/parcela_{i[1]}.csv', index=False)
