import os
from pathlib import PurePath

import pandas as pd


for file in os.listdir(r'C:\Users\Isai\Documents\Tesis\code\data_analysis\all_vars\zafra2022'):
    data = pd.read_csv(PurePath(r'C:\Users\Isai\Documents\Tesis\code\data_analysis\all_vars\zafra2022', file))
    data['kc'] = 1.15 * data['ndvi'] + 0.17
    data['etc'] = data['kc'] * data['et']
    data['etc_acum'] = data['etc'].cumsum()
    data.to_csv(PurePath(r'C:\Users\Isai\Documents\Tesis\code\data_analysis\all_vars\zafra2022', file), index=False)
