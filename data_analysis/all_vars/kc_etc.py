import os
from pathlib import PurePath

import pandas as pd

zafra = 2022
for file in os.listdir(rf'C:\Users\Isai\Documents\Tesis\code\data_analysis\all_vars\zafra{zafra}'):
    data = pd.read_csv(PurePath(fr'C:\Users\Isai\Documents\Tesis\code\data_analysis\all_vars\zafra{zafra}', file))
    data['kc'] = (1.15 * data['ndvi']) + 0.17
    data['etc'] = data['kc'] * data['et']
    data['etc_acum'] = data['etc'].cumsum()
    data['et_acum'] = data['et'].cumsum()
    data.to_csv(PurePath(fr'C:\Users\Isai\Documents\Tesis\code\data_analysis\all_vars\zafra{zafra}', file), index=False)
