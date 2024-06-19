import os

import pandas as pd

zafra = 2021
for file in os.listdir(rf'zafra{zafra}'):
    data = pd.read_csv(os.path.join(rf'zafra{zafra}', file), parse_dates=['Fecha'])
    data['kc'] = (1.15 * data['ndvi']) + 0.17
    data['etc'] = data['kc'] * data['et']
    data['etc_acum'] = data['etc'].cumsum()
    data['et_acum'] = data['et'].cumsum()
    print(data)
    data.to_csv(os.path.join(rf'zafra{zafra}/{file}'), index=False)
