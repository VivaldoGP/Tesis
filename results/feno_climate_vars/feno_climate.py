from pathlib import Path
import pandas as pd
import os


zafra = int(input('Zafra: '))
root = Path(rf'../../data_analysis/all_vars/zafra{zafra}')
variable = str(input('Variable: '))
fases = int(input('Fases: germination(1), emergence(2), fast_growth(3), maturation(4): '))
min_max = int(input('Min(1) or Max(2): '))

# Etapas fenológicas
feno = {
    "germination": {"min": 30, "max": 50},
    "emergence": {"min": 80, "max": 120},
    "fast_growth": {"min": 260, "max": 340},
    "maturation": {"min": 320, "max": 480}
    }

mean_list = []

for file in root.iterdir():
    if file.is_file():
        if file.suffix == '.csv':
            parcela_id = file.stem.split('_')[1]
            df = pd.read_csv(file)
            if fases == 1:
                if min_max == 1:
                    df_filtered = df[(df['dias'] >= 1) & (df['dias'] <= feno['germination']['min'])]
                    goal_var = df_filtered[variable].mean()
                    mean_list.append((parcela_id, goal_var))
                    print(parcela_id, goal_var, 'usamos el minimo del germination')
                elif min_max == 2:
                    df_filtered = df[(df['dias'] >= 1) & (df['dias'] <= feno['germination']['max'])]
                    goal_var = df_filtered[variable].mean()
                    mean_list.append((parcela_id, goal_var))
                    print('usamos el maximo del germination')
            elif fases == 2:
                if min_max == 1:
                    df_filtered = df[(df['dias'] >= feno['germination']['min']) & (df['dias'] <= feno['emergence']['min'])]
                    goal_var = df_filtered[variable].mean()
                    mean_list.append((parcela_id, goal_var))
                    print('usamos el minimo del emergence')
                elif min_max == 2:
                    df_filtered = df[(df['dias'] >= feno['germination']['max']) & (df['dias'] <= feno['emergence']['max'])]
                    goal_var = df_filtered[variable].mean()
                    mean_list.append((parcela_id, goal_var))
                    print('usamos el maximo del emergence')

            elif fases == 3:
                if min_max == 1:
                    df_filtered = df[(df['dias'] >= feno['emergence']['min']) & (df['dias'] <= feno['fast_growth']['min'])]
                    goal_var = df_filtered[variable].mean()
                    mean_list.append((parcela_id, goal_var))
                    print('usamos el minimo del fast_growth')
                elif min_max == 2:
                    df_filtered = df[(df['dias'] >= feno['emergence']['max']) & (df['dias'] <= feno['fast_growth']['max'])]
                    goal_var = df_filtered[variable].mean()
                    mean_list.append((parcela_id, goal_var))
                    print('usamos el maximo del fast_growth')

            elif fases == 4:
                if min_max == 1:
                    df_filtered = df[(df['dias'] >= feno['fast_growth']['min']) & (df['dias'] <= feno['maturation']['min'])]
                    goal_var = df_filtered[variable].mean()
                    mean_list.append((parcela_id, goal_var))
                    print('usamos el minimo del maturation')
                elif min_max == 2:
                    df_filtered = df[(df['dias'] >= feno['fast_growth']['max']) & (df['dias'] <= feno['maturation']['max'])]
                    goal_var = df_filtered[variable].mean()
                    mean_list.append((parcela_id, goal_var))
                    print('usamos el maximo del maturation')

mean_df = pd.DataFrame(mean_list, columns=['parcela', variable])

if not os.path.exists(f'../../results/feno_climate_vars/feno_combs/{variable}'):
    os.makedirs(f'../../results/feno_climate_vars/feno_combs/{variable}')
mean_df.to_csv(rf'../../results/feno_climate_vars/feno_combs/{variable}/zafra{zafra}_var_{variable}_fase_{fases}_rango_{min_max}.csv', index=False)
