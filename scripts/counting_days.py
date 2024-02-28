import pandas as pd
import os


root = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\kc"
destiny_path = r"C:\Users\Isai\Documents\Tesis\code\datos\parcelas\ready_to_analyze"

for file in os.listdir(root):
    print(file)
    full_path = os.path.join(root, file)
    df = pd.read_csv(full_path)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df['Dia'] = (df['Fecha'] - df['Fecha'].min()).dt.days
    df.to_csv(os.path.join(destiny_path, file), index=False)
