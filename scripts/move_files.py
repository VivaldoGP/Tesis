import os
import shutil

main_path = r'G:\Mi unidad\Proyecto_tesis\Tesis_cloudless_definitivo'

parcel_ids = [i.split('_')[0] for i in os.listdir(main_path) if i.endswith('.tif')]
print(parcel_ids)

for id_ in set(parcel_ids):
    try:
        os.makedirs(f"{main_path}/Parcela_{id_}")
    except OSError:
        os.rmdir(f"{main_path}/Parcela_{id_}")
        os.makedirs(f"{main_path}/Parcela_{id_}")

for root, dirs, files in os.walk(main_path):
    for file in files:
        parcel_id = file.split('_')[0]
        for dir_ in dirs:
            if parcel_id == dir_.split('_')[1]:
                shutil.move(os.path.join(main_path, file), f"{main_path}/{dir_}/{file}")
