import requests
from datetime import datetime, timedelta


def download_data(url, filename):
    try:
        response = requests.get(url)

        if response.status_code == 200:
            with open(filename, 'wb') as file:
                file.write(response.content)
            print(f"Archivo descargado en: {filename}")
        else:
            print(f"Error al descargar el archivo. Código de estado: {response.status_code}")
    except Exception as error:
        print(f"Ocurrió un error: {str(error)}")


def get_dates_from(start_date, end_date):

    dates = []
    actual_date = start_date

    while actual_date <= end_date:
        dates.append(actual_date.strftime('%Y%m%d'))
        actual_date += timedelta(days=1)

    return dates


try:
    initial_date = datetime(2023, 3, 26)
    final_date = datetime(2023, 5, 1)

    dates_generated = get_dates_from(start_date=initial_date, end_date=final_date)

    for date in dates_generated:
        print(date)
        url_main = f"https://data.apps.fao.org/static/data/c3s/AGERA5_ET0/AGERA5_ET0_{date}.tif"

        local_filename = f"C:/Users/DELL/Documents/Tesis_sugarCane/pruebas/AGERA5_ET0_{date}.tif"

        download_data(url=url_main, filename=local_filename)

finally:
    print("a ver que salió")