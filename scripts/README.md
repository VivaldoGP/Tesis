# Cómo replicar los procedimientos

En esta carpeta llamada **scripts** se encuentran cada uno de los programas que llevan a cabo las tareas específicas y 
funciones necesarias para los pasos posteriores.

# Adquisición de las imágenes

El primer paso es adquirir las imágenes que se van a utilizar en el análisis, para esto se desarrolló el script
[ee_fetch_data.py](https://github.com/VivaldoGP/Tesis/blob/main/scripts/ee_fecth_images.py), 
que se conecta a los datasets de [Google Earth Engine](https://earthengine.google.com/), filtra las imágenes cuya geometría logra intersectar 
con la geometría de las parcelas de estudio, por medio de los parámetros establecidos descarta las imágenes que 
presentan nubes en la escena y descarga imágenes recortadas a la extensión máxima de cada una de las parcelas.

## Adquisición de los datos de Evapotranspiración

El script [get_agera5_data.py](https://github.com/VivaldoGP/Tesis/blob/main/scripts/agera5_data.py) descarga los rasters en
el rango de fechas establecido, la URL es la proporcionada por el 
programa copernicus en el siguiente [link](https://data.apps.fao.org/static/data/index.html?prefix=static%2Fdata%2Fc3s%2FAGERA5_ET0).

## Ordenamiento y estructuración de los datos

Las imágenes que se descargaron fueron almacenadas en una sola carpeta, para esto se desarrolló un script simple que 
ordena las imágenes en una estructura de carpetas en las que cada carpeta representa una parcela y dentro se encuentran 
imágenes exclusivamente de la zona de estudio de esa parcela.
El script es [move_files](https://github.com/VivaldoGP/Tesis/blob/main/scripts/move_files.py).

## Procesamiento de las imágenes

Hasta este punto se tienen imágenes con un nivel de procesamiento 2A, dentro de cada archivo raster se encuentran las 
bandas necesarias para realizar los análisis que se consideren necesarios, se desarrollaron utilidades y scripts para 
obtener los datos deseados, que en este caso son los valores estadísticos de los índices espectrales para cada parcela, 
con una especial consideración, se toma la geometría de cada parcela y se realiza un buffer interno de 5 metros para 
asegurar que los datos obtenidos correspondan al interior de la parcela y no a los bordes, a continuación se listan 
las utilidades y scripts con su respectiva función:

- [geopro_tools.py](https://github.com/VivaldoGP/Tesis/blob/main/vector_utils/geopro_tools.py) realiza el buffer de la
geometría y lo almacena en memoria para no tener que guardarlo en el disco, lo cual hace el proceso más dinámico y flexible.
- [spectral_indices.py](https://github.com/VivaldoGP/Tesis/blob/main/raster_utils/spectral_indices.py) calcula los 
índices espectrales.
- [process_data.py](https://github.com/VivaldoGP/Tesis/blob/main/scripts/process_data.py) calcula las estadísticas de
cada índice y exporta los valores a un documento tipo csv, esto para la parcela especificada.

## Procesamiento de los datos de ET

Para cada parcela se genera su centroide y se extrae el valor del pixel que intersecta con el punto del centroide, 
se exporta un csv para cada parcela y se obtienen los datos para cada imagen y se almacenan, el script que realiza 
esta tarea es [process_et_data.py](https://github.com/VivaldoGP/Tesis/blob/main/scripts/process_et_data.py).
 

## Procesamiento de los datos meteorológicos
Se realiza el mismo procedimiento, pero con los otros datos, las variables son:
- Temperature_Air_2m_Mean_24h
- Solar_Radiation_Flux
- Relative_Humidity_2m_12h
- Precipitation_Flux

El script que realiza esta tarea es [process_climate_data.py](https://github.com/VivaldoGP/Tesis/blob/main/scripts/process_prep_data.py),
la diferencia con el anterior es que los datos de entrada se encuentran en formato NetCDF, por lo que se utiliza la 
librería xarray para realizar la extracción de los datos y se exportan a un archivo csv. Por esa razón se tienen dos 
scripts distintos que tienen la misma finalidad.


## Limpieza de los datos

Hasta este momento ya tenemos las imágenes y sus datos listos para comenzar a realizar los diferentes análisis deseados
pero es fundamental asegurar que los datos actuales con los que se cuentan sean útiles o válidos, para esto se deberá 
'limpiar' la información con la intención de no incluir en los análisis datos que correspondan a imágenes con presencia 
de nubes, ya que el primer filtro solo es capaz de descartar las imágenes que en su máscara de nubes identificaban 
píxeles con esta descripción, en algunos casos este primer filtro no es suficiente y es necesario hacer una depuración. 
Una primera forma de realizar esta tarea es graficar una serie de tiempo con los datos que ya se tienen y ver el 
comportamiento de estos, buscando anomalías y posteriormente corroborar individualmente si esos valores que a primera 
vista son anómalos son en realidad por la presencia de una nube o alguna otra causa es la razón de la anomalía., 
con esto se puede proceder a generar un archivo que contenga las imágenes que presentan nubes y sus respectivas fechas, 
esto para cada parcela, así se podrán descartar al momento de realizar los análisis posteriores.
El script que realiza esta tarea es [prepare_data.py](https://github.com/VivaldoGP/Tesis/blob/main/scripts/prepare_data.py),
el cual está estructurado de la siguiente manera:


## División en temporadas

Los datos presentes corresponden a 2 temporadas diferentes, por lo que se deben dividir mediante el script
[split_seasons.py]() que se encarga de dividir los datos en dos temporadas, de acuerdo a las fechas de cosecha, al ser
consecutivas el final de una será la primera de la siguiente.


# Diagrama de flujo

```mermaid
---
title: Diagrama de flujo
---
flowchart TD
    A[Obtención de datos] -->B[Limpieza de datos]
    B --> C[Procesamiento de las imágenes]
    C --> D[Procesamiento de los datos de ET]
    D --> E[Procesamiento de los datos meteorológicos]
    E --> F[Limpieza de los datos]
    F --> G[Unión de los datos]
    G --> H[Cálculo de la evapotranspiración]
    H --> I[Última preparación de los datos]
    I --> J[Delimitación de los datos meteorológicos]
    J --> K[Análisis de los datos]
    K --> L[Resultados]
```