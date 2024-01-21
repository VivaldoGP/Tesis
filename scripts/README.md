# Cómo replicar los procedimientos

En esta carpeta llamada **scripts** se encuentran cada uno de los programas que llevan acabo las tareas específicas y funciones necesarias para los pasos posteriores.

# Adquisición de las imágenes

El primer paso es adquirir las imágenes que se van a utilizar en el análisis, para esto se desarrolló el script [fetch_data.py](), que se conecta a los datasets de [Google Earth Engine](https://earthengine.google.com/), filtra las imágenes cuya geometría logra intersectar con la geometría de las parcelas de estudio, por medio de los parámetros establecidos descarta las imágenes que presentan nubes en la escena y descarga imágenes recortadas a la extensión máxima de cada una de las parcelas.

## Adquisición de los datos de Evapotranspiración

El script [agera5_data.py]() descarga los rasters en el rango de fechas establecido, la URL es la proporcionada por el programa copernicus en el siguiente [link](https://data.apps.fao.org/static/data/index.html?prefix=static%2Fdata%2Fc3s%2FAGERA5_ET0).

## Ordenamiento y estructuración de los datos

Las imágenes que se descargaron fueron almacenadas en una sola carpeta, para esto se desarrolló un script simple que ordena las imágenes en una estructura de carpetas en las que cada carpeta representa una parcela y dentro se encuentran imágenes exclusivamente de la zona de estudio de esa parcela. El script es [move_files]().

## Procesamiento de las imágenes

Hasta este punto se tienen imágenes con un nivel de procesamiento 2A, dentro de cada archivo raster se encuentran las bandas necesarias para realizar los análisis que se consideren necesarios, se desarrollaron utilidades y scripts para obtener los datos deseados, que en este caso son los valores estadísticos de los índices espectrales para cada parcela, con una especial consideración, se toma la geometría de cada parcela y se realiza un buffer interno de 5 metros para asegurar que los datos obtenidos correspondan al interior de la parcela y no a los bordes, a continuación se listan las utilidades y scripts con su respectiva función:

- [geopro_tools.py]() realiza el buffer de la geometría y lo almacena en memoria para no tener que guardarlo en el disco, lo cual hace el proceso más dinámico y flexible.
- [spectral_indices.py]() calcula los indices espectrales.
- [process_data]() calcula las estadísticas de cada índice y exporta los valores a un documento tipo csv, esto para la parcela especificada.

## Procesamiento de los datos de ET

Para cada parcela se genera su centroide y se extra el valor del pixel que intersecta con el punto del centroide, se exporta un csv para cada parcela y se obtienen los datos para cada imagen y se almacenan, el script que realiza esta tarea es [process_et_data.py]().


## Limpieza de los datos

Hasta este momento ya tenemos las imágenes y sus datos listos para comenzar a realizar los diferentes análisis deseados pero es fundamental asegurar que los datos actuales con los que se cuentan sean útiles o validos, para esto se deberá 'limpiar' la información con la intención de no incluir en los análisis datos que correspondan a imágenes con presencia de nubes, ya que el primer filtro solo es capaz de descartar las imágenes que en su mascara de nubes identificaban pixeles con esta descripción, en algunos casos este primer filtro no es suficiente y es necesario hacer una depuración. Una primera forma de realizar esta tarea es graficar una serie de tiempo con los datos que ya se tienen y ver el comportamiento de estos, buscando anomalías y posteriormente corroborar individualmente si esos valores que a primera vista son anómalos son en realidad por la presencia de una nube o alguna otra causa es la razón de la anomalía., con esto se puede proceder a generar un archivo que contenga las imágenes que presentan nubes y sus respectivas fechas, esto para cada parcela, así se podrán descartar al momento de realizar los análisis posteriores.
El script que realiza esta tarea es [prepare_data.py](), el cual está estructurado de la siguiente manera:
1. Se indican las rutas con los datos a utilizar. Los csv con los datos de las parcelas sin limpiar, el path de destino
y los archivos que contienen las fechas de las imágenes con nubes y las fechas que delimitan el análisis por parcela.
- [datos/parcelas]()
- [fechas_claves]()

Las funciones que utiliza el script mencionado anteriormente están en el package [some_utils]().

## Unión de los datos

Una vez que tienen los datos limpios y se cuenta con los valores de la evapotranspiración se procede a realizar la unión
de ambos conjuntos de datos, para esto se desarrolló el script [merge_data.py](), el cual realiza la unión de los datos
mediante la columna **fecha**, la cual es común en ambos conjuntos de datos, el resultado es un archivo csv con los datos
de cada parcela y su respectiva evapotranspiración para cada fecha.