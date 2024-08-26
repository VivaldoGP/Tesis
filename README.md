# Desarrollo del proyecto de tesis

### Organización del repositorio

En este repo se encuentran mayoría de los datos utilizados para los respectivos análisis, los que no se encuentran se
mencionarán en un archivo independiente y se explicará como obtenerlos, ya que el peso es excesivo para subirlo a GitHub.

#### Datos
La carpeta `datos` contiene los datos utilizados para los análisis, estos son:
- `parcelas`
Se divide en multiples carpetas, cada una contiene los datos para cada una de las parcelas, la carpeta principal es
  [parcelas](https://github.com/VivaldoGP/Tesis/tree/main/datos/parcelas), la distribución y contenido son:
  - `indices_stats` contiene los datos de los índices de vegetación calculados para cada una de las parcelas. Es importante
  aclarar que en este punto los datos no están revisados ni han sido filtrados, por lo que se encuentran en su versión 'cruda'.
  - ``indices_stats_cleaned`` contiene los datos de los índices de vegetación calculados para cada una de las parcelas,
  para este punto las fechas que presentan datos inconsistentes han sido ignoradas, se delimitaron al rango de fechas
  correctas y se eliminaron los datos que no cumplen con los criterios de calidad.
  - ``ready_to_analyze`` contiene los datos presentes en la carpeta kc pero con el conteo de los días transcurridos desde
  la primera fecha de la serie temporal, esto con el fin de facilitar el análisis de los datos.

La carpeta `agroclimate` contiene los datos climáticos utilizados para los análisis, estos son:
  - ``evapotranspiration`` contiene los datos de la evapotranspiración de cada una de las parcelas, estos datos tienen
  una resolución temporal de 1 día, y corresponden al periodo de fechas con el que fueron descargados originalmente,
  se considera como la versión cruda de los datos. Su adquisición fue posible con el script 
  [agera5_data.py](https://github.com/VivaldoGP/Tesis/blob/main/scripts/agera5_data.py)
  - ``precipitation`` es el caso idéntico al de la evapotranspiración, pero con los datos de precipitación.
  - ``temperature`` es el caso idéntico al de la evapotranspiración, pero con los datos de temperatura.
  - ``solar_radiation`` lo mismo pero con la radiación solar.
  - ``relative_humidity`` lo mismo pero con la humedad relativa.
  

La carpeta ``data_analysis`` contiene: [data_analysis](https://github.com/VivaldoGP/Tesis/tree/main/data_analysis)
  - ``datos`` carpeta en la que se encuentran los resultados de los análisis:
    - ``model_predicts`` contiene los datos resultantes de los modelos de regresión polinomial en sus diferentes grados
    y los valores 'reales' ya presentes en la carpeta ``ready_to_analyze``.
    - ``arima`` resultados de los modelos ARIMA aplicados a los datos climáticos.
    - ``fourier`` resultados de los modelos de Fourier aplicados a los datos climáticos.
    - ``holtwinters`` resultados de los modelos de Holt-Winters aplicados a los datos climáticos.
  - ``linear_reg`` contiene los scripts que se encargan de realizar la regresión lineal de los datos y también almacena
  los coeficientes de la regresión dividida por zafra y por parcela.
  - ``arima`` lo mismo pero para los datos climáticos.
  - ``fourier`` lo mismo pero para los datos climáticos.
  - ``holtwinters`` lo mismo pero para los datos climáticos.
  - ``notebooks`` contiene los notebooks de Jupyter que se encargan de realizar los análisis de los datos, divididos en
análisis individuales y para la visualización de los datos.
  - ``all_vars`` el conjunto de todos los datos necesarios para los análisis posteriores, divididos por zafra y con los
scripts que unen los archivos previos.

La carpeta ``fechas_clave`` contiene:
  - [clouds.json](https://github.com/VivaldoGP/Tesis/blob/main/fechas_claves/clouds.json) para cada parcela se especifica
  la fecha que se considera que presenta una nube en la escena o que presenta una anomalia desconocida.
  - [harvest.json](https://github.com/VivaldoGP/Tesis/blob/main/fechas_claves/harvest.json) para cada parcela se 
  especifica la fecha de cosecha.

La carpeta ``Parcelas`` contiene la información espacial de las parcelas de estudio, su ubicación en distintos formatos,
sus geometrías y los datos necesarios en formato vectorial apto para SIG y un formato más apto para el uso en Python.
[Parcelas](https://github.com/VivaldoGP/Tesis/tree/main/Parcelas)
- ``centroides`` como su nombre lo dice, son los centroides de cada uno de los polígonos de las parcelas.
- ``poligonos_parcelas`` son los polígonos de las parcelas.
- [poligonos_parcelas.geojson](https://github.com/VivaldoGP/Tesis/blob/main/Parcelas/poligonos_parcelas.geojson) es el 
formato amigable para su apertura en Python y otros lenguajes de programación.


#### Código

La carpeta `scripts` contiene los scripts utilizados para la adquisición de datos, limpieza, análisis y visualización
de los datos, es la carpeta más importante del repositorio, ya que contiene los scripts que permiten que los procesos
se puedan replicar junto a los datos base. [scripts](https://github.com/VivaldoGP/Tesis/tree/main/scripts). Dentro de la
carpeta se encuentra la descripción de cada uno de los scripts y su funcionamiento o directamente desde 
[aquí](https://github.com/VivaldoGP/Tesis/blob/main/scripts/README.md).

### Resultados

La carpeta ``results`` se encuentra dividida en función de las partes fundamentales del proyecto, comenzando por la
revisión estadística de las métricas de los modelos polinomiales y de los realizados en los datos climáticos, después 
se procede con la obtención de los marcadores fenológicos a partir de las series de tiempo y al final el análisis del
rendimiento.