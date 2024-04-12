## Análisis de los datos

Se tienen como referencia 2 tipos de análisis dependiendo de los datos, para los datos climatológicos se aplica
el análisis de Fourier con la Transformada Rápida de Fourier (FFT). Teniendo como resultado los elementos para generar
una función sinusoidal que se ajuste a los datos.

Para los datos de las series de tiempo de las parcelas se aplica una regresión polinomial de grado n, donde n es el
grado del polinomio que mejor se ajuste a los datos, comenzando desde una regresión lineal hasta una regresión
polinomial de grado 6. Los valores de los coeficientes de la regresión se guardan en un archivo .json para su posterior
análisis.