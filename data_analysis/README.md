El archivo [find_degree.py]() calcula el polinomio que mejor describa el comportamiento de los
datos de entrada, iterando desde el grado 1 que es un modelo lineal 
hasta el grado 10, que es un modelo polinomial de grado 10.
Para esto se dividen los datos en dos conjuntos, uno de entrenamiento y otro de prueba, las 
variables son los días transcurridos y el valor del ndvi promedio para
la parcela en esa fecha/día.
Se utiliza el metodo de OLS de statsmodels para calcular el polinomio que mejor se ajuste a los datos.
Se contrastan los datos con el criterio de Akaike y se elige el polinomio que mejor se ajuste a los datos.
Se grafican los datos y el polinomio que mejor se ajusta a los datos.