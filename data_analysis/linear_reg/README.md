El archivo [find_degree.py]() calcula el polinomio que mejor describa el comportamiento de los
datos de entrada, iterando desde el grado 1 que es un modelo lineal 
hasta el grado 6, que es un modelo polinomial de grado 6.
Para esto se dividen los datos en dos conjuntos, uno de entrenamiento y otro de prueba, las 
variables son los días transcurridos y el valor del índice de vegetación promedio para
la parcela en esa fecha/día.
Se utiliza el metodo de OLS de statsmodels para calcular el polinomio que mejor se ajuste a los datos.
Se contrastan los datos con el criterio de Akaike y se elige el grado del polinomio que mejor se ajuste a los datos.
Se grafican los datos y el polinomio que mejor se ajusta a los datos.
Se guardan los valores predichos en un archivo .csv y las graficas en un archivo .png para cada parcela.

Se mejoró con el archivo [find_degree_op.py](), el cual es una versión optimizada
y solo necesita la zafra y si se desea exportar los metadatos.