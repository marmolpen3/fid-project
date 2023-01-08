# Proyecto asignatura FID
El presente documento es el informe del resultado del trabajo del grupo 7, formado por:

- Mario Ruano Fernández
- Juan Carlos Cortés Muñoz
- Alejandro José Muñoz Aranda
- María Elena Molino Peña

Previo al inicio de la documentación puntualizar que se ha hecho uso de dos datasets, uno de ellos para el análisis 
[básico]( https://www.kaggle.com/datasets/aguado/bike-rental-data-set-uci) y otro para el [avanzado](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data). 
La justificación de la toma de esta decisión se detallará más adelante y pude encontrarse en el notebook de análisis avanzado.

El documento se organizará de la siguiente manera, en primer lugar se definirán los dos problemas elegidos que se pretenden resolver aplicando Machine learning, 
en la sección 2 hablaremos sobre el proceso de análisis y los algoritmos seleccionados para cada caso, en cuanto a la sección 3 se describen todas las fases que se ha aplicado a los datos,
y por último, se recogen las asignaciones y definiciones del trabajo realizado por cada subequipo.

## Sección 1. Descripción del problema planteado
Los problemas que se plantean resolver se han seleccionado de la herramienta [Kaggle](https://www.kaggle.com/), donde se pueden encontrar multitud de retos disponibles 
y dataset adecuados para aplicar diferentes técnicas de Machine learning.
 
### Bike Rental Data

Los sistemas de alquiler de bicicletas suelen recopilar información interesante sobre la trazabilidad de cada arrendamiento. 
El principal objetivo es mejorar la gestión y anticipar la demanda de bicicletas que habrá en un determinado rango de tiempo. 
Por lo tanto, la tarea que se propone es predecir la demanda de bicicletas en una serie de franjas horarias, 
empleando el conjunto de [datos](https://www.kaggle.com/datasets/aguado/bike-rental-data-set-uci) que contiene las siguientes variables:

- id: identificador (variable cuantitativa discreta)
- year: años (2011 y 2012) (variable cuantitativa discreta)
- hour: hora del día (0 to 23) (variable cuantitativa discreta)
- season: 1 = invierno, 2 = primavera, 3 = verano, 4 = otoño (variable cuantitativa discreta)
- holiday: si fue un día de vacaciones (variable cuantitativa discreta)
- workingday: si fue un día de trabajo (variable cuantitativa discreta)
- weather: tres categorías en rango de mejor a peor tiempo (1 a 3) (variable cuantitativa discreta)
- temp: temperatura en grados Celsius (variable cuantitativa continua)
- atemp: sensación de temperatura en grados Celsius (variable cuantitativa continua)
- humidity: humedad relativa (variable cuantitativa continua)
- windspeed: velocidad del viento (km/h) (variable cuantitativa continua)
- num_bikes: número total de bicicletas alquiladas en para esas condiciones (variable cuantitativa discreta)

Para el desarrollo de esta parte del proyecto se propone estudiar el conjunto de datos, la viabilidad de aplicar algoritmos de regresión lineal 
y los algoritmos más eficientes para resolver el problema que se encuentren en el paquete caret. Además, se analizará si es posible la 
aplicación de técnicas de Clustering.

### Country Data
Algunas de las Organizaciones No Gubernamentales suelen recaudar dinero para ayudar a personas o países que se encuentran en situación de necesidad. 
En este caso, la organización HELP International desea identificar los países más precarios para emplear de forma estratégica y eficaz el dinero recaudado. 
Por lo tanto, la tarea que se propone es agrupar los países según sus valores sociales, económicos y de salud, empleando el conjunto de [datos](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data)
que dispone de los siguientes valores:

- contry: nombre del país (variable cualitativa)
- child_mort: muerte de niños menores de 5 años por cada 1000 nacimientos
- exports: exportaciones de bienes y servicios per cápita. En porcentaje del PIB per cápita
- health: gasto sanitario total per cápita. En porcentaje del PIB per cápita
- imports: importaciones de bienes y servicios per cápita. En porcentaje del PIB per cápita
- income: renta neta per cápita
- inflation: medida de la tasa de crecimiento anual del PIB total
- life_expec: número medio de años que viviría un recién nacido si se mantuvieran las actuales pautas de mortalidad
- total_fer: número de hijos que nacerían de cada mujer si se mantienen las actuales tasas de fecundidad por edad
- gdpp: PIB per cápita. Calculado como el PIB total dividido por la población total.

Para el desarrollo de esta parte del proyecto se propone estudiar el conjunto de los datos y aplicar técnicas de Clustering 
para detectar los países más necesitados según sus características.

## Sección 2. Descripción del proceso de análisis
La selección del conjunto de datos y la definición del problema a resolver es esencial para definir el proceso de análisis que se va a llevar a cabo.

### Análisis básico. Regresión
En esta primera fase del proyecto se aplican algoritmos Supervisados para resolver el problema de las bicicletas anteriormente propuesto. Tomando como conjunto de datos el dataset "Bike Rental Data" y aplicando modelos de regresión se pretende predecir el número de bicicletas que se alquilarán en el futuro según las codiciones temporales.

Con el proposito de resolver este problema se plantean una serie de etapas a seguir y métodos a aplicar. Las etapas y objetivos que plantea el grupo son los siguientes:
- Análisis y preprocesado de los datos: renombrar y eliminar atributos, búsqueda de valores perdidos, detección de outliers, estudio de las variables.
- Análisis y Visualización de los datos: comprobar la homogeneidad de los datos, histogramas, estudio de los datos.
- Regresión lineal: dispersión de los ejemplos frente a la variable a predecir, estudio de la correlación, construcción y estudio del modelo, predicción.
- Algoritmos paquete Caret: creación modelos Random Forest, xgbTree y gbm.
- Comparativa: obtener el modelo con mejores resultados.

### Análisis avanzado. Clustering
El segundo análisis del proyecto estará enfocado al estudio y aplicación de algoritmos No Supervisados con el fin de resolver el problema de los países anteriormente identificado. Empleando como conjunto de datos el dataset "Country Data" y aplicando modelos de Clustering como K-means y jerárquicos se desea ayudar a la ONG HELP International a seleccionar los países con más necesidades.

Con el propósito de resolver este problema se plantean una serie de etapas a seguir y métodos a aplicar. Las etapas y objetivos que plantea el grupo son los siguientes:
- Viabilidad de aplicar clustering al dataset "Bike Rental Data": estudiar los resultados obtenidos tras aplicar Clustering.
- Análisis y preprocesado de los datos: renombrar y eliminar atributos, búsqueda de valores perdidos, detección de outliers, estudio de las variables.
- Clustering: escalar valores de las variables, aplicar algoritmo Particional K-means y Clustering Jerárquico.

## Sección 3. Descomposición de etapas

### Análisis básico. Regresión
#### Preprocesado de los datos
#### Análisis y visualización
#### Viabilidad sobre la regresión lineal
#### Algoritmos paquete Caret

### Análisis avanzado. Clustering
#### Preprocesado de los datos
#### Algoritmo Particional. K-means
#### Clustering Jerárquico


## Sección 4. Subdivisión y descripción del trabajo por cada subequipo

El proyecto se ha dividido de forma equitativa por todos los miembros del grupo. Ambos subequipos han participado en el estudio, desarrollo o búsqueda de información del análisis básico y avanzado. A pesar de ello, cada notebook ha sido auditado por un subequipo concreto que se ha encargado de revisar y coordinar la ejecución del mismo. En concreto:
- Mario Ruano Fernández y Juan Carlos Cortés Muñoz: dirección del notebook de análisis básico, estudio de los problemas seleccionados, búsqueda de información sobre regresión y clustering, aplicación de los algoritmos supervisados.
- Alejandro José Muñoz Aranda y María Elena Molino Peña: dirección del notebook de análisis avanzado, estudio de los problemas seleccionados, búsqueda de información sobre regresión y clustering, aplicación de los algoritmos de clustering.
