# TP Análisis de Datos

El objetivo es preparar y analizar los datos del UCDP Georeferenced Event Dataset (GED) con el fin de construir un conjunto de datos adecuado para el entrenamiento de un modelo de aprendizaje supervisado de clasificación, cuyo propósito será predecir el nivel de letalidad de los eventos de conflicto armado.

Cada registro del dataset representa un evento individual de violencia, por lo que la clasificación se realizará a nivel de evento y no de conflicto.

Para ello, se unificaron y depuraron las estimaciones de muertes (low, best, high) generando la variable best_est_cleaned, una medida continua, coherente y representativa de la cantidad estimada de víctimas por evento.
A partir de esta variable, se definieron tres rangos de letalidad que categorizan los eventos como de baja, media o alta letalidad.

De este modo, el modelo tendrá un enfoque clasificador, y no de regresión, utilizando la variable lethality_class (derivada de best_est_cleaned) como output o variable objetivo.
El objetivo final es dejar el dataset completamente preparado para el entrenamiento posterior de dicho modelo.

Los pasos realizados:
- Exploración y comprensión de los datos
- Aplicación de técnicas de visualización (pevio a la limpieza)
- Plantear un posible problema de ML supervisado a partir de los datos elegidos (Arriba)
- Creacion de la variable de tipo de violencia

Los pasos a seguir
- Realizar la limpieza del dataset:
* Identificar posible candidatos de variables para uso en el modelo y cuales no
* Determinar el tratamiento de los nulls de las vaiables candidatas
* Determinar el tratamiento de los outliers de las variables candidatas (usar tecnicas de estadisticas, rango intercuartil, simetria, usar curtosis para demostrar el acomodo)

* Division en train y test

Feature Engineering:
* Creacion de la variable lethality_class (alta media baja) y de best_est_cleaned (ya creada en outliers)
----------
### VER MAS ADELANTE
* escalar y/o normalizar las variables
* Analizar balance/desbalance
* Reduccion de dimensionalidad

## Algunas notas
El rango de eventos de 1989 a 2024

Número de observaciones: 385918
Número de variables: 49

Grafico tablas:
- MAPA
- Mortalidad x anio
- Tipos de conflictos
- tabla resumida de tipos de variable (poner cantidad y tipo de variables en numeros y porcentajes)

10 columnas contienen Nulls, de la cual gwnob tiene mas del 90% y las demas menos de 30%

No hay duplicadas,

### Hallazgos:

- Poner como fue incrementando el conflicto a lo largo de los anios y ver su relacion con los eventos de conocimiento comun como el ataque de rusia a ucrania y el ataque de palestina a israel
- Todas las columnas de source no contienen datos hasta el anio 2013, apartir de el comiezan a tener datos

*Doc:* somos grupo 3
- https://docs.google.com/presentation/d/1K3IO084mpPHqmIpnmvmjrlrZdIqV2WnC0Ju3aL-GZPQ/edit?slide=id.g385f7747de7_0_76#slide=id.g385f7747de7_0_76
- https://docs.google.com/spreadsheets/d/1y-voBJHSrUeRk7HqJno7JaQxiYq1KN6-KybRlkNlBmU/edit?gid=1867552948#gid=1867552948

*Tema:* Eventos de violencia organizada (UCDP GED)

## Variables
|    | variable_name     | description                                                                                     | type Pandas   | eliminate   | comment                                                            |
|---:|:------------------|:------------------------------------------------------------------------------------------------|:--------------|:------------|:-------------------------------------------------------------------|
|  0 | id                | Identificador unico del evento                                                                  | int64         | True        | Valor único por observación                                        |
|  1 | relid             | Identificador interno del evento (cambia si el ano o los participantes cambian)                 | object        | True        | Valor único por observación                                        |
|  2 | year              | Ano del evento                                                                                  | int64         | False       | nan                                                                |
|  3 | active_year       | True si el evento pertenece a un conflicto activo con más de 25 muertes                         | bool          | False       | nan                                                                |
|  4 | code_status       | Si el registro est  ok (clear) o si hay que chequear algo                                       | object        | True        | Valor único (solo contiene 'clear')                                |
|  5 | type_of_violence  | Tipo de violencia: 1. Conflicto entre estados, 2. Conflicto no estatal, 3. Conflicto unilateral | int64         | False       | nan                                                                |
|  6 | conflict_dset_id  | Identificador deprecado                                                                         | int64         | True        | V=1 con [side_a]                                                   |
|  7 | conflict_new_id   | Identificador único del conflicto (usar esto para agregar)                                      | int64         | True        | V=1 con [side_a]                                                   |
|  8 | conflict_name     | Nombre del conflicto                                                                            | object        | True        | V=1 con [side_a]                                                   |
|  9 | dyad_dset_id      | Identificador deprecado                                                                         | int64         | True        | V=1 con [side_a]                                                   |
| 10 | dyad_new_id       | Identificador unico de diadas participantes (usar esto para agregar)                            | int64         | True        | V=1 con [side_a]                                                   |
| 11 | dyad_name         | Nombre de los participantes de la diada en conflicto                                            | object        | False       | V=1 con [side_a]                                                   |
| 12 | side_a_dset_id    | Identificador del participante A cuando la info no es certera                                   | int64         | True        | V=1 con [side_a]                                                   |
| 13 | side_a_new_id     | Identificador unico del participante A                                                          | int64         | True        | V=1 con [side_a]                                                   |
| 14 | side_a            | Nombre del participante A en la diada                                                           | object        | True        | nan                                                                |
| 15 | side_b_dset_id    | Identificador del participante B cuando la info no es certera                                   | int64         | True        | V=1 con [side_b]                                                   |
| 16 | side_b_new_id     | Identificador unico del participante B                                                          | int64         | True        | V=1 con [side_b]                                                   |
| 17 | side_b            | Nombre del participante B en la diada                                                           | object        | True        | V=1 con [side_b]                                                   |
| 18 | number_of_sources | Cantidad de fuentes incorporadas con informacion del evento                                     | int64         | False       | nan                                                                |
| 19 | source_article    | Nombres, fechas y titulos de las fuentes empleadas                                              | object        | True        | Datos de la fuentes                                                |
| 20 | source_office     | Nombre de la organizacion que publica las fuentes empleadas                                     | object        | True        | Datos de la fuentes                                                |
| 21 | source_date       | Fecha de publicacion de la fuente empleada (de cada una de las fuentes)                         | object        | True        | Datos de la fuentes                                                |
| 22 | source_headline   | Titulo de la fuente empleada                                                                    | object        | True        | Datos de la fuentes                                                |
| 23 | source_original   | Persona u organizacion que dio informacion en la fuente empleada                                | object        | True        | Datos de la fuentes                                                |
| 24 | where_prec        | Precision de las coordenadas y ubicacion asignada al evento (1. mayor - 7. menor)               | int64         | False       | Parece no agregar, a evaluar                                       |
| 25 | where_coordinates | Nombre normalizado de la ubicacion del evento                                                   | object        | False       | nan                                                                |
| 26 | where_description | Comentario (detalle) sobre where_coordinates (puede estar vacio)                                | object        | True        | Detalle de alta cardinalidad (224241) de [where_coordinates]       |
| 27 | adm_1             | Nombre de la division administrativa donde ocurre el evento (nivel 1)                           | object        | False       | nan                                                                |
| 28 | adm_2             | Nombre de la division administrativa donde ocurre el evento (nivel 2)                           | object        | True        | V=0.81 con [adm_1] y más nulos                                     |
| 29 | latitude          | Latitud en grados decimales                                                                     | float64       | False       | nan                                                                |
| 30 | longitude         | Longitud en grados decimales                                                                    | float64       | False       | nan                                                                |
| 31 | geom_wkt          | Representacion en texto de la ubicacion del evento                                              | object        | True        | Coincide con la concatenación de [longitude] y [latitude] (358384) |
| 32 | priogrid_gid      | Asignacion de la ubicacion del evento a la grilla PRIO                                          | int64         | False       | Sirve para enriquecer el dataset con info extra                    |
| 33 | country           | Nombre del pais donde ocurre el evento                                                          | object        | False       | nan                                                                |
| 34 | country_id        | Identificador del pais donde ocurre el evento                                                   | int64         | True        | V=1 con [country]                                                  |
| 35 | region            | Region (Africa, Americas, Asia, Europe, Middle East) donde ocurre el evento                     | object        | False       | Alta correlación con [country] y [dyad_name]                       |
| 36 | event_clarity     | 1: High, 2: Low dependiendo de lo desagregado de la informacion en la fuente                    | int64         | False       | nan                                                                |
| 37 | date_prec         | Precision de la fecha del evento (1. mayor - 5. menor)                                          | int64         | False       | nan                                                                |
| 38 | date_start        | Fecha probable del inicio del evento                                                            | object        | False       | nan                                                                |
| 39 | date_end          | Fecha probable del fin del evento                                                               | object        | False       | nan                                                                |
| 40 | deaths_a          | Muertes del lado A (0 si el evento es unilateral)                                               | int64         | False       | [deaths_a]+[deaths_b]+[deaths_civilians]+[deaths_unknown]=[best]   |
| 41 | deaths_b          | Muertes del lado B (0 si el evento es unilateral)                                               | int64         | False       | [deaths_a]+[deaths_b]+[deaths_civilians]+[deaths_unknown]=[best]   |
| 42 | deaths_civilians  | Muertes civiles                                                                                 | int64         | False       | [deaths_a]+[deaths_b]+[deaths_civilians]+[deaths_unknown]=[best]   |
| 43 | deaths_unknown    | Muertes con estatus desconocido                                                                 | int64         | False       | [deaths_a]+[deaths_b]+[deaths_civilians]+[deaths_unknown]=[best]   |
| 44 | best              | Mejor estimacion (mas probable) de muertes totales                                              | int64         | False       | nan                                                                |
| 45 | high              | Maxima estimacion de muertes totales                                                            | int64         | True        | r=0.83 con [best] La sacaría proque es una versión de [best]       |
| 46 | low               | Minima estimacion de muertes totales                                                            | int64         | True        | r=0.93 con [best] La sacaría porque es una versión de [best]       |
| 47 | gwnoa             | Identificador alfabetico GW del participante A (si es un gobierno)                              | object        | True        | V=1 con [side_a]                                                   |
| 48 | gwnob             | Identificador alfabetico GW del participante B (si es un gobierno)                              | float64       | True        | V=1 con [side_b]                                                   |
