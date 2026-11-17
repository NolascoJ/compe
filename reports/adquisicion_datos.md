# Adquisición de datos (consigna 3.4)

Qué datos incorporaríamos y por qué esperamos que mejoren el sistema. La propuesta se apoya en dos evidencias:
- la [curva de aprendizaje](curva_aprendizaje.md) (3.2);
- dónde se concentran los errores fuera de fold del HistGB elegido ([evidencia.py](../experiments/adquisicion_datos/evidencia.py), [evidencia_errores.csv](../experiments/adquisicion_datos/evidencia_errores.csv)).

Todo sale de TRAIN, con el corte interno de cada fold y sin GPU asignada; `test.csv` no se usó. Los segmentos son descriptivos y se superponen: muestran dónde falta información, no prueban qué dato la resolvería.

## Punto de partida

- **Más filas del mismo tipo rinden poco.** HistGB gana alrededor de +1 punto de BA cada vez que se duplican las filas. Reentrenar con train+test daría del orden de +0,5 puntos, por extrapolación. La logística ya está estancada. El límite principal es la **información** de las variables, no la cantidad de filas.
- **Muchos errores no son dudosos.** El 28% de los errores (1.612 de 5.769) tiene un score a más de 0,25 del corte: el modelo se equivoca con confianza. Eso apunta a información que falta, no a ruido cerca del corte.

## Dónde falla el modelo (OOF de TRAIN, BA global 84,83%)

| Segmento | Filas | BA | Recall | Parte de los falsos negativos | Parte del error total |
|---|---:|---:|---:|---:|---:|
| Sin historial previo del cliente | 91,2% | 83,3% | 81,6% | 98,2% | 99,0% |
| Con historial previo | 8,8% | **98,4%** | 98,2% | 1,8% | 1,0% |
| Anticipación de 0 a 7 | 13,6% | **61,4%** | **25,3%** | 20,6% | 10,9% |
| Anticipación de 8 a 30 | 15,7% | 78,1% | 70,2% | 21,3% | 19,5% |
| Con pedidos especiales | 39,9% | **70,5%** | **51,8%** | 63,8% | 50,3% |
| Regiones distintas de PRT | 61,4% | 80,9% | 74,9% | 63,3% | 66,8% |
| `no_show` (cancelados de hecho) | 1,1% | — | **44,6%** | 9,4% | 4,1% |
| Sin `locked_allocation` | 83,9% | 79,9% | 74,4% | 100% | 99,9% |

«Sin historial previo» significa `returning_client = 0`, sin cancelaciones previas y sin trabajos previos exitosos. La fila de regiones suma las frecuentes y las raras.

## Datos que incorporaríamos, por prioridad

### 1. Historial del cliente con un identificador estable
**Qué:** un ID de cliente que permita unir sus reservas. Con él se calculan cancelaciones y finalizaciones previas reales, la tasa histórica de cancelación, el tiempo desde la última reserva y las reservas activas al mismo tiempo.

**Por qué:** es el mayor salto observado. Cuando hay historial la BA es 98,4%; el 91% de las reservas no lo tiene y concentra el 99% de los errores. Probablemente muchos de esos clientes no son nuevos: simplemente sus reservas no están vinculadas, como ocurre con los huéspedes en el dataset de hoteles del que parece venir este. También ayuda frente al cambio de distribución, porque describe al cliente y no a su región.

**Cuidado:** el historial tiene que calcularse solo con reservas anteriores a la que se predice, y como estaba al momento de predecir. Si no, hay fuga.

### 2. Eventos de la reserva con fecha
**Qué:** la fecha de creación y la de cada modificación, qué cambió (fecha, recursos, cantidad de horas) y los días desde el último cambio. Hoy solo existe el conteo `resource_modifications`.

**Por qué:** el modelo casi no detecta las cancelaciones de último momento. Con anticipación de 0 a 7 detecta el 25% de las cancelaciones (BA 61,4%), y entre 0 y 30 se acumula el 42% de los falsos negativos. Una reserva cancelada pronto suele mostrar actividad reciente que hoy no se ve.

### 3. Señales de preparación cerca de la ejecución
**Qué:** si se confirmó la reserva, si se cargaron los datos o la imagen del contenedor, si el pago o el depósito se acreditó de verdad, y la actividad del cliente en la plataforma en las 24 a 72 horas previas.

**Por qué:** los `no_show` se parecen a los completados en todas las variables actuales. Se detecta el 44,6% y explican el 9,4% de los falsos negativos. Su diferencia está en el comportamiento cerca de la ejecución, y eso es lo que miden estas señales. También explicaría parte de los errores con confianza.

**Cuidado:** hay que definir el momento en que se predice, por ejemplo 72 horas antes, y usar solo lo que se conoce en ese momento. Es la misma lección de `assigned_gpu_type`.

### 4. Detalle de los pedidos especiales
**Qué:** el tipo de cada pedido, no solo la cantidad, y si fue atendido o rechazado.

**Por qué:** con pedidos el recall cae al 51,8% (BA 70,5%) y ahí está el 64% de los falsos negativos. Hoy el pedido funciona como señal de compromiso, pero el modelo no distingue a quien pidió algo y después canceló. Un pedido rechazado puede explicar justamente esa cancelación.

### 5. Más reservas de regiones fuera de PRT, y su fecha de registro
**Qué:** más filas de clientes de otras regiones y la fecha en que se registró cada reserva, para poder validar en orden temporal.

**Por qué:** esas regiones son el 61% de las filas y explican el 67% del error. Entrenar solo con PRT y evaluar en el resto da alrededor de 51%: es la principal debilidad ante un test distinto. La curva de aprendizaje indica que sumar filas sin elegirlas rinde poco. Si se suman, conviene que sean de esta cohorte.

### Prioridad baja
- **Hora de la reasignación de GPU.** Permitiría usar `assigned_gpu_type` sin fuga cuando la reasignación ocurre antes del momento de predecir. Hoy aporta solo unos 0,3 puntos.
- **Política de cancelación detallada** (penalidad, plazos) y cambios de precio desde que se reservó. Hoy el costo tiene poca señal propia y `locked_allocation` ya se predice casi perfecto.
- **Más filas al azar del mismo tipo.** Según 3.2, rinden alrededor de +1 punto por cada duplicación.

## Cómo validaríamos cada fuente nueva

Se usa el mismo protocolo: folds agrupados, corte elegido con folds internos, y resultados por subgrupos y con validación dejando afuera un año o una región. Una fuente entra solo si:
1. mejora la BA más que el desvío entre folds;
2. no empeora a las cohortes débiles (sin historial, anticipación corta, fuera de PRT);
3. está disponible en el momento de predecir.

## Para la diapositiva

**Adquisición de datos: más información, no más filas**
- La curva de aprendizaje muestra unos +1 punto por cada duplicación de filas. El límite es la información de las variables.
- **1. Historial del cliente** con un ID estable: con historial la BA es 98%, y sin historial están el 91% de las reservas y el 99% de los errores.
- **2. Eventos con fecha y señales cercanas a la ejecución:** con menos de una semana de anticipación se detecta el 25% de las cancelaciones, y los `no_show` se detectan en un 45%.
- **3. Detalle de pedidos especiales y más datos fuera de PRT:** las cohortes con más error (BA de 70,5% y de 80,9%).
- Toda variable nueva tiene que existir al momento de predecir y se acepta solo si mejora la CV más que la variación entre folds.
