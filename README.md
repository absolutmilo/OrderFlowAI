# OrderFlowAI ⚡
### Motor de Inferencia de Flujo de Órdenes & Sonificación en Tiempo Real

**OrderFlowAI** es una plataforma de análisis cuantitativo de microestructura de mercado y sonificación acústica para Bitcoin (`BTC/USDT`), diseñada para detectar en tiempo real:
1. **Ejecución Algorítmica Institucional** (TWAP / VWAP / Slicing).
2. **Iceberg Orders** (Comprador o Vendedor con recarga pasiva oculta).
3. **Market Maker Absorbiendo** (Presión agresiva neutralizada sin desplazamiento de precio).
4. **Trading Minorista Dominante** (Órdenes pequeñas fragmentadas).
5. **Alta Competencia HFT** (Ráfagas violentas de microsegundos en ambos lados de la cinta).

---

## 🏗 Arquitectura del Sistema

```mermaid
flowchart TD
    WS["WebSockets Binance Spot + Bybit Linear (BTCUSDT)"] --> Agg["TradeAggregator (ventana de hasta 50ms)"]
    Agg --> Tape["Cinta de Trades Agregados"]
    Agg --> Class["MicrostructureClassifier (Rolling Window)"]
    Class --> Metrics["Métricas (CVD, TPS, Volatilidad, Entropía Temporal)"]
    Class --> Probs["Clasificador heurístico (puntuaciones 0-100%)"]
    Probs --> Srv["FastAPI + WebSocket Broadcaster"]
    Tape --> Srv
    Srv --> UI["Dashboard Web (HTML5/CSS3)"]
    Srv --> Audio["Web Audio API (Sonificación Espacial)"]
```

---

## 🎯 Firmas de Microestructura Detectadas

| Patrón | Métrica Cuantitativa | Comportamiento en la Cinta |
| :--- | :--- | :--- |
| **Ejecución algorítmica (heurística)** | Regularidad temporal (60 %) y uniformidad del tamaño en USD (40 %) | Cadencia y tamaños regulares; no identifica la estrategia ni al operador. |
| **Iceberg posible** | Más del 22 % del volumen en un precio redondeado a 0.1, con al menos 6 grupos | Concentración ejecutada en un nivel; sin libro de órdenes no confirma liquidez oculta. |
| **Absorción posible** | Al menos $200k$ de volumen, ratio $|CVD| / \text{Volumen} > 35\%$ y desplazamiento inferior a 1.5 bps | Desequilibrio agresivo con poco desplazamiento del precio. |
| **Actividad rápida / alternancia** | Más de 60 grupos por segundo; puntaje combina TPS y alternancia | Ráfagas en la cinta, no identificación de HFT. |
| **Breakout** | Desplazamiento superior a 6 bps y desequilibrio de volumen superior a 40 % | Movimiento acompañado por presión direccional. |
| **Puntuación retail** | Inversa a la mayor puntuación de algoritmo, iceberg, absorción o HFT | No clasifica directamente tamaños ni identidades minoristas. |

> Las puntuaciones actuales son heurísticas, no probabilidades estadísticas calibradas. Los datos públicos de trades no incluyen el libro de órdenes y no permiten confirmar directamente órdenes iceberg, identidad institucional ni actividad HFT. El panel de información del dashboard describe las reglas y sus limitaciones.

## 🧪 Captura, señales y backtesting de scalping

El modo inicial se define en `CONFIG.classifier.default_inference_mode` en `config.py` (actualmente `model`). En modo entrenado, el servidor carga `data/hmm_regime_model.json`; si no puede cargar el artefacto, inicia en modo heurístico e indica el modo efectivo en el dashboard. Usa el botón **MODO** del panel para alternar entre el HMM entrenado y las reglas heurísticas; la página confirma el cambio o muestra el error. Al volver al modo HMM, el estado de inferencia se reinicia.

Las cinco tarjetas de clasificación muestran el valor actual y una serie temporal de la última hora, consultada de SQLite y refrescada con el feed en intervalos de 5 segundos. La serie sigue el modo de inferencia seleccionado; si no existen snapshots guardados para ese modo, se completará a medida que lleguen datos en vivo. La cinta muestra las transacciones en vivo y, en una segunda columna, una cola de las operaciones ≥ $100,000. Esa cola vive solo mientras la página está abierta, conserva hasta 120 elementos y puede limpiarse desde el panel.

Después del calentamiento, un muestreador por reloj guarda snapshots cada segundo con las métricas/features, puntuaciones heurísticas y configuración versionada, sin depender de que llegue un trade justo en ese instante. Solo captura mientras Bybit Linear haya actualizado el feed en los últimos 5 segundos; durante una desconexión no duplica datos obsoletos. `/api/status` informa el tiempo desde el último trade y snapshot; `/api/scalping/status` informa también la cobertura de las últimas 24 horas, que la página consulta al cargar y después de ejecutar un backtest. El porcentaje de la última validación es un resultado histórico del backtest y puede diferir de esa cobertura reciente. Las observaciones de scalping se capturan exclusivamente de Bybit Linear; Binance Spot sigue participando en el clasificador general, pero no se mezcla en las entradas/performance del backtest. Los registros anteriores se migran como `legacy_combined` y se excluyen del backtest nuevo. La base está en `data/orderflow.sqlite3` y no tiene borrado automático.

Cada snapshot nuevo de Bybit conserva por separado las puntuaciones heurísticas y el posterior del HMM-4 (`hmm_4`). Los registros anteriores no contienen ese posterior: no deben interpretarse como historial HMM. El estado HMM se reinicia después de una interrupción de captura mayor a 5 segundos, para no asumir continuidad durante una caída del feed. El HMM clasifica cuatro regímenes; no detecta armónicos. El análisis Fourier es un componente separado.

El panel **Setups de scalping BTCUSDT · Bybit Linear** permite seleccionar hipótesis y hacer backtesting paper, sin envío de órdenes:

1. Para **Validar order flow (SQLite)**, deja correr la captura de Bybit hasta reunir al menos 51,429 snapshots (aproximadamente 14.3 horas): así el primer 70 % contiene 36,000 observaciones para entrenamiento y queda un 30 % posterior para validación. Fourier puede requerir más velas completas según la temporalidad seleccionada.
2. Selecciona la estrategia, temporalidad de 1, 5, 10 o 15 minutos y dirección original/invertida, y pulsa **Validar order flow (SQLite)**. Esta ruta mantiene las estrategias microestructurales existentes. La variante invertida conserva los mismos timestamps de entrada y voltea LONG↔SHORT antes de volver a simular las salidas; también vuelve a seleccionar SL/TP solo en entrenamiento. `Flujo + momentum` es la línea base; `Fourier: anomalía + ruptura` busca residuo anómalo, contexto de transición y ruptura posterior; `Volumen excepcional + tendencia` requiere volumen snapshot elevado, EMA rápida/lenta alineada y desequilibrio del flujo en la misma dirección.
3. Como alternativa independiente, pulsa **Validar velas OHLCV directo** para usar hasta 30 días de velas públicas de Bybit Linear, sin leer ni modificar SQLite. El histórico se guarda en `data/bybit_klines_cache/`, separado por símbolo y temporalidad fuente (el timeframe de 10 minutos agrega velas de 5 minutos). Al repetir una consulta completa se reutiliza el archivo y solo se descargan velas recientes que falten. Errores transitorios de Bybit como el código 10016 se reintentan con esperas crecientes; las páginas recibidas se guardan mientras continúa la descarga para poder reanudarla al volver a intentarlo. Este modo prueba una hipótesis OHLCV propia (EMA 8/21, momentum de tres velas y confirmación de volumen); no reconstruye CVD ni valida las estrategias de order flow. La entrada se simula en la apertura de la vela posterior a la señal. El sistema optimiza múltiplos de stop/objetivo ATR solo en el 70 % inicial y valida en el 30 % cronológico final; si stop y objetivo aparecen tocados en la misma vela, cuenta primero el stop. Comisión taker y slippage por lado se descuentan en ambos modos. El resultado indica si reutilizó, actualizó, reanudó o descargó el histórico.
4. El modo OHLCV es de investigación/paper únicamente: nunca habilita alertas live. Si Bybit no devuelve velas suficientes o la serie tiene huecos, el resultado se informa como insuficiente en vez de rellenar los huecos. Este modo no borra, sustituye ni agrega datos a `data/orderflow.sqlite3`; esa base sigue disponible para snapshots, etiquetas y validación/tuning de estrategias microestructurales.
5. El Fourier de la ruta SQLite es una adaptación de corto plazo (ciclos de 8 y 24 velas, ventana de 48 y mínimo de 72 velas de entrenamiento); no replica literalmente las entradas en días ni todos los filtros del script Pine. Requiere dos anomalías residuales consecutivas con |z| > 2.7, zona cerrada, contexto válido y ruptura; puede no generar entradas aunque el número mínimo de velas se haya alcanzado. El panel reporta anomalías, zonas, rupturas y operaciones simuladas para distinguir falta de señales de falta de historial. El volumen de esa ruta es una tasa proxy derivada de una ventana móvil de trades. No se incorpora aún SP500/VIX.
6. Solo la estrategia base original de la ruta SQLite puede activar señales live; direcciones invertidas, Fourier, volumen+tendencia y el backtest OHLCV descargado se mantienen como investigación, no activan señales live ni órdenes. Compara original e invertida usando el mismo corte temporal; no escojas repetidamente por el mismo conjunto de validación porque se convierte en ajuste a la muestra. Resultados pasados no garantizan rendimiento futuro.
7. El backtest SQLite exige al menos 80 % de cobertura dentro de los tramos activos (las pausas de más de 5 segundos se tratan como interrupciones) y descarta velas u operaciones que atraviesen esos huecos. Si la captura pierde continuidad dentro de los tramos, el panel lo informa en vez de presentar estadísticas como si el mercado se hubiera observado sin interrupciones. Reúne primero historial continuo con HMM-4 antes de evaluar persistencia de regímenes o SL/TP condicionados por régimen.

Esta primera versión es experimental, no una recomendación ni garantía de rentabilidad. La ruta SQLite usa snapshots de precio de 1 segundo; la ruta OHLCV opera sobre velas descargadas. Ninguna incorpora funding, latencia, cola de ejecución ni impacto real de mercado, y una muestra corta puede sobreajustarse. No se conectan endpoints privados ni se colocan órdenes.

Para reentrenar el HMM de clasificación de regímenes con la base:

```bash
python -m src.ml.train_hmm
```

El entrenamiento requiere al menos 100 observaciones guardadas. Al terminar, reinicia el servidor para cargar el artefacto actualizado.

Para cada snapshot Bybit Linear, se registran retornos futuros a 5, 30 y 60 segundos y a 5, 10 y 15 minutos en puntos básicos. Se etiquetan como reversión, continuación o neutral respecto a la dirección del CVD inicial; movimientos inferiores a 1 bp son neutrales. Son etiquetas de resultado derivadas del precio, no ground truth de intención del mercado.

---

## 🎧 Sonificación Acústica Procedural (Web Audio API)

El motor acústico integrado en el navegador permite "escuchar la cinta" (*tape sonification*) en tiempo real:
* **Canal Derecho (Stereo Pan +0.35):** Compras agresivas (*Market Buys* que golpean el Ask) en frecuencias armónicas agudas (520Hz – 880Hz).
* **Canal Izquierdo (Stereo Pan -0.35):** Ventas agresivas (*Market Sells* que golpean el Bid) en frecuencias graves (220Hz – 380Hz).
* **Whale Orders ($\ge \$100\text{k}$):** Golpe resonante de sub-bajo (45Hz – 110Hz).
* **Icebergs Detectados:** Doble tono campaniforme en intervalo armónico.
* **Absorción:** Acorde resonante con filtro paso-bajo que simula impacto contra un muro de liquidez.

---

## 🚀 Instalación y Puesta en Marcha

### 1. Requisitos Previos
* Python 3.9 o superior.

### 2. Configurar Entorno Virtual e Instalar Dependencias
Desde una terminal (PowerShell o CMD):

```bash
cd "C:\Users\PERSONAL\Documents\DEV\OrderFlowAI"

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Windows PowerShell:
.\venv\Scripts\Activate.ps1
# En Windows CMD:
.\venv\Scripts\activate.bat

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Iniciar el Servidor y Dashboard
```bash
python run.py
```
El navegador se abrirá automáticamente en:
👉 **`http://localhost:8000`**

Haz clic en el botón superior:
**🔊 "Activar Sonificación en Tiempo Real"** para habilitar el audio procedural en el navegador.

---

## 📂 Estructura del Código

* [`config.py`](file:///C:/Users/PERSONAL/Documents/DEV/OrderFlowAI/config.py): Configuración de pares, WebSocket URIs y umbrales matemáticos.
* [`src/core/trade_model.py`](file:///C:/Users/PERSONAL/Documents/DEV/OrderFlowAI/src/core/trade_model.py): Modelos de datos para trades, agregación y probabilidades.
* [`src/core/aggregator.py`](file:///C:/Users/PERSONAL/Documents/DEV/OrderFlowAI/src/core/aggregator.py): Agrupador sub-milisegundo para emular la cinta de SignificantTrades.
* [`src/core/classifier.py`](file:///C:/Users/PERSONAL/Documents/DEV/OrderFlowAI/src/core/classifier.py): Motor heurístico de puntuación de microestructura.
* [`src/core/market_data_store.py`](file:///C:/Users/PERSONAL/Documents/DEV/OrderFlowAI/src/core/market_data_store.py): Captura SQLite de features, configuración y resultados futuros.
* [`src/connectors/binance.py`](file:///C:/Users/PERSONAL/Documents/DEV/OrderFlowAI/src/connectors/binance.py): Conector asíncrono WebSocket a Binance Spot/Futures.
* [`src/api/server.py`](file:///C:/Users/PERSONAL/Documents/DEV/OrderFlowAI/src/api/server.py): Servidor FastAPI y WebSocket broadcaster.
* [`static/`](file:///C:/Users/PERSONAL/Documents/DEV/OrderFlowAI/static):
  * `index.html`: Dashboard con medidores de probabilidad y cinta de transacciones.
  * `js/audio.js`: Sintetizador Web Audio API espacial.
  * `js/app.js`: Cliente WebSocket y renderizado a 60 FPS.
  * `css/style.css`: Estilo oscuro Fintech / Trading Terminal.
* [`tests/test_classifier.py`](file:///C:/Users/PERSONAL/Documents/DEV/OrderFlowAI/tests/test_classifier.py): Suite de pruebas unitarias.
