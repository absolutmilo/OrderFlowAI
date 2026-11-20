/**
 * Main Frontend Application Script.
 * Manages WebSocket connection, live UI updates, probability sparklines, and tape rendering.
 */

document.addEventListener('DOMContentLoaded', () => {
  // UI Elements
  const elBtcPrice = document.getElementById('btc-price');
  const elCvdUsd = document.getElementById('cvd-usd');
  const elTps = document.getElementById('trades-per-sec');
  const elConnectionStatus = document.getElementById('connection-status');
  const elStatusText = document.getElementById('status-text');
  const elTapeList = document.getElementById('tape-list');
  const elWhaleTapeList = document.getElementById('whale-tape-list');
  const elWhaleQueueCount = document.getElementById('whale-queue-count');
  const btnClearWhaleQueue = document.getElementById('clear-whale-queue');
  const elPatternNotes = document.getElementById('pattern-notes');
  const elFeedStats = document.getElementById('feed-stats');
  const elClassificationBadge = document.getElementById('classification-badge');
  const elInferenceMode = document.getElementById('inference-mode');
  const elInferenceModeStatus = document.getElementById('inference-mode-status');
  const infoDialog = document.getElementById('detector-info-dialog');
  const btnDetectorInfo = document.getElementById('btn-detector-info');
  const btnCloseInfo = document.getElementById('btn-close-info');
  const elDataCaptureStatus = document.getElementById('data-capture-status');

  // Probability Elements
  const elProbAlgo = document.getElementById('prob-algo');
  const elSparkAlgo = document.getElementById('spark-algo');

  const elProbIceberg = document.getElementById('prob-iceberg');
  const elSparkIceberg = document.getElementById('spark-iceberg');
  const elDescIceberg = document.getElementById('desc-iceberg');

  const elProbAbsorption = document.getElementById('prob-absorption');
  const elSparkAbsorption = document.getElementById('spark-absorption');
  const elDescAbsorption = document.getElementById('desc-absorption');

  const elProbRetail = document.getElementById('prob-retail');
  const elSparkRetail = document.getElementById('spark-retail');

  const elProbHft = document.getElementById('prob-hft');
  const elSparkHft = document.getElementById('spark-hft');

  const elCvdProfileWindow = document.getElementById('cvd-profile-window');
  const elCvdProfileRows = document.getElementById('cvd-profile-rows');
  const elCvdProfileDelta = document.getElementById('cvd-profile-delta');
  const elCvdProfileVolume = document.getElementById('cvd-profile-volume');
  const elCvdProfileVwap = document.getElementById('cvd-profile-vwap');
  const elCvdProfileVwapDistance = document.getElementById('cvd-profile-vwap-distance');
  const elCvdProfileStatus = document.getElementById('cvd-profile-status');
  const elMarketComparisonStatus = document.getElementById('market-comparison-status');

  // Audio Controls
  const btnToggleAudio = document.getElementById('btn-toggle-audio');
  const btnTestAudio = document.getElementById('btn-test-audio');
  const sliderVolume = document.getElementById('slider-volume');
  const selectThreshold = document.getElementById('select-threshold');
  const btnScalpBacktest = document.getElementById('btn-scalp-backtest');
  const selectScalpStrategy = document.getElementById('scalp-strategy');
  const selectScalpTimeframe = document.getElementById('scalp-timeframe');
  const selectScalpDirection = document.getElementById('scalp-direction');
  const selectScalpHmmExit = document.getElementById('scalp-hmm-exit');
  const btnScalpCandleBacktest = document.getElementById('btn-scalp-candle-backtest');

  const MAX_TAPE_ROWS = 120;
  const MAX_WHALE_ROWS = 120;
  const PROBABILITY_WINDOW_MS = 60 * 60 * 1000;
  const PROBABILITY_SAMPLE_MS = 5 * 1000;
  const PROBABILITY_SERIES = {
    algo_institutional: { path: elSparkAlgo, dot: document.getElementById('spark-dot-algo') },
    iceberg: { path: elSparkIceberg, dot: document.getElementById('spark-dot-iceberg') },
    absorption: { path: elSparkAbsorption, dot: document.getElementById('spark-dot-absorption') },
    retail: { path: elSparkRetail, dot: document.getElementById('spark-dot-retail') },
    hft_competition: { path: elSparkHft, dot: document.getElementById('spark-dot-hft') },
  };
  const probabilityHistory = Object.fromEntries(
    Object.keys(PROBABILITY_SERIES).map((key) => [key, []])
  );
  let socket = null;
  let reconnectTimer = null;
  let modeStatusTimer = null;
  let probabilityMode = null;
  let lastProbabilitySampleBucket = -1;
  let historyRequestId = 0;
  let profileRequestId = 0;
  let latestMarketPrice = null;
  let whaleQueueSize = 0;

  btnDetectorInfo.addEventListener('click', () => infoDialog.showModal());
  btnCloseInfo.addEventListener('click', () => infoDialog.close());
  infoDialog.addEventListener('click', (event) => {
    if (event.target === infoDialog) infoDialog.close();
  });

  // Audio setup
  btnToggleAudio.addEventListener('click', () => {
    const isNowActive = window.audioEngine.toggle();
    if (isNowActive) {
      btnToggleAudio.classList.add('active');
      btnToggleAudio.innerHTML = '🔊 Sonificación: ACTIVA';
      // Play brief test ping
      window.audioEngine.testChime();
    } else {
      btnToggleAudio.classList.remove('active');
      btnToggleAudio.innerHTML = '🔇 Sonificación: MUTED';
    }
  });

  if (btnTestAudio) {
    btnTestAudio.addEventListener('click', () => {
      window.audioEngine.testChime();
    });
  }

  if (btnClearWhaleQueue) {
    btnClearWhaleQueue.addEventListener('click', () => {
      if (elWhaleTapeList) {
        elWhaleTapeList.replaceChildren();
        const emptyMessage = document.createElement('p');
        emptyMessage.className = 'tape-empty';
        emptyMessage.textContent = 'Las transacciones ≥ $100k aparecerán aquí.';
        elWhaleTapeList.appendChild(emptyMessage);
      }
      whaleQueueSize = 0;
      updateWhaleQueueCount();
    });
  }

  sliderVolume.addEventListener('input', (e) => {
    window.audioEngine.setVolume(parseFloat(e.target.value));
  });

  selectThreshold.addEventListener('change', (e) => {
    window.audioEngine.setThreshold(parseFloat(e.target.value));
  });

  // Toggle inference mode (Heuristic vs HMM Model)
  if (elInferenceMode) {
    elInferenceMode.addEventListener('click', async () => {
      if (elInferenceMode.disabled) return;

      elInferenceMode.disabled = true;
      elInferenceMode.setAttribute('aria-busy', 'true');
      setInferenceModeStatus('Cambiando modo...');

      try {
        const response = await fetch('/api/inference-mode/toggle', { method: 'POST' });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.detail || `Error HTTP ${response.status}`);
        }
        if (!data.inference_mode) {
          throw new Error('El servidor no devolvió el modo de inferencia.');
        }
        updateSystemMode(data.inference_mode);
        setInferenceModeStatus('Modo actualizado.');
      } catch (error) {
        console.error('Error alternando modo de inferencia:', error);
        setInferenceModeStatus(`No se pudo cambiar: ${error.message}`, true);
      } finally {
        elInferenceMode.disabled = false;
        elInferenceMode.removeAttribute('aria-busy');
      }
    });
  }

  function setInferenceModeStatus(message, isError = false) {
    if (!elInferenceModeStatus) return;
    if (modeStatusTimer) clearTimeout(modeStatusTimer);
    elInferenceModeStatus.textContent = message;
    elInferenceModeStatus.classList.toggle('error', isError);
    if (message && !isError) {
      modeStatusTimer = setTimeout(() => {
        elInferenceModeStatus.textContent = '';
      }, 2500);
    }
  }

  function renderScalpingSignal(scalping) {
    if (!scalping) return;
    const signal = scalping.signal || { status: 'no_setup' };
    const signalElement = document.getElementById('scalp-signal');
    const detailElement = document.getElementById('scalp-signal-detail');
    const dataStatusElement = document.getElementById('scalp-data-status');
    if (!signalElement || !detailElement) return;

    signalElement.classList.remove('long', 'short');
    if (signal.status === 'signal') {
      signalElement.textContent = `SETUP ${signal.direction}: ${Number(signal.entry_price).toLocaleString('en-US', { maximumFractionDigits: 2 })}`;
      signalElement.classList.add(signal.direction === 'LONG' ? 'long' : 'short');
      detailElement.textContent = `SL ${Number(signal.stop_loss).toLocaleString('en-US', { maximumFractionDigits: 2 })} · TP ${Number(signal.take_profit).toLocaleString('en-US', { maximumFractionDigits: 2 })} · ATR ${signal.atr_bps} bps · ${signal.horizon_minutes} min`;
    } else if (signal.status === 'no_setup') {
      signalElement.textContent = scalping.alert_enabled ? 'Sin setup confirmado' : 'Esperando backtest validado';
      detailElement.textContent = scalping.alert_enabled
        ? 'Se requiere una secuencia confirmada de flujo y momentum.'
        : 'No se muestran entradas hasta que la validación neta fuera de muestra sea suficiente.';
    } else {
      signalElement.textContent = 'Señal no disponible';
      detailElement.textContent = signal.message || '';
    }

    if (dataStatusElement && scalping.observations !== undefined) {
      const required = scalping.minimum_observations || scalping.minimum_training_observations;
      const hmmCount = Number(scalping.hmm_snapshot_count || 0).toLocaleString('es-ES');
      const databaseSizeMb = (Number(scalping.database_size_bytes || 0) / (1024 * 1024)).toLocaleString('es-ES', { maximumFractionDigits: 1 });
      const databaseLimitMb = (Number(scalping.database_max_size_bytes || 0) / (1024 * 1024)).toLocaleString('es-ES', { maximumFractionDigits: 0 });
      dataStatusElement.textContent = `Histórico ${scalping.source}: ${Number(scalping.observations).toLocaleString('es-ES')} / ${Number(required).toLocaleString('es-ES')} snapshots (${Number(scalping.minimum_training_observations || required).toLocaleString('es-ES')} para entrenamiento) · DB ${databaseSizeMb}/${databaseLimitMb} MB · ${hmmCount} con estado HMM-4 · Últimas 24 h: cobertura ${Number(scalping.recent_snapshot_coverage_pct || 0)}%, ${Number(scalping.recent_observations || 0).toLocaleString('es-ES')} snapshots, hueco máximo ${Number(scalping.recent_maximum_snapshot_gap_seconds || 0)} s.`;
    }
  }

  function renderScalpingBacktest(backtest) {
    const resultElement = document.getElementById('scalp-backtest-result');
    const metricsElement = document.getElementById('scalp-backtest-metrics');
    const breakdownElement = document.getElementById('scalp-backtest-breakdown');
    const exitsElement = document.getElementById('scalp-backtest-exits');
    if (!resultElement || !metricsElement || !backtest) return;

    const cacheLabel = {
      cached: 'Velas reutilizadas desde caché local.',
      updated: `Caché actualizada con ${Number(backtest.downloaded_candles || 0).toLocaleString('es-ES')} velas recientes.`,
      downloaded: `Histórico descargado y guardado en caché (${Number(backtest.downloaded_candles || 0).toLocaleString('es-ES')} velas).`,
      resumed: `Descarga histórica reanudada desde la caché (${Number(backtest.downloaded_candles || 0).toLocaleString('es-ES')} velas adicionales).`,
    }[backtest.cache_status];
    resultElement.textContent = `${backtest.message || `Resultado: ${backtest.status}`}${cacheLabel ? ` ${cacheLabel}` : ''}`;
    if (backtest.validation) {
      const validation = backtest.validation;
      const training = backtest.training;
      const parameters = backtest.parameters;
      const candleSource = backtest.data_source === 'bybit_rest_klines';
      const sampleDescription = candleSource
        ? `${Number(backtest.candle_count || 0).toLocaleString('es-ES')} velas OHLCV (${backtest.lookback_days} días)`
        : 'snapshots guardados en SQLite';
      const parameterDescription = candleSource
        ? `EMA ${parameters.ema_fast_period}/${parameters.ema_slow_period}, volumen ${parameters.volume_lookback_bars} velas, horizonte ${parameters.holding_period_bars} velas`
        : `SL ${parameters.stop_atr_multiple} ATR, TP ${parameters.target_atr_multiple} ATR`;
      const lowerBound = validation.expectancy_lower_95_bps;
      const lowerBoundText = lowerBound === null || lowerBound === undefined ? 'n/d' : `${lowerBound}`;
      metricsElement.textContent = `Estrategia: ${backtest.strategy_name} · ${backtest.timeframe_minutes} min · dirección ${backtest.direction_mode_name || 'Original'} · ${sampleDescription}. Entrenamiento (${training.trade_count} ops): bruto ${training.gross_expectancy_bps} bps/operación, neto ${training.expectancy_bps} bps/operación. Validación (${validation.trade_count} ops; mínimo ${backtest.minimum_validation_trades}): bruto ${validation.gross_expectancy_bps} bps/operación, neto ${validation.expectancy_bps} bps/operación · límite inferior HAC 95% ${lowerBoundText} bps/op · aciertos netos ${validation.win_rate_pct}% · PF ${validation.profit_factor} · DD ${validation.max_drawdown_bps} bps · costos ${validation.total_costs_bps} bps. ${parameterDescription}.`;
      if (breakdownElement && validation.by_direction) {
        const long = validation.by_direction.LONG;
        const short = validation.by_direction.SHORT;
        breakdownElement.textContent = `Por dirección — LONG: ${long.trade_count} ops, neto ${long.expectancy_bps} bps/op, aciertos ${long.win_rate_pct}%, PF ${long.profit_factor}; SHORT: ${short.trade_count} ops, neto ${short.expectancy_bps} bps/op, aciertos ${short.win_rate_pct}%, PF ${short.profit_factor}.`;
      }
      if (exitsElement && validation.by_exit_reason) {
        const exits = validation.by_exit_reason;
        exitsElement.textContent = `Salidas — stop: ${exits.stop_loss.trade_count} (neto ${exits.stop_loss.total_net_bps} bps); objetivo: ${exits.take_profit.trade_count} (neto ${exits.take_profit.total_net_bps} bps); tiempo: ${exits.time.trade_count} (neto ${exits.time.total_net_bps} bps).`;
      }
    } else {
      if (backtest.status === 'no_trainable_setup') {
        const diagnostics = backtest.signal_diagnostics || {};
        const bars = Number(backtest.training_bars || 0);
        const requiredBars = Number(backtest.required_training_bars || 0);
        const candleLabel = backtest.data_source === 'bybit_rest_klines'
          ? `OHLCV descargado: ${Number(backtest.candle_count || 0).toLocaleString('es-ES')} velas · `
          : '';
        metricsElement.textContent = `${backtest.strategy_name} · ${backtest.timeframe_minutes} min · dirección ${backtest.direction_mode_name || 'Original'}. ${candleLabel}${bars} velas de entrenamiento${requiredBars ? ` (${requiredBars} requeridas)` : ''} · ${Number(backtest.training_signal_count || 0)} señales · máximo ${Number(backtest.max_training_trades || 0)} operaciones simuladas; se requieren ${Number(backtest.minimum_training_trades || 0)}.`;
        if (breakdownElement) {
          breakdownElement.textContent = diagnostics.complete_bars !== undefined
            ? `Diagnóstico Fourier: ${diagnostics.raw_anomalies} anomalías · ${diagnostics.persistent_anomaly_bars} barras anómalas persistentes · ${diagnostics.completed_zones} zonas cerradas · ${diagnostics.context_qualified_bars} barras con contexto · ${diagnostics.breakout_candidates} rupturas.`
            : '';
        }
      } else {
        metricsElement.textContent = backtest.status === 'insufficient_data_quality'
          ? `Última validación: cobertura histórica ${Number(backtest.snapshot_coverage_pct || 0)}% · hueco máximo ${Number(backtest.maximum_snapshot_gap_seconds || 0)} s. Se requiere captura continua para validar el backtest.`
          : backtest.observations !== undefined
            ? backtest.data_source === 'bybit_rest_klines'
              ? `${backtest.strategy_name} · ${backtest.timeframe_minutes} min. ${Number(backtest.candle_count || 0).toLocaleString('es-ES')} velas OHLCV descargadas; ${Number(backtest.training_observations || 0).toLocaleString('es-ES')} en entrenamiento.`
              : `${backtest.strategy_name || ''} · ${backtest.timeframe_minutes || ''} min. ${Number(backtest.observations).toLocaleString('es-ES')} observaciones Bybit Linear; ${Number(backtest.training_observations || 0).toLocaleString('es-ES')} snapshots en entrenamiento${backtest.training_bars !== undefined ? `; ${backtest.training_bars}/${backtest.required_training_bars || '?'} velas completas` : ''}${backtest.required_training_observations !== undefined ? `; se requieren ${Number(backtest.required_training_observations).toLocaleString('es-ES')} para entrenar y ${Number(backtest.required_observations || 0).toLocaleString('es-ES')} en total` : ''}.`
          : '';
        if (breakdownElement) breakdownElement.textContent = '';
      }
      if (exitsElement) exitsElement.textContent = '';
    }
  }

  async function loadScalpingStatus() {
    try {
      const response = await fetch('/api/scalping/status');
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `Error HTTP ${response.status}`);
      renderScalpingSignal(data);
      if (data.backtest) renderScalpingBacktest(data.backtest);
    } catch (error) {
      console.error('Error consultando estado de scalping:', error);
      const status = document.getElementById('scalp-data-status');
      if (status) status.textContent = `No se pudo consultar el histórico: ${error.message}`;
    }
  }

  if (btnScalpBacktest) {
    btnScalpBacktest.addEventListener('click', async () => {
      btnScalpBacktest.disabled = true;
      btnScalpBacktest.textContent = 'Optimizando...';
      const resultElement = document.getElementById('scalp-backtest-result');
      if (resultElement) resultElement.textContent = 'Ejecutando optimización cronológica; puede tardar un momento.';
      try {
        const query = new URLSearchParams({
          strategy: selectScalpStrategy ? selectScalpStrategy.value : 'flow_momentum',
          timeframe_minutes: selectScalpTimeframe ? selectScalpTimeframe.value : '5',
          direction_mode: selectScalpDirection ? selectScalpDirection.value : 'original',
          early_exit_on_hmm: selectScalpHmmExit ? selectScalpHmmExit.value : 'true',
        });
        const response = await fetch(`/api/scalping/backtest?${query}`, { method: 'POST' });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || `Error HTTP ${response.status}`);
        renderScalpingBacktest(data);
        renderScalpingSignal({
          signal: { status: 'no_setup' },
          alert_enabled: data.alert_enabled,
          source: data.source,
        });
        await loadScalpingStatus();
      } catch (error) {
        console.error('Error ejecutando el backtest de scalping:', error);
        if (resultElement) resultElement.textContent = `Backtest fallido: ${error.message}`;
      } finally {
        btnScalpBacktest.disabled = false;
        btnScalpBacktest.textContent = 'Validar order flow (SQLite)';
      }
    });
  }

  if (btnScalpCandleBacktest) {
    btnScalpCandleBacktest.addEventListener('click', async () => {
      btnScalpCandleBacktest.disabled = true;
      btnScalpCandleBacktest.textContent = 'Revisando caché...';
      const resultElement = document.getElementById('scalp-backtest-result');
      if (resultElement) {
        resultElement.textContent = 'Revisando el histórico guardado; solo se descargarán velas si faltan.';
      }
      try {
        const query = new URLSearchParams({
          timeframe_minutes: selectScalpTimeframe ? selectScalpTimeframe.value : '5',
          direction_mode: selectScalpDirection ? selectScalpDirection.value : 'original',
        });
        const response = await fetch(`/api/scalping/candle-backtest?${query}`, { method: 'POST' });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || `Error HTTP ${response.status}`);
        renderScalpingBacktest(data);
      } catch (error) {
        console.error('Error ejecutando el backtest OHLCV de Bybit:', error);
        if (resultElement) {
          resultElement.textContent = error.message.includes('10016')
            ? 'Bybit está respondiendo con un error temporal (10016). Se reintentó la solicitud y el progreso descargado quedó guardado; vuelve a pulsar para continuar.'
            : `Backtest OHLCV fallido: ${error.message}`;
        }
      } finally {
        btnScalpCandleBacktest.disabled = false;
        btnScalpCandleBacktest.textContent = 'Validar velas OHLCV directo';
      }
    });
  }

  if (elCvdProfileWindow) {
    elCvdProfileWindow.addEventListener('change', loadOrderFlowProfile);
    loadOrderFlowProfile();
    window.setInterval(loadOrderFlowProfile, 5000);
  }

  loadScalpingStatus();

  function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/stream`;

    console.log(`Conectando WebSocket a ${wsUrl}...`);
    socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      console.log('WebSocket local conectado exitosamente.');
      elStatusText.textContent = 'RECIBIENDO FEED (BINANCE + BYBIT)';
      elConnectionStatus.style.color = '#00e676';
      if (reconnectTimer) clearTimeout(reconnectTimer);
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.market_comparison) {
          renderMarketComparison(data.market_comparison);
        }

        if (data.type === 'order_flow_tick') {
          handleTradeTick(data.trade);
          if (data.exchange_stats) {
            updateExchangeStats(
              data.exchange_stats,
              data.primary_market_source,
              data.feature_windows_seconds
            );
          }
          if (data.market_source === data.primary_market_source) {
            updateProbabilities(data.probabilities);
            if (data.metrics) updateMetrics(data.metrics);
            updateSystemMode(data.inference_mode, data.data_capture_status, data.detector_config);
          }
          renderScalpingSignal(data.scalping);
        } else if (data.type === 'snapshot') {
          updateProbabilities(data.probabilities);
          if (data.metrics) updateMetrics(data.metrics);
          if (data.exchange_stats) {
            updateExchangeStats(
              data.exchange_stats,
              data.primary_market_source,
              data.feature_windows_seconds
            );
          }
          updateSystemMode(data.inference_mode, data.data_capture_status, data.detector_config);
          renderScalpingSignal(data.scalping);
        } else if (data.type === 'scalping_backtest') {
          renderScalpingBacktest(data.backtest);
          renderScalpingSignal({
            signal: { status: 'no_setup' },
            alert_enabled: data.alert_enabled,
          });
        } else if (data.type === 'mode_change') {
          updateSystemMode(data.inference_mode);
        } else if (data.type === 'mode_error') {
          setInferenceModeStatus(data.message || 'No se pudo cambiar el modo.', true);
        }
      } catch (err) {
        console.error('Error procesando mensaje WS:', err);
      }
    };

    socket.onclose = () => {
      console.warn('WebSocket desconectado. Reintentando en 2s...');
      elStatusText.textContent = 'RECONECTANDO A SERVIDOR...';
      elConnectionStatus.style.color = '#ff9100';
      reconnectTimer = setTimeout(connectWebSocket, 2000);
    };

    socket.onerror = (err) => {
      console.error('WebSocket error:', err);
      socket.close();
    };
  }

  function updateExchangeStats(stats, primaryMarketSource, featureWindows) {
    if (!stats || !elFeedStats) return;
    const windows = Array.isArray(featureWindows) ? featureWindows.join('/') : '';
    const windowLabel = windows ? `(${windows}s)` : '';
    elFeedStats.textContent = `BINANCE: ${stats.Binance || 0} | BYBIT: ${stats.Bybit || 0} · Clasificador: ${primaryMarketSource || '—'} ${windowLabel}`;
  }

  function updateSystemMode(mode, captureStatus, detectorConfig) {
    if (elInferenceMode && mode) {
      if (probabilityMode !== mode) {
        probabilityMode = mode;
        resetProbabilityHistory();
        loadProbabilityHistory(mode);
      }
      const isModel = mode === 'model';
      elInferenceMode.textContent = isModel ? '⚡ MODO: MODELO (HMM-4)' : '⚙️ MODO: HEURÍSTICO';
      elInferenceMode.title = `Click para alternar a modo ${isModel ? 'Heurístico' : 'Modelo HMM'}`;
      if (isModel) {
        elInferenceMode.classList.add('mode-model');
        elInferenceMode.classList.remove('mode-heuristic');
      } else {
        elInferenceMode.classList.add('mode-heuristic');
        elInferenceMode.classList.remove('mode-model');
      }
    }
    if (elDataCaptureStatus && captureStatus) {
      const statusText = {
        recording: 'registrando snapshots en SQLite',
        waiting_for_warmup: 'esperando que termine el calentamiento',
        error: 'error al guardar; revisa el log del servidor',
        stopped: 'captura detenida'
      };
      elDataCaptureStatus.textContent = `Estado de captura: ${statusText[captureStatus] || captureStatus}.`;
    }
    if (detectorConfig) updateDetectorConfig(detectorConfig);
  }

  function updateDetectorConfig(config) {
    const setValue = (id, value) => {
      const element = document.getElementById(id);
      if (element) element.textContent = value;
    };
    const percentage = (value) => `${(value * 100).toLocaleString('es-ES', { maximumFractionDigits: 1 })} %`;
    const usd = (value) => `$${Number(value).toLocaleString('en-US', { maximumFractionDigits: 0 })}`;

    setValue('algo-time-threshold', config.algo_inter_arrival_cv_threshold.toFixed(2));
    setValue('algo-size-threshold', config.algo_size_cv_threshold.toFixed(2));
    setValue('iceberg-ratio', percentage(config.iceberg_min_volume_ratio));
    setValue('iceberg-count', config.iceberg_min_executions_at_level);
    setValue('iceberg-usd', usd(config.iceberg_min_usd_at_level));
    setValue('absorption-usd', usd(config.absorption_min_usd_vol));
    setValue('absorption-ratio', percentage(config.absorption_min_cvd_ratio));
    setValue('absorption-bps', config.absorption_max_price_delta_bps.toLocaleString('es-ES'));
    setValue('hft-tps', config.hft_burst_tps_threshold.toLocaleString('es-ES'));
    setValue('breakout-bps', config.breakout_min_price_delta_bps.toLocaleString('es-ES'));
    setValue('breakout-ratio', percentage(config.breakout_min_cvd_ratio));
  }

  function updateMetrics(metrics) {
    if (!metrics) return;
    latestMarketPrice = Number(metrics.current_price);
    elBtcPrice.textContent = `$${metrics.current_price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

    const cvdVal = metrics.cvd_usd;
    const cvdSign = cvdVal >= 0 ? '+' : '';
    elCvdUsd.textContent = `${cvdSign}$${(cvdVal / 1000).toFixed(1)}k`;
    elCvdUsd.style.color = cvdVal >= 0 ? '#00e676' : '#ff1744';

    elTps.textContent = `${metrics.trades_per_second.toFixed(1)} /s`;
  }

  function formatMarketPrice(value) {
    return `$${Number(value).toLocaleString('en-US', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })}`;
  }

  function formatSignedSpread(value, fractionDigits = 2) {
    const number = Number(value);
    const sign = number >= 0 ? '+' : '−';
    return `${sign}$${Math.abs(number).toLocaleString('en-US', {
      minimumFractionDigits: fractionDigits,
      maximumFractionDigits: fractionDigits,
    })}`;
  }

  function formatSignedBps(value) {
    const number = Number(value);
    return `${number >= 0 ? '+' : '−'}${Math.abs(number).toFixed(2)} bps`;
  }

  function renderMarketComparison(comparison) {
    if (!comparison) return;

    const prices = comparison.prices || {};
    const flows = comparison.flows || {};
    const renderPrice = (elementId, quote, waitingText) => {
      const element = document.getElementById(elementId);
      if (!element) return;
      if (!quote || !Number.isFinite(Number(quote.price))) {
        element.textContent = waitingText;
        element.classList.add('spread-stale');
        return;
      }
      element.textContent = `${formatMarketPrice(quote.price)}${quote.fresh ? '' : ' · desactualizado'}`;
      element.classList.toggle('spread-stale', !quote.fresh);
    };
    const renderFlow = (cvdElementId, moveElementId, flow) => {
      const cvdElement = document.getElementById(cvdElementId);
      const moveElement = document.getElementById(moveElementId);
      if (!cvdElement || !moveElement) return;
      if (!flow || !flow.fresh) {
        cvdElement.textContent = 'Sin datos recientes';
        moveElement.textContent = '--';
        cvdElement.classList.add('spread-stale');
        moveElement.classList.add('spread-stale');
        return;
      }
      cvdElement.textContent = formatSignedSpread(flow.cvd_usd, 0);
      moveElement.textContent = formatSignedBps(flow.price_delta_bps);
      cvdElement.classList.toggle('spread-positive', Number(flow.cvd_usd) >= 0);
      cvdElement.classList.toggle('spread-negative', Number(flow.cvd_usd) < 0);
      cvdElement.classList.remove('spread-stale');
      moveElement.classList.toggle('spread-positive', Number(flow.price_delta_bps) >= 0);
      moveElement.classList.toggle('spread-negative', Number(flow.price_delta_bps) < 0);
      moveElement.classList.remove('spread-stale');
    };
    const renderSpread = (elementId, spread, waitingText) => {
      const element = document.getElementById(elementId);
      if (!element) return;
      if (!spread || !Number.isFinite(Number(spread.difference_usd))) {
        element.textContent = waitingText;
        element.classList.add('spread-stale');
        return;
      }
      if (!spread.fresh) {
        element.textContent = 'Datos desactualizados';
        element.classList.add('spread-stale');
        return;
      }
      element.textContent = `${formatSignedSpread(spread.difference_usd)} · ${formatSignedBps(spread.difference_bps)}`;
      element.classList.remove('spread-stale');
      element.classList.toggle('spread-positive', Number(spread.difference_usd) >= 0);
      element.classList.toggle('spread-negative', Number(spread.difference_usd) < 0);
    };

    renderPrice('comparison-binance-price', prices['Binance-Spot'], 'Esperando Binance');
    renderPrice('comparison-bybit-price', prices['Bybit-Linear'], 'Esperando Bybit');
    renderPrice('comparison-aggr-price', prices['AGGR-BTCUSD'], 'Conectando...');
    renderFlow(
      'comparison-binance-cvd',
      'comparison-binance-move',
      flows['Binance-Spot'],
    );
    renderFlow(
      'comparison-bybit-cvd',
      'comparison-bybit-move',
      flows['Bybit-Linear'],
    );
    const spreads = comparison.spreads || {};
    renderSpread(
      'spread-bybit-binance',
      spreads.bybit_vs_binance,
      'Esperando ambos feeds',
    );
    renderSpread(
      'spread-aggr-bybit',
      spreads.aggr_vs_bybit,
      'Conectando...',
    );

    if (elMarketComparisonStatus) {
      const freshCount = ['Binance-Spot', 'Bybit-Linear', 'AGGR-BTCUSD']
        .filter((source) => prices[source] && prices[source].fresh)
        .length;
      elMarketComparisonStatus.textContent = `Fuentes activas: ${freshCount}/3 · actualización esperada cada 1 s`;
    }
  }

  function formatProfilePrice(value) {
    return `$${value.toLocaleString('en-US', { maximumFractionDigits: 2 })}`;
  }

  function formatProfileUsd(value, signed = false) {
    const absolute = Math.abs(value);
    const amount = absolute >= 1_000_000
      ? `${(absolute / 1_000_000).toFixed(1)}M`
      : absolute >= 1_000
        ? `${(absolute / 1_000).toFixed(1)}k`
        : absolute.toFixed(0);
    const sign = signed ? (value >= 0 ? '+' : '-') : (value < 0 ? '-' : '');
    return `${sign}$${amount}`;
  }

  function formatProfileTime(timestamp) {
    if (!Number.isFinite(Number(timestamp))) return '';
    return new Date(Number(timestamp)).toLocaleTimeString('es-ES', {
      timeZone: 'UTC',
      hour: '2-digit',
      minute: '2-digit',
    });
  }

  function adaptivePriceStep(range) {
    const rawStep = Math.max(0.1, range / 24);
    const magnitude = 10 ** Math.floor(Math.log10(rawStep));
    const normalized = rawStep / magnitude;
    const factor = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
    return Math.max(0.1, factor * magnitude);
  }

  function addProfileCell(row, className, text) {
    const cell = document.createElement('span');
    cell.className = className;
    cell.textContent = text;
    row.appendChild(cell);
    return cell;
  }

  function renderOrderFlowProfile(data) {
    if (!elCvdProfileRows || !elCvdProfileStatus) return;
    const levels = (data.levels || []).filter((level) =>
      Number.isFinite(Number(level.price))
      && Number.isFinite(Number(level.buy_volume_usd))
      && Number.isFinite(Number(level.sell_volume_usd))
      && Number.isFinite(Number(level.delta_usd))
    );
    const totalDelta = levels.reduce((total, level) => total + Number(level.delta_usd), 0);
    const totalVolume = levels.reduce((total, level) =>
      total + Number(level.buy_volume_usd) + Number(level.sell_volume_usd), 0);

    if (elCvdProfileDelta) {
      elCvdProfileDelta.textContent = levels.length
        ? formatProfileUsd(totalDelta, true)
        : 'Esperando trades';
      elCvdProfileDelta.classList.toggle('positive', totalDelta >= 0 && levels.length > 0);
      elCvdProfileDelta.classList.toggle('negative', totalDelta < 0);
    }
    if (elCvdProfileVolume) {
      elCvdProfileVolume.textContent = levels.length ? formatProfileUsd(totalVolume) : '--';
    }

    let adaptiveBandWidth = null;
    const currentPriceFromServer = Number(data.current_price);
    const currentPrice = Number.isFinite(currentPriceFromServer) && currentPriceFromServer > 0
      ? currentPriceFromServer
      : latestMarketPrice;
    const hasCurrentPrice = Number.isFinite(currentPrice) && currentPrice > 0;
    const vwap = Number(data.session_vwap);
    const hasVwap = Number.isFinite(vwap) && vwap > 0;
    if (elCvdProfileVwap) {
      elCvdProfileVwap.textContent = hasVwap ? formatProfilePrice(vwap) : 'Esperando trades';
    }
    if (elCvdProfileVwapDistance) {
      if (hasVwap && hasCurrentPrice) {
        const distanceBps = ((currentPrice - vwap) / vwap) * 10_000;
        const relation = distanceBps >= 0 ? 'sobre' : 'bajo';
        elCvdProfileVwapDistance.textContent =
          `Precio ${Math.abs(distanceBps).toFixed(1)} bps ${relation} el VWAP`;
      } else {
        elCvdProfileVwapDistance.textContent = 'VWAP diario anclado a las 00:00 UTC';
      }
    }

    if (!levels.length) {
      elCvdProfileRows.replaceChildren();
      const empty = document.createElement('p');
      empty.className = 'profile-empty';
      empty.textContent = 'Esperando trades de Bybit Linear para construir el perfil...';
      elCvdProfileRows.appendChild(empty);
    } else {
      const priceBounds = levels.reduce((bounds, level) => ({
        min: Math.min(bounds.min, Number(level.price)),
        max: Math.max(bounds.max, Number(level.price)),
      }), { min: Infinity, max: -Infinity });
      const step = adaptivePriceStep(priceBounds.max - priceBounds.min);
      adaptiveBandWidth = step;
      const buckets = new Map();
      levels.forEach((level) => {
        const price = Number(level.price);
        const key = Math.floor(price / step);
        const bucket = buckets.get(key) || {
          buyUsd: 0,
          sellUsd: 0,
          deltaUsd: 0,
        };
        bucket.buyUsd += Number(level.buy_volume_usd);
        bucket.sellUsd += Number(level.sell_volume_usd);
        bucket.deltaUsd += Number(level.delta_usd);
        buckets.set(key, bucket);
      });

      const profile = [...buckets.entries()]
        .map(([key, bucket]) => ({
          key,
          price: key * step,
          ...bucket,
          volumeUsd: bucket.buyUsd + bucket.sellUsd,
        }))
        .sort((left, right) => right.price - left.price);
      const maxVolume = profile.reduce((maximum, level) =>
        Math.max(maximum, level.volumeUsd), 0
      );
      const maxDelta = profile.reduce((maximum, level) =>
        Math.max(maximum, Math.abs(level.deltaUsd)), 0
      );
      const pointOfControl = profile.reduce((best, level) =>
        level.volumeUsd > best.volumeUsd ? level : best
      );
      const currentBucket = hasCurrentPrice ? Math.floor(currentPrice / step) : null;
      const vwapBucket = hasVwap ? Math.floor(vwap / step) : null;
      const fragment = document.createDocumentFragment();

      profile.forEach((level) => {
        const isCurrent = currentBucket === level.key;
        const isVwap = vwapBucket === level.key;
        const isPoc = pointOfControl.key === level.key;
        const row = document.createElement('div');
        row.className = `cvd-profile-level${isCurrent ? ' current' : ''}${isVwap ? ' vwap' : ''}${isPoc ? ' poc' : ''}`;
        row.setAttribute('role', 'row');
        row.setAttribute(
          'aria-label',
          `${formatProfilePrice(level.price)}${isCurrent ? ', precio actual' : ''}${isVwap ? ', nivel VWAP' : ''}${isPoc ? ', punto de control' : ''}`
        );

        const priceCell = addProfileCell(row, 'profile-price', formatProfilePrice(level.price));
        const markers = [
          isCurrent ? 'ACTUAL' : '',
          isVwap ? 'VWAP' : '',
          isPoc ? 'POC' : '',
        ].filter(Boolean);
        if (markers.length) {
          const marker = document.createElement('small');
          marker.className = 'profile-markers';
          marker.textContent = ` · ${markers.join(' · ')}`;
          priceCell.appendChild(marker);
        }

        const volumeTrack = document.createElement('span');
        volumeTrack.className = 'profile-volume-track';
        volumeTrack.setAttribute('aria-hidden', 'true');
        const buyBar = document.createElement('span');
        buyBar.className = 'profile-volume-buy';
        buyBar.style.width = `${(level.buyUsd / maxVolume) * 100}%`;
        const sellBar = document.createElement('span');
        sellBar.className = 'profile-volume-sell';
        sellBar.style.width = `${(level.sellUsd / maxVolume) * 100}%`;
        volumeTrack.append(buyBar, sellBar);
        row.appendChild(volumeTrack);

        const deltaCell = document.createElement('span');
        deltaCell.className = 'profile-delta-cell';
        const deltaTrack = document.createElement('span');
        deltaTrack.className = 'profile-delta-track';
        deltaTrack.setAttribute('aria-hidden', 'true');
        const deltaBar = document.createElement('span');
        const deltaWidth = maxDelta > 0 ? (Math.abs(level.deltaUsd) / maxDelta) * 50 : 0;
        deltaBar.className = `profile-delta-bar${level.deltaUsd >= 0 ? ' positive' : ' negative'}`;
        deltaBar.style.width = `${deltaWidth}%`;
        deltaBar.style.left = level.deltaUsd >= 0 ? '50%' : `calc(50% - ${deltaWidth}%)`;
        deltaTrack.appendChild(deltaBar);
        const deltaValue = document.createElement('span');
        deltaValue.className = 'profile-delta-value';
        deltaValue.textContent = formatProfileUsd(level.deltaUsd, true);
        deltaCell.append(deltaTrack, deltaValue);
        row.appendChild(deltaCell);
        fragment.appendChild(row);
      });
      elCvdProfileRows.replaceChildren(fragment);
    }

    const selectedOption = elCvdProfileWindow
      ? elCvdProfileWindow.options[elCvdProfileWindow.selectedIndex]
      : null;
    const expectedBuckets = (startTimestamp, endTimestamp) => {
      const startMinute = Math.floor(Number(startTimestamp) / 60_000);
      const endMinute = Math.floor(Number(endTimestamp) / 60_000);
      return Math.max(0, endMinute - startMinute + 1);
    };
    const expectedProfileBuckets = expectedBuckets(data.profile_start_ms, data.profile_end_ms);
    const expectedSessionBuckets = expectedBuckets(data.session_start_ms, data.session_end_ms);
    const profileComplete = data.profile_first_trade_ms !== null
      && data.profile_first_trade_ms !== undefined
      && Number.isFinite(Number(data.profile_first_trade_ms))
      && Number(data.profile_first_trade_ms) <= Number(data.profile_start_ms)
      && Number(data.profile_bucket_count) >= expectedProfileBuckets;
    const sessionComplete = data.session_first_trade_ms !== null
      && data.session_first_trade_ms !== undefined
      && Number.isFinite(Number(data.session_first_trade_ms))
      && Number(data.session_first_trade_ms) <= Number(data.session_start_ms)
      && Number(data.session_bucket_count) >= expectedSessionBuckets;
    const profileCoverage = profileComplete
      ? 'ventana continua'
      : data.profile_first_trade_ms
        ? `parcial ${Number(data.profile_bucket_count) || 0}/${expectedProfileBuckets} min · desde ${formatProfileTime(data.profile_first_trade_ms)} UTC`
        : 'sin histórico todavía';
    const vwapCoverage = sessionComplete
      ? 'VWAP con sesión completa'
      : data.session_first_trade_ms
        ? `VWAP parcial ${Number(data.session_bucket_count) || 0}/${expectedSessionBuckets} min · desde ${formatProfileTime(data.session_first_trade_ms)} UTC`
        : 'VWAP esperando datos de la sesión';
    elCvdProfileStatus.classList.remove('error');
    elCvdProfileStatus.textContent = `${selectedOption ? selectedOption.textContent : 'Perfil'} · ${profileCoverage} · ${vwapCoverage}${adaptiveBandWidth ? ` · Bandas adaptativas de $${adaptiveBandWidth.toLocaleString('en-US')}` : ''} · Actualización cada 5 s.`;
  }

  async function loadOrderFlowProfile() {
    if (!elCvdProfileWindow || !elCvdProfileStatus) return;
    const requestId = ++profileRequestId;
    const query = new URLSearchParams({ window_minutes: elCvdProfileWindow.value });
    try {
      const response = await fetch(`/api/orderflow/profile?${query}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `Error HTTP ${response.status}`);
      if (requestId !== profileRequestId) return;
      renderOrderFlowProfile(data);
    } catch (error) {
      if (requestId !== profileRequestId) return;
      console.error('Error cargando el perfil de volumen/CVD:', error);
      elCvdProfileStatus.classList.add('error');
      elCvdProfileStatus.textContent = `No se pudo actualizar el perfil: ${error.message}`;
    }
  }

  function updateProbabilities(probs) {
    if (!probs) return;

    if (elClassificationBadge) {
      if (probs.dominant_regime && probs.dominant_regime !== 'RETAIL_DOMINANT') {
        elClassificationBadge.textContent = `REGIMEN: ${probs.dominant_regime}`;
        elClassificationBadge.style.color = '#00e5ff';
      } else {
        elClassificationBadge.textContent = 'INFERENCIA EN VIVO';
      }
    }

    // 1. Algo Institucional
    elProbAlgo.textContent = `${probs.algo_institutional}%`;

    // 2. Iceberg
    elProbIceberg.textContent = `${probs.iceberg}%`;
    if (probs.iceberg_side) {
      elDescIceberg.textContent = `Iceberg ${probs.iceberg_side} activo en nivel modal`;
    } else {
      elDescIceberg.textContent = `Acumulación oculta en nivel de precio`;
    }

    // 3. Absorción
    elProbAbsorption.textContent = `${probs.absorption}%`;
    if (probs.absorption_side) {
      elDescAbsorption.textContent = probs.absorption_side;
    } else {
      elDescAbsorption.textContent = `Órdenes límite absorbiendo presión agresiva`;
    }

    // 4. Retail
    elProbRetail.textContent = `${probs.retail}%`;

    // 5. HFT
    elProbHft.textContent = `${probs.hft_competition}%`;

    if (addProbabilitySample(probs, Number(probs.timestamp_ms) || Date.now())) {
      renderProbabilitySparklines();
    }

    // Actualizar notas en vivo
    if (probs.notes && probs.notes.length > 0) {
      elPatternNotes.innerHTML = probs.notes.map(n =>
        `<div class="pattern-note-item"><span class="note-bullet">▶</span> ${n}</div>`
      ).join('');
    }
  }

  function handleTradeTick(trade) {
    if (!trade) return;

    // Trigger Sonification
    window.audioEngine.playTradeSound(trade, trade.patterns || []);

    const isWhale = trade.usd_value >= 100000;
    appendTradeToTape(trade, elTapeList, isWhale, MAX_TAPE_ROWS);
    if (isWhale && elWhaleTapeList) {
      const emptyMessage = elWhaleTapeList.querySelector('.tape-empty');
      if (emptyMessage) emptyMessage.remove();
      appendTradeToTape(trade, elWhaleTapeList, true, MAX_WHALE_ROWS);
      whaleQueueSize = Math.min(whaleQueueSize + 1, MAX_WHALE_ROWS);
      updateWhaleQueueCount();
    }
  }

  function appendTradeToTape(trade, list, isWhale, maxRows) {
    if (!list) return;
    const isBuy = trade.side === 'BUY';
    const row = document.createElement('div');
    row.className = `tape-row ${isBuy ? 'buy' : 'sell'} ${isWhale ? 'whale' : ''}`;

    const date = new Date(trade.timestamp_ms);
    const timeStr = date.toTimeString().split(' ')[0] + '.' + String(date.getMilliseconds()).padStart(3, '0').slice(0, 2);
    const exchange = trade.exchange || '';
    const exchShort = exchange.includes('Binance') ? 'BN' : exchange.includes('Bybit') ? 'BYB' : '--';
    const cells = [
      timeStr,
      exchShort,
      `$${Number(trade.price).toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 })}`,
      `${Number(trade.size).toFixed(3)} BTC`,
      `$${Math.round(Number(trade.usd_value)).toLocaleString('en-US')}`,
    ];
    cells.forEach((value) => {
      const cell = document.createElement('span');
      cell.textContent = value;
      row.appendChild(cell);
    });

    const patternCell = document.createElement('span');
    const pattern = trade.patterns && trade.patterns.length ? trade.patterns[0] : trade.side;
    patternCell.textContent = pattern;
    patternCell.title = pattern;
    row.appendChild(patternCell);

    list.insertBefore(row, list.firstChild);
    while (list.children.length > maxRows) {
      list.removeChild(list.lastChild);
    }
  }

  function updateWhaleQueueCount() {
    if (elWhaleQueueCount) {
      elWhaleQueueCount.textContent = `${whaleQueueSize} / ${MAX_WHALE_ROWS}`;
    }
  }

  function resetProbabilityHistory() {
    Object.values(probabilityHistory).forEach((samples) => samples.splice(0));
    lastProbabilitySampleBucket = -1;
    renderProbabilitySparklines();
  }

  function addProbabilitySample(probabilities, timestamp) {
    const bucket = Math.floor(timestamp / PROBABILITY_SAMPLE_MS);
    if (bucket === lastProbabilitySampleBucket) return false;
    lastProbabilitySampleBucket = bucket;

    Object.keys(PROBABILITY_SERIES).forEach((key) => {
      const value = Number(probabilities[key]);
      if (!Number.isFinite(value)) return;
      const samples = probabilityHistory[key];
      samples.push({
        timestamp,
        value: Math.max(0, Math.min(100, value)),
      });
      while (
        samples.length
        && (timestamp - samples[0].timestamp > PROBABILITY_WINDOW_MS
          || samples.length > PROBABILITY_WINDOW_MS / PROBABILITY_SAMPLE_MS)
      ) {
        samples.shift();
      }
    });
    return true;
  }

  function renderProbabilitySparklines(now = Date.now()) {
    const start = now - PROBABILITY_WINDOW_MS;
    Object.entries(PROBABILITY_SERIES).forEach(([key, elements]) => {
      if (!elements.path) return;
      const samples = probabilityHistory[key].filter((sample) => sample.timestamp >= start);
      const segments = [];
      let segment = [];
      let previousTimestamp = null;
      let lastPoint = null;

      samples.forEach((sample) => {
        if (
          previousTimestamp !== null
          && sample.timestamp - previousTimestamp > PROBABILITY_SAMPLE_MS * 2
        ) {
          if (segment.length) segments.push(segment);
          segment = [];
        }
        const x = Math.max(0, Math.min(240, ((sample.timestamp - start) / PROBABILITY_WINDOW_MS) * 240));
        const y = 40 - (sample.value / 100) * 36;
        segment.push(`${segment.length ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`);
        lastPoint = { x, y, rawValue: sample.value };
        previousTimestamp = sample.timestamp;
      });
      if (segment.length) segments.push(segment);
      elements.path.setAttribute('d', segments.map((points) => points.join(' ')).join(' '));
      if (elements.dot) {
        elements.dot.style.display = lastPoint ? '' : 'none';
        if (lastPoint) {
          elements.dot.setAttribute('cx', lastPoint.x.toFixed(1));
          elements.dot.setAttribute('cy', lastPoint.y.toFixed(1));
        }
      }
    });
  }

  async function loadProbabilityHistory(mode) {
    const requestId = ++historyRequestId;
    try {
      const response = await fetch(`/api/classification-history?mode=${encodeURIComponent(mode)}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `Error HTTP ${response.status}`);
      if (requestId !== historyRequestId || data.mode !== probabilityMode) return;

      const buckets = new Map();
      (data.samples || []).forEach((sample) => {
        const timestamp = Number(sample.timestamp_ms);
        if (!Number.isFinite(timestamp)) return;
        const bucket = Math.floor(timestamp / PROBABILITY_SAMPLE_MS) * PROBABILITY_SAMPLE_MS;
        const existing = buckets.get(bucket) || {};
        Object.entries(sample.values || {}).forEach(([key, value]) => {
          if (Object.prototype.hasOwnProperty.call(probabilityHistory, key)) {
            existing[key] = Number(value);
          }
        });
        buckets.set(bucket, existing);
      });

      resetProbabilityHistory();
      [...buckets.entries()].sort(([left], [right]) => left - right).forEach(([timestamp, values]) => {
        Object.keys(PROBABILITY_SERIES).forEach((key) => {
          const value = values[key];
          if (Number.isFinite(value)) {
            probabilityHistory[key].push({
              timestamp,
              value: Math.max(0, Math.min(100, value)),
            });
          }
        });
      });
      const timestamps = [...buckets.keys()];
      if (timestamps.length) {
        lastProbabilitySampleBucket = Math.floor(Math.max(...timestamps) / PROBABILITY_SAMPLE_MS);
      }
      renderProbabilitySparklines();
    } catch (error) {
      console.error('Error cargando el historial de clasificación:', error);
    }
  }

  // Start WebSocket
  connectWebSocket();
});
