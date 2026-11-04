/**
 * Main Frontend Application Script.
 * Manages WebSocket connection, live UI updates, probability gauges, and tape rendering.
 */

document.addEventListener('DOMContentLoaded', () => {
  // UI Elements
  const elBtcPrice = document.getElementById('btc-price');
  const elCvdUsd = document.getElementById('cvd-usd');
  const elTps = document.getElementById('trades-per-sec');
  const elConnectionStatus = document.getElementById('connection-status');
  const elStatusText = document.getElementById('status-text');
  const elTapeList = document.getElementById('tape-list');
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
  const elBarAlgo = document.getElementById('bar-algo');

  const elProbIceberg = document.getElementById('prob-iceberg');
  const elBarIceberg = document.getElementById('bar-iceberg');
  const elDescIceberg = document.getElementById('desc-iceberg');

  const elProbAbsorption = document.getElementById('prob-absorption');
  const elBarAbsorption = document.getElementById('bar-absorption');
  const elDescAbsorption = document.getElementById('desc-absorption');

  const elProbRetail = document.getElementById('prob-retail');
  const elBarRetail = document.getElementById('bar-retail');

  const elProbHft = document.getElementById('prob-hft');
  const elBarHft = document.getElementById('bar-hft');

  // Audio Controls
  const btnToggleAudio = document.getElementById('btn-toggle-audio');
  const btnTestAudio = document.getElementById('btn-test-audio');
  const sliderVolume = document.getElementById('slider-volume');
  const selectThreshold = document.getElementById('select-threshold');
  const btnScalpBacktest = document.getElementById('btn-scalp-backtest');
  const selectScalpStrategy = document.getElementById('scalp-strategy');
  const selectScalpTimeframe = document.getElementById('scalp-timeframe');
  const selectScalpDirection = document.getElementById('scalp-direction');

  const MAX_TAPE_ROWS = 120;
  let socket = null;
  let reconnectTimer = null;
  let modeStatusTimer = null;

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
      dataStatusElement.textContent = `Histórico ${scalping.source}: ${Number(scalping.observations).toLocaleString('es-ES')} / ${Number(required).toLocaleString('es-ES')} snapshots totales (${Number(scalping.minimum_training_observations || required).toLocaleString('es-ES')} para entrenamiento).`;
    }
  }

  function renderScalpingBacktest(backtest) {
    const resultElement = document.getElementById('scalp-backtest-result');
    const metricsElement = document.getElementById('scalp-backtest-metrics');
    const breakdownElement = document.getElementById('scalp-backtest-breakdown');
    const exitsElement = document.getElementById('scalp-backtest-exits');
    if (!resultElement || !metricsElement || !backtest) return;

    resultElement.textContent = backtest.message || `Resultado: ${backtest.status}`;
    if (backtest.validation) {
      const validation = backtest.validation;
      const training = backtest.training;
      const parameters = backtest.parameters;
      metricsElement.textContent = `Estrategia: ${backtest.strategy_name} · ${backtest.timeframe_minutes} min · dirección ${backtest.direction_mode_name || 'Original'}. Entrenamiento (${training.trade_count} ops): bruto ${training.gross_expectancy_bps} bps/operación, neto ${training.expectancy_bps} bps/operación. Validación (${validation.trade_count} ops): bruto ${validation.gross_expectancy_bps} bps/operación, neto ${validation.expectancy_bps} bps/operación · aciertos netos ${validation.win_rate_pct}% · PF ${validation.profit_factor} · DD ${validation.max_drawdown_bps} bps · costos ${validation.total_costs_bps} bps. SL ${parameters.stop_atr_multiple} ATR, TP ${parameters.target_atr_multiple} ATR.`;
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
        metricsElement.textContent = `${backtest.strategy_name} · ${backtest.timeframe_minutes} min · dirección ${backtest.direction_mode_name || 'Original'}. ${bars} velas de entrenamiento${requiredBars ? ` (${requiredBars} requeridas)` : ''} · ${Number(backtest.training_signal_count || 0)} señales · máximo ${Number(backtest.max_training_trades || 0)} operaciones simuladas; se requieren ${Number(backtest.minimum_training_trades || 0)}.`;
        if (breakdownElement) {
          breakdownElement.textContent = diagnostics.complete_bars !== undefined
            ? `Diagnóstico Fourier: ${diagnostics.raw_anomalies} anomalías · ${diagnostics.persistent_anomaly_bars} barras anómalas persistentes · ${diagnostics.completed_zones} zonas cerradas · ${diagnostics.context_qualified_bars} barras con contexto · ${diagnostics.breakout_candidates} rupturas.`
            : '';
        }
      } else {
        metricsElement.textContent = backtest.observations !== undefined
          ? `${backtest.strategy_name || ''} · ${backtest.timeframe_minutes || ''} min. ${Number(backtest.observations).toLocaleString('es-ES')} observaciones Bybit Linear; ${Number(backtest.training_observations || 0).toLocaleString('es-ES')} snapshots en entrenamiento${backtest.training_bars !== undefined ? `; ${backtest.training_bars}/${backtest.required_training_bars || '?'} velas completas` : ''}${backtest.required_training_observations !== undefined ? `; se requieren ${Number(backtest.required_training_observations).toLocaleString('es-ES')} para entrenar y ${Number(backtest.required_observations || 0).toLocaleString('es-ES')} en total` : ''}.`
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
        btnScalpBacktest.textContent = 'Optimizar y validar backtest';
      }
    });
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

        if (data.type === 'order_flow_tick') {
          handleTradeTick(data.trade);
          updateProbabilities(data.probabilities);
          if (data.metrics) updateMetrics(data.metrics);
          if (data.exchange_stats) updateExchangeStats(data.exchange_stats);
          updateSystemMode(data.inference_mode, data.data_capture_status, data.detector_config);
          renderScalpingSignal(data.scalping);
        } else if (data.type === 'snapshot') {
          updateProbabilities(data.probabilities);
          if (data.metrics) updateMetrics(data.metrics);
          if (data.exchange_stats) updateExchangeStats(data.exchange_stats);
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

  function updateExchangeStats(stats) {
    if (!stats || !elFeedStats) return;
    elFeedStats.textContent = `BINANCE: ${stats.Binance || 0} | BYBIT: ${stats.Bybit || 0}`;
  }

  function updateSystemMode(mode, captureStatus, detectorConfig) {
    if (elInferenceMode && mode) {
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
    elBtcPrice.textContent = `$${metrics.current_price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

    const cvdVal = metrics.cvd_usd;
    const cvdSign = cvdVal >= 0 ? '+' : '';
    elCvdUsd.textContent = `${cvdSign}$${(cvdVal / 1000).toFixed(1)}k`;
    elCvdUsd.style.color = cvdVal >= 0 ? '#00e676' : '#ff1744';

    elTps.textContent = `${metrics.trades_per_second.toFixed(1)} /s`;
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
    elBarAlgo.style.width = `${probs.algo_institutional}%`;

    // 2. Iceberg
    elProbIceberg.textContent = `${probs.iceberg}%`;
    elBarIceberg.style.width = `${probs.iceberg}%`;
    if (probs.iceberg_side) {
      elDescIceberg.textContent = `Iceberg ${probs.iceberg_side} activo en nivel modal`;
    } else {
      elDescIceberg.textContent = `Acumulación oculta en nivel de precio`;
    }

    // 3. Absorción
    elProbAbsorption.textContent = `${probs.absorption}%`;
    elBarAbsorption.style.width = `${probs.absorption}%`;
    if (probs.absorption_side) {
      elDescAbsorption.textContent = probs.absorption_side;
    } else {
      elDescAbsorption.textContent = `Órdenes límite absorbiendo presión agresiva`;
    }

    // 4. Retail
    elProbRetail.textContent = `${probs.retail}%`;
    elBarRetail.style.width = `${probs.retail}%`;

    // 5. HFT
    elProbHft.textContent = `${probs.hft_competition}%`;
    elBarHft.style.width = `${probs.hft_competition}%`;

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

    // Create row in visual tape
    const isBuy = trade.side === 'BUY';
    const row = document.createElement('div');
    const isWhale = trade.usd_value >= 100000;

    row.className = `tape-row ${isBuy ? 'buy' : 'sell'} ${isWhale ? 'whale' : ''}`;

    const date = new Date(trade.timestamp_ms);
    const timeStr = date.toTimeString().split(' ')[0] + '.' + String(date.getMilliseconds()).padStart(3, '0').slice(0, 2);

    let patternBadge = '';
    if (trade.patterns && trade.patterns.length > 0) {
      const p = trade.patterns[0];
      const tagClass = p.includes('ALGO') ? 'tag-algo' :
                       p.includes('ICEBERG') ? 'tag-iceberg' :
                       p.includes('ABSORPTION') ? 'tag-absorption' : 'tag-hft';
      patternBadge = `<span class="badge-tag ${tagClass}">${p}</span>`;
    }

    const exchShort = trade.exchange ? (trade.exchange.includes('Binance') ? 'BN' : 'BYB') : '--';

    row.innerHTML = `
      <span>${timeStr}</span>
      <span style="font-size:0.75rem; color:#8b949e;">${exchShort}</span>
      <span>$${trade.price.toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 })}</span>
      <span>${trade.size.toFixed(3)} BTC</span>
      <span>$${Math.round(trade.usd_value).toLocaleString('en-US')}</span>
      <span>${patternBadge || trade.side}</span>
    `;

    elTapeList.insertBefore(row, elTapeList.firstChild);

    // Limit DOM count for smooth 60fps rendering
    while (elTapeList.children.length > MAX_TAPE_ROWS) {
      elTapeList.removeChild(elTapeList.lastChild);
    }
  }

  // Start WebSocket
  connectWebSocket();
});
