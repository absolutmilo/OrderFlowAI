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

  const MAX_TAPE_ROWS = 120;
  let socket = null;
  let reconnectTimer = null;

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
        } else if (data.type === 'snapshot') {
          updateProbabilities(data.probabilities);
          if (data.metrics) updateMetrics(data.metrics);
          if (data.exchange_stats) updateExchangeStats(data.exchange_stats);
          updateSystemMode(data.inference_mode, data.data_capture_status, data.detector_config);
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
    if (elInferenceMode) {
      const modeNames = { heuristic: 'HEURÍSTICO', model: 'MODELO' };
      elInferenceMode.textContent = `MODO: ${modeNames[mode] || 'HEURÍSTICO'}`;
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
