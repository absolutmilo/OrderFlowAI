/**
 * Audio Sonification Engine for Order Flow.
 * Synthesizes procedural audio for buys, sells, institutional icebergs, and absorptions
 * using the browser's Web Audio API with stereo panning and dynamic harmonic modulation.
 */

class OrderFlowAudioEngine {
  constructor() {
    this.audioCtx = null;
    this.isEnabled = false;
    this.masterGain = null;
    this.volume = 0.6;
    this.minUsdThreshold = 2500.0; // Responsive threshold so sounds trigger immediately
  }

  init() {
    if (!this.audioCtx) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      this.audioCtx = new AudioContextClass();

      // Master gain node
      this.masterGain = this.audioCtx.createGain();
      this.masterGain.gain.setValueAtTime(this.volume, this.audioCtx.currentTime);
      this.masterGain.connect(this.audioCtx.destination);
    }

    if (this.audioCtx.state === 'suspended') {
      this.audioCtx.resume();
    }
    this.isEnabled = true;
  }

  toggle() {
    if (!this.isEnabled) {
      this.init();
      return true;
    } else {
      this.isEnabled = false;
      return false;
    }
  }

  setVolume(val) {
    this.volume = Math.max(0, Math.min(1.0, val));
    if (this.masterGain && this.audioCtx) {
      this.masterGain.gain.setValueAtTime(this.volume, this.audioCtx.currentTime);
    }
  }

  setThreshold(usdVal) {
    this.minUsdThreshold = Math.max(0, usdVal);
  }

  testChime() {
    this.init();
    const now = this.audioCtx.currentTime;
    // Play a 3-tone demo: Buy ping -> Sell ping -> Whale bass
    this.playTradeSound({ side: 'BUY', usd_value: 30000 }, []);
    setTimeout(() => {
      this.playTradeSound({ side: 'SELL', usd_value: 30000 }, []);
    }, 200);
    setTimeout(() => {
      this.playTradeSound({ side: 'BUY', usd_value: 250000 }, []);
    }, 450);
  }

  playTradeSound(trade, patterns = []) {
    if (!this.isEnabled || !this.audioCtx) return;
    if (trade.usd_value < this.minUsdThreshold) return;

    const ctx = this.audioCtx;
    const now = ctx.currentTime;
    const isBuy = trade.side === 'BUY';
    const usd = trade.usd_value;

    // Amplitude scaling by volume (logarithmic scaling)
    const normalizedVol = Math.min(1.0, Math.log10(Math.max(100, usd) / 500.0) / 3.0);
    const soundGain = ctx.createGain();
    soundGain.gain.setValueAtTime(Math.max(0.08, normalizedVol * 0.45), now);
    soundGain.gain.exponentialRampToValueAtTime(0.0001, now + (usd > 200000 ? 0.35 : 0.09));

    // Stereo Panner (Buys on the right +0.35, Sells on the left -0.35)
    if (ctx.createStereoPanner) {
      const panner = ctx.createStereoPanner();
      panner.pan.setValueAtTime(isBuy ? 0.35 : -0.35, now);
      soundGain.connect(panner);
      panner.connect(this.masterGain);
    } else {
      soundGain.connect(this.masterGain);
    }

    // 1. Institutional Patterns
    if (patterns.some(p => p.includes('ICEBERG'))) {
      this._playIcebergTone(ctx, soundGain, now, isBuy);
      return;
    }

    if (patterns.some(p => p.includes('ABSORPTION'))) {
      this._playAbsorptionTone(ctx, soundGain, now, isBuy);
      return;
    }

    // 2. Whale Block (> $150k) -> Deep Sub-bass Impact
    if (usd >= 150000) {
      this._playWhaleImpact(ctx, soundGain, now, isBuy);
      return;
    }

    // 3. Regular Aggressive Order Flow Click/Tone
    const baseFreq = isBuy ? 660 : 330;
    const osc = ctx.createOscillator();
    osc.type = isBuy ? 'triangle' : 'sine';
    osc.frequency.setValueAtTime(baseFreq, now);
    osc.frequency.exponentialRampToValueAtTime(baseFreq * (isBuy ? 1.25 : 0.8), now + 0.07);

    osc.connect(soundGain);
    osc.start(now);
    osc.stop(now + 0.08);
  }

  _playIcebergTone(ctx, parentGain, now, isBuy) {
    const osc1 = ctx.createOscillator();
    const osc2 = ctx.createOscillator();
    osc1.type = 'sine';
    osc2.type = 'triangle';

    const freq1 = isBuy ? 784 : 440;
    const freq2 = isBuy ? 1046 : 587;

    osc1.frequency.setValueAtTime(freq1, now);
    osc2.frequency.setValueAtTime(freq2, now + 0.04);

    osc1.connect(parentGain);
    osc2.connect(parentGain);

    osc1.start(now);
    osc2.start(now + 0.04);
    osc1.stop(now + 0.16);
    osc2.stop(now + 0.20);
  }

  _playAbsorptionTone(ctx, parentGain, now, isBuy) {
    const osc = ctx.createOscillator();
    const filter = ctx.createBiquadFilter();

    osc.type = 'sawtooth';
    osc.frequency.setValueAtTime(isBuy ? 261.63 : 196.0, now);

    filter.type = 'lowpass';
    filter.frequency.setValueAtTime(550, now);
    filter.frequency.exponentialRampToValueAtTime(100, now + 0.25);

    osc.connect(filter);
    filter.connect(parentGain);

    osc.start(now);
    osc.stop(now + 0.26);
  }

  _playWhaleImpact(ctx, parentGain, now, isBuy) {
    const subOsc = ctx.createOscillator();
    subOsc.type = 'sine';
    subOsc.frequency.setValueAtTime(110, now);
    subOsc.frequency.exponentialRampToValueAtTime(45, now + 0.3);

    subOsc.connect(parentGain);
    subOsc.start(now);
    subOsc.stop(now + 0.32);
  }
}

window.audioEngine = new OrderFlowAudioEngine();
