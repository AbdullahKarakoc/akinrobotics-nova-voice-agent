// Captures microphone audio, resamples it to 16 kHz mono and posts int16 PCM
// blocks (~64 ms) plus an RMS level to the main thread.
//
// Downsampling needs a proper anti-aliasing low-pass first: without it, energy
// above 8 kHz (fricatives like s, z, ş, f) folds back into the speech band and
// smears consonants, which hurts speech recognition. A 63-tap windowed-sinc
// (Blackman) FIR with a 7 kHz cutoff costs ~3 M multiply-adds per second.
const TAPS = 63;
const CUTOFF_HZ = 7000;

function designLowPass(sampleRate) {
  const fc = CUTOFF_HZ / sampleRate;
  const m = (TAPS - 1) / 2;
  const h = new Float32Array(TAPS);
  let sum = 0;
  for (let n = 0; n < TAPS; n++) {
    const x = n - m;
    const sinc = x === 0 ? 2 * fc : Math.sin(2 * Math.PI * fc * x) / (Math.PI * x);
    const blackman = 0.42 - 0.5 * Math.cos((2 * Math.PI * n) / (TAPS - 1)) + 0.08 * Math.cos((4 * Math.PI * n) / (TAPS - 1));
    h[n] = sinc * blackman;
    sum += h[n];
  }
  for (let n = 0; n < TAPS; n++) h[n] /= sum; // unity gain at DC
  return h;
}

class PcmCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetRate = 16000;
    this.step = sampleRate / this.targetRate; // input samples per output sample
    this.pos = 0;
    this.taps = designLowPass(sampleRate);
    this.history = new Float32Array(TAPS); // ring buffer of the last TAPS input samples
    this.head = 0;
    this.block = new Int16Array(1024);
    this.n = 0;
    this.energy = 0;
  }

  filter(x) {
    this.history[this.head] = x;
    this.head = (this.head + 1) % TAPS;
    let y = 0;
    for (let k = 0, i = this.head; k < TAPS; k++, i = (i + 1) % TAPS) y += this.taps[k] * this.history[i];
    return y;
  }

  process(inputs) {
    const input = inputs[0] && inputs[0][0];
    if (!input) return true;
    const filtered = new Float32Array(input.length);
    for (let i = 0; i < input.length; i++) filtered[i] = this.filter(input[i]);
    // Linear interpolation on the band-limited signal; `pos` is the fractional read position within the block.
    while (this.pos < filtered.length) {
      const i = Math.floor(this.pos);
      const frac = this.pos - i;
      const a = filtered[i];
      const b = i + 1 < filtered.length ? filtered[i + 1] : a;
      const s = Math.max(-1, Math.min(1, a + (b - a) * frac));
      this.block[this.n++] = s * 32767;
      this.energy += s * s;
      if (this.n === this.block.length) {
        const rms = Math.sqrt(this.energy / this.n);
        this.port.postMessage({ pcm: this.block.buffer, rms }, [this.block.buffer]);
        this.block = new Int16Array(1024);
        this.n = 0;
        this.energy = 0;
      }
      this.pos += this.step;
    }
    this.pos -= filtered.length;
    return true;
  }
}

registerProcessor("pcm-capture", PcmCapture);
