// Generative audio: detuned drones through a breathing lowpass, sparse
// pentatonic plucks through a feedback delay. No samples, no loops, no repeats.
// ponytail: audio follows interaction state (cursor speed, hover, clicks), not a
// GPU readback of the sim; wire a trail-density reduction pass in if you want
// literal coupling.

const PENTA = [220.0, 261.63, 293.66, 349.23, 392.0, 440.0, 523.25, 587.33];

export class AudioEngine {
  constructor() {
    this.ctx = null;
    this.on = false;
  }

  start() {
    if (this.ctx) return;
    const ctx = this.ctx = new AudioContext();
    this.on = true;

    this.master = ctx.createGain();
    this.master.gain.value = 0;
    this.master.gain.setTargetAtTime(0.6, ctx.currentTime, 3); // slow fade-in
    const comp = ctx.createDynamicsCompressor();
    this.master.connect(comp).connect(ctx.destination);

    // feedback delay bus for the plucks
    this.delay = ctx.createDelay(1);
    this.delay.delayTime.value = 0.42;
    const fb = ctx.createGain();
    fb.gain.value = 0.38;
    const wet = ctx.createGain();
    wet.gain.value = 0.5;
    this.delay.connect(fb).connect(this.delay);
    this.delay.connect(wet).connect(this.master);

    // drone: two detuned saws + a sine an octave up, through a breathing lowpass
    this.filter = ctx.createBiquadFilter();
    this.filter.type = 'lowpass';
    this.filter.frequency.value = 200;
    this.filter.Q.value = 1.2;
    const droneGain = ctx.createGain();
    droneGain.gain.value = 0.05;
    this.filter.connect(droneGain).connect(this.master);

    for (const [type, freq, det] of [['sawtooth', 55, 0], ['sawtooth', 55, 7], ['sine', 110.3, 0]]) {
      const o = ctx.createOscillator();
      o.type = type;
      o.frequency.value = freq;
      o.detune.value = det;
      o.connect(this.filter);
      o.start();
    }

    // slow LFO breathing on the filter cutoff
    const lfo = ctx.createOscillator();
    lfo.frequency.value = 0.05;
    const lfoGain = ctx.createGain();
    lfoGain.gain.value = 90;
    lfo.connect(lfoGain).connect(this.filter.frequency);
    lfo.start();

    this._scheduleWander();
  }

  // random sparse plucks, forever
  _scheduleWander() {
    if (!this.ctx) return;
    this._pluck(PENTA[Math.floor(Math.random() * PENTA.length)], 0.03 + Math.random() * 0.03);
    setTimeout(() => this._scheduleWander(), 2500 + Math.random() * 5500);
  }

  _pluck(freq, vol = 0.05, when = 0) {
    if (!this.ctx || !this.on) return;
    const ctx = this.ctx;
    const t = ctx.currentTime + when;
    const o = ctx.createOscillator();
    o.type = 'sine';
    o.frequency.value = freq * (1 + (Math.random() - 0.5) * 0.004);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(vol, t + 0.015);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 2.2 + Math.random());
    o.connect(g);
    g.connect(this.master);
    g.connect(this.delay);
    o.start(t);
    o.stop(t + 4);
  }

  // cursor speed opens the filter: the organism gets excited
  setActivity(x) {
    if (!this.ctx || !this.on) return;
    this.filter.frequency.setTargetAtTime(200 + x * 1100, this.ctx.currentTime, 0.25);
  }

  hover() {
    this._pluck(PENTA[2 + Math.floor(Math.random() * 4)], 0.045);
  }

  chord() {
    const root = Math.floor(Math.random() * 3);
    [root, root + 2, root + 4].forEach((n, i) => this._pluck(PENTA[n], 0.05, i * 0.13));
  }

  toggle() {
    if (!this.ctx) return false;
    this.on = !this.on;
    this.master.gain.setTargetAtTime(this.on ? 0.6 : 0, this.ctx.currentTime, 0.3);
    return this.on;
  }
}
