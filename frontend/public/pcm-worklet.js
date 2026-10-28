export class PcmResampler {
  constructor(rate, target = 16000) {
    if (rate < target) throw new Error('Unsupported sample rate');
    this.width = rate / target; this.remaining = this.width; this.sum = 0;
  }
  push(input, emit) {
    for (const sample of input) {
      let left = 1;
      while (left > 1e-8) {
        const used = Math.min(left, this.remaining);
        this.sum += sample * used; this.remaining -= used; left -= used;
        if (this.remaining < 1e-8) {
          emit(Math.max(-1, Math.min(1, this.sum / this.width)));
          this.remaining = this.width; this.sum = 0;
        }
      }
    }
  }
}
if (typeof registerProcessor !== 'undefined') {
  class Capture extends AudioWorkletProcessor {
    constructor() {
      super(); this.resampler = new PcmResampler(sampleRate); this.frame = new ArrayBuffer(16000);
      this.view = new DataView(this.frame); this.count = 0; this.stopped = false;
      this.port.onmessage = e => { if (e.data === 'stop') { this.stopped = true; this.flush(); this.port.postMessage('flushed'); } };
    }
    flush() {
      if (!this.count) return;
      const frame = this.frame.slice(0, this.count * 2); this.port.postMessage(frame, [frame]); this.count = 0;
    }
    process(inputs) {
      if (this.stopped) return false;
      const channels = inputs[0];
      if (channels?.length) {
        const mono = new Float32Array(channels[0].length);
        for (const channel of channels) for (let i=0;i<mono.length;i++) mono[i] += channel[i]/channels.length;
        this.resampler.push(mono, sample => {
          this.view.setInt16(this.count*2, Math.round(sample*(sample<0?32768:32767)), true);
          if (++this.count === 8000) this.flush();
        });
      }
      return true;
    }
  }
  registerProcessor('pcm-capture', Capture);
}
