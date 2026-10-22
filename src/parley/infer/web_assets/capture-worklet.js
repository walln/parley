"use strict";

const FRAME = 1280;

class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.pending = new Float32Array(FRAME * 2);
    this.count = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (input && input.length && input[0].length) {
      const channels = input.length;
      const length = input[0].length;
      for (let i = 0; i < length; i += 1) {
        let sample = 0;
        for (let ch = 0; ch < channels; ch += 1) {
          sample += input[ch][i];
        }
        this.pending[this.count] = sample / channels;
        this.count += 1;
        if (this.count === FRAME) {
          const pcm = new Int16Array(FRAME);
          for (let j = 0; j < FRAME; j += 1) {
            const v = Math.round(this.pending[j] * 32768);
            pcm[j] = v < -32768 ? -32768 : v > 32767 ? 32767 : v;
          }
          this.port.postMessage(pcm.buffer, [pcm.buffer]);
          this.count = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor("capture", CaptureProcessor);
