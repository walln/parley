"use strict";

const JITTER = 2;
const MAX_QUEUE = 25;

class PlaybackProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.offset = 0;
    this.started = false;
    this.draining = false;
    this.underruns = 0;
    this.port.onmessage = (event) => {
      const msg = event.data;
      if (msg.type === "pcm") {
        if (this.queue.length >= MAX_QUEUE) {
          this.port.postMessage({ type: "overflow" });
        } else {
          this.queue.push(new Int16Array(msg.buffer));
        }
      } else if (msg.type === "drain") {
        this.draining = true;
      }
    };
  }

  process(_inputs, outputs) {
    const out = outputs[0][0];
    let drained = false;
    for (let i = 0; i < out.length; i += 1) {
      if (!this.started) {
        if (this.queue.length >= JITTER || (this.draining && this.queue.length)) {
          this.started = true;
        } else {
          out[i] = 0;
          if (this.draining && this.queue.length === 0) {
            drained = true;
          }
          continue;
        }
      }
      if (!this.queue.length) {
        out[i] = 0;
        this.underruns += 1;
        if (this.draining) {
          drained = true;
        }
        continue;
      }
      const head = this.queue[0];
      out[i] = head[this.offset] / 32768;
      this.offset += 1;
      if (this.offset === head.length) {
        this.queue.shift();
        this.offset = 0;
      }
    }
    if (drained) {
      this.port.postMessage({ type: "drained", underruns: this.underruns });
      this.draining = false;
      this.started = false;
    }
    return true;
  }
}

registerProcessor("playback", PlaybackProcessor);
