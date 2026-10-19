class EvaAffectAudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frameLimit = Math.max(1, Math.round(sampleRate * 0.5));
    this.frames = [];
    this.frameCount = 0;
    this.port.onmessage = (event) => {
      if (event.data?.type === 'flush') {
        this.flushFrames();
        this.port.postMessage({ type: 'flushed' });
      }
    };
  }

  flushFrames() {
    if (!this.frameCount) return;
    const samples = new Float32Array(this.frameCount);
    let offset = 0;
    for (const frame of this.frames) {
      samples.set(frame, offset);
      offset += frame.length;
    }
    this.frames = [];
    this.frameCount = 0;
    this.port.postMessage({ type: 'audio', samples }, [samples.buffer]);
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (channel?.length) {
      this.frames.push(new Float32Array(channel));
      this.frameCount += channel.length;
      if (this.frameCount >= this.frameLimit) this.flushFrames();
    }
    return true;
  }
}

registerProcessor('eva-affect-audio', EvaAffectAudioProcessor);
