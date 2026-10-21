// AudioWorklet: マイク入力を 16kHz mono PCM16 にして 20ms（320 サンプル）ごとにメインスレッドへ渡す。
// AudioContext を 16kHz で作れていれば変換なし。そうでなければ区間平均で間引く。
const TARGET_RATE = 16000;
const CHUNK = 320;

class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / TARGET_RATE;
    this.acc = 0;
    this.accCount = 0;
    this.pos = 0;
    this.out = new Int16Array(CHUNK);
    this.filled = 0;
  }

  push(sample) {
    const s = Math.max(-1, Math.min(1, sample));
    this.out[this.filled++] = s < 0 ? s * 0x8000 : s * 0x7fff;
    if (this.filled === CHUNK) {
      this.port.postMessage(this.out.buffer, [this.out.buffer]);
      this.out = new Int16Array(CHUNK);
      this.filled = 0;
    }
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    if (this.ratio === 1) {
      for (let i = 0; i < ch.length; i++) this.push(ch[i]);
      return true;
    }
    for (let i = 0; i < ch.length; i++) {
      this.acc += ch[i];
      this.accCount++;
      this.pos += 1;
      if (this.pos >= this.ratio) {
        this.pos -= this.ratio;
        this.push(this.acc / this.accCount);
        this.acc = 0;
        this.accCount = 0;
      }
    }
    return true;
  }
}

registerProcessor("mic-processor", MicProcessor);
