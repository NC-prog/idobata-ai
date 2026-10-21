// マイク取り込み（AudioWorklet → PCM16 16kHz）と、応答音声の再生キュー（口パク付き）。

import type { MouthKey, Viseme } from "./protocol";

const MIC_RATE = 16000;
// 再生開始の余裕（デコード直後に start すると頭が欠けることがある）
const START_LEAD_SEC = 0.03;

export interface MicInfo {
  aec: boolean;
  sampleRate: number;
}

/** getUserMedia（AEC・ノイズ抑制・AGC あり）で取り、20ms ごとに PCM16 を渡す。 */
export async function startMic(onChunk: (pcm: ArrayBuffer) => void): Promise<MicInfo> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
  });
  const track = stream.getAudioTracks()[0];
  const settings = track.getSettings();
  // 16kHz の AudioContext を作れればブラウザ側で高品質にリサンプルされる
  const ctx = new AudioContext({ sampleRate: MIC_RATE });
  await ctx.audioWorklet.addModule(`${import.meta.env.BASE_URL}mic-processor.js`);
  const source = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, "mic-processor");
  node.port.onmessage = (ev: MessageEvent<ArrayBuffer>) => onChunk(ev.data);
  source.connect(node);
  // 出力に繋がないと process() が呼ばれない実装があるので、無音のまま繋いでおく
  const sink = ctx.createGain();
  sink.gain.value = 0;
  node.connect(sink).connect(ctx.destination);
  await ctx.resume();
  return { aec: settings.echoCancellation === true, sampleRate: ctx.sampleRate };
}

interface Scheduled {
  utteranceId: string;
  seq: number;
  text: string;
  start: number;
  end: number;
  mouth: MouthKey[];
}

export interface SpeechSegment {
  utteranceId: string;
  seq: number;
  text: string;
  mouth: MouthKey[];
}

/**
 * 1 文ずつ届く WAV を隙間なく続けて鳴らす。キューが空になったら onIdle を呼ぶ。
 * デコードは届いた順に直列で行い、順序を入れ替えない。
 */
export class SpeechPlayer {
  readonly ctx: AudioContext;
  private nextStart = 0;
  private scheduled: Scheduled[] = [];
  private decoding: Promise<void> = Promise.resolve();
  private pendingDecodes = 0;
  private readonly onIdle: (utteranceId: string, lastSeq: number) => void;

  constructor(ctx: AudioContext, onIdle: (utteranceId: string, lastSeq: number) => void) {
    this.ctx = ctx;
    this.onIdle = onIdle;
  }

  enqueue(seg: SpeechSegment, wav: ArrayBuffer): void {
    this.pendingDecodes++;
    this.decoding = this.decoding
      .then(async () => {
        const buffer = await this.ctx.decodeAudioData(wav);
        this.schedule(seg, buffer);
      })
      .catch((err: unknown) => {
        console.warn("応答音声をデコードできません", err);
      })
      .finally(() => {
        this.pendingDecodes--;
        this.checkIdle(seg.utteranceId, seg.seq);
      });
  }

  private schedule(seg: SpeechSegment, buffer: AudioBuffer): void {
    const src = this.ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(this.ctx.destination);
    const start = Math.max(this.ctx.currentTime + START_LEAD_SEC, this.nextStart);
    src.start(start);
    this.nextStart = start + buffer.duration;
    const item: Scheduled = { ...seg, start, end: this.nextStart };
    this.scheduled.push(item);
    src.onended = () => {
      this.scheduled = this.scheduled.filter((s) => s !== item);
      this.checkIdle(item.utteranceId, item.seq);
    };
  }

  private checkIdle(utteranceId: string, seq: number): void {
    if (this.pendingDecodes === 0 && this.scheduled.length === 0) {
      this.onIdle(utteranceId, seq);
    }
  }

  /** いま鳴っている口形と、その区間内の位置（0〜1）。鳴っていなければ null。 */
  currentViseme(): { v: Viseme; phase: number } | null {
    const now = this.ctx.currentTime;
    for (const s of this.scheduled) {
      if (now < s.start || now >= s.end) continue;
      const t = now - s.start;
      for (const k of s.mouth) {
        if (t >= k.t && t < k.t + k.d) return { v: k.v, phase: (t - k.t) / k.d };
        if (k.t > t) break;
      }
      return null;
    }
    return null;
  }

  /** いま鳴っている文（字幕を音声に合わせて出すのに使う）。鳴っていなければ null。 */
  currentSegment(): SpeechSegment | null {
    const now = this.ctx.currentTime;
    return this.scheduled.find((s) => now >= s.start && now < s.end) ?? null;
  }

  get isPlaying(): boolean {
    const now = this.ctx.currentTime;
    return this.scheduled.some((s) => now >= s.start && now < s.end);
  }
}
