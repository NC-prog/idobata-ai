// フロントAI への WebSocket（1 本で JSON とバイナリを混在させる）。切れたら張り直す。

import type { DownMessage, UpMessage } from "./protocol";

/** 別のエッジがつながったのでフロントが閉じた（server/src/edge_bridge.py の REPLACED_CLOSE_CODE） */
export const REPLACED_CLOSE_CODE = 4001;

export interface LinkHandlers {
  onOpen: () => void;
  onClose: (replaced: boolean) => void;
  onMessage: (msg: DownMessage) => void;
  onBinary: (data: ArrayBuffer) => void;
}

const RETRY_MIN_MS = 500;
const RETRY_MAX_MS = 5000;

export class FrontLink {
  private ws: WebSocket | null = null;
  private retryMs = RETRY_MIN_MS;
  private readonly url: string;
  private readonly handlers: LinkHandlers;

  constructor(url: string, handlers: LinkHandlers) {
    this.url = url;
    this.handlers = handlers;
  }

  get isOpen(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  connect(): void {
    const ws = new WebSocket(this.url);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      this.retryMs = RETRY_MIN_MS;
      this.handlers.onOpen();
    };
    ws.onclose = (ev: CloseEvent) => {
      this.ws = null;
      const replaced = ev.code === REPLACED_CLOSE_CODE;
      this.handlers.onClose(replaced);
      if (replaced) return; // 取り合いにしない（ボタンで戻す）
      window.setTimeout(() => this.connect(), this.retryMs);
      this.retryMs = Math.min(this.retryMs * 2, RETRY_MAX_MS);
    };
    ws.onmessage = (ev: MessageEvent<string | ArrayBuffer>) => {
      if (typeof ev.data === "string") {
        try {
          this.handlers.onMessage(JSON.parse(ev.data) as DownMessage);
        } catch (err) {
          console.warn("フロントからの JSON を解釈できません", err);
        }
      } else {
        this.handlers.onBinary(ev.data);
      }
    };
    this.ws = ws;
  }

  send(msg: UpMessage): void {
    if (this.isOpen) this.ws!.send(JSON.stringify(msg));
  }

  sendBinary(data: ArrayBuffer): void {
    // 送信が詰まっている（フロントが重い）ときは古い音声を溜めずに捨てる
    if (this.isOpen && this.ws!.bufferedAmount < 64 * 1024) this.ws!.send(data);
  }
}
