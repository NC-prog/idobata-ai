// エッジ ⇄ フロントAI の契約（docs/WP01_edge_and_avatar.md B-1、server/src/edge_bridge.py）

export type AvatarState = "idle" | "listening" | "thinking" | "speaking";

/** 口形 1 区間。t は WAV 先頭からの秒、d は長さ（秒）。v は VRM の口形プリセット名 */
export interface MouthKey {
  t: number;
  v: Viseme;
  d: number;
}

export type Viseme = "aa" | "ih" | "ou" | "ee" | "oh";

export interface StateMessage {
  type: "state";
  value: AvatarState;
}

export interface CaptionMessage {
  type: "caption";
  role: "user" | "assistant";
  text: string;
  final: boolean;
}

/** 直後のバイナリフレームがこの文の WAV */
export interface SpeakMessage {
  type: "speak";
  utterance_id: string;
  seq: number;
  text: string;
  mouth: MouthKey[];
}

/** アバターと背景の切り替え（WP08）。avatar・background はエッジのページ（/edge/）からの相対パス */
export interface ThemeMessage {
  type: "theme";
  site: string;
  season: string;
  name?: string;
  avatar?: string;
  background?: string;
}

/** 応答が使った参考資料の出典タイトル（字幕の下に薄く出す。読み上げない。WP03） */
export interface SourcesMessage {
  type: "sources";
  titles: string[];
}

/** カメラの取り方（hello の後にフロントから届く。WP02） */
export interface CameraMessage {
  type: "camera";
  enabled: boolean;
  /** 複数のカメラがあるとき、label にこの文字を含むものを使う */
  label: string;
  capture: { width: number; height: number; fps: number };
  send: { width: number; height: number; fps: number; quality: number };
}

export type PartyState = "NONE" | "PRESENT" | "PARTY";

/** 目の前の輪（アバターの視線・姿勢に使う。WP02） */
export interface PartyMessage {
  type: "party";
  state: PartyState;
  count: number;
  facing: boolean;
}

export type DownMessage =
  | StateMessage
  | CaptionMessage
  | SpeakMessage
  | ThemeMessage
  | SourcesMessage
  | CameraMessage
  | PartyMessage;

/** 上りバイナリの先頭 1 バイト（server/src/edge_bridge.py の KIND_*） */
export const KIND_PCM = 0x01;
export const KIND_JPEG = 0x02;

export interface HelloMessage {
  type: "hello";
  edge_id: string;
  sample_rate: number;
  caps: { aec: boolean };
}

/** カメラを開けたか（開けなければフロントは輪の判定をせず、従来どおり聞く） */
export interface CameraStatusMessage {
  type: "camera_status";
  ok: boolean;
  label: string;
  width: number;
  height: number;
}

/** 再生キューが空になった。last_seq は最後に鳴らし終えた文の番号（契約への追加） */
export interface PlaybackEndedMessage {
  type: "playback_ended";
  utterance_id: string;
  last_seq: number;
}

export type UpMessage = HelloMessage | PlaybackEndedMessage | CameraStatusMessage;
