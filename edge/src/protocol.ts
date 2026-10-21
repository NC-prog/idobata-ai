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

export interface ThemeMessage {
  type: "theme";
  site: string;
  season: string;
}

export type DownMessage = StateMessage | CaptionMessage | SpeakMessage | ThemeMessage;

export interface HelloMessage {
  type: "hello";
  edge_id: string;
  sample_rate: number;
  caps: { aec: boolean };
}

/** 再生キューが空になった。last_seq は最後に鳴らし終えた文の番号（契約への追加） */
export interface PlaybackEndedMessage {
  type: "playback_ended";
  utterance_id: string;
  last_seq: number;
}

export type UpMessage = HelloMessage | PlaybackEndedMessage;
