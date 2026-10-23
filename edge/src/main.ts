// 井戸端AI エッジページ: ブラウザのマイクで聞き、フロントAI の応答を口パク付きで鳴らす。

import * as THREE from "three";

import { startMic, SpeechPlayer } from "./audio";
import { Avatar } from "./avatar";
import { startCamera } from "./camera";
import { FrontLink } from "./link";
import { KIND_JPEG, KIND_PCM } from "./protocol";
import type { CameraMessage, DownMessage, SpeakMessage, ThemeMessage } from "./protocol";

// WebSocket のポート。既定は 8765、ページの URL に ?ws=<番号> を付けると変えられる（試験用に別ポートで動かすとき）
const WS_PORT = Number(new URLSearchParams(location.search).get("ws")) || 8765;
// 開発用の既定のアバター。本番のアバターはフロントの theme（~/idobata-data/avatars/ から /edge/avatars/ で配信）で読む。
// フロントにつながらないまま THEME_WAIT_MS 経ったら、これを出しておく
const DEFAULT_MODEL = "models/VRM1_Constraint_Twist_Sample.vrm";
const THEME_WAIT_MS = 4000;
const asset = (path: string) => `${import.meta.env.BASE_URL}${path}`;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const statusEl = $<HTMLDivElement>("status");
const userCaption = $<HTMLParagraphElement>("caption-user");
const assistantCaption = $<HTMLParagraphElement>("caption-assistant");
const sourcesCaption = $<HTMLParagraphElement>("caption-sources");
const previousUser = $<HTMLParagraphElement>("previous-user");
const previousAssistant = $<HTMLParagraphElement>("previous-assistant");
const startEl = $<HTMLDivElement>("start");
const backgroundEl = $<HTMLDivElement>("background");
const replacedEl = $<HTMLDivElement>("replaced");

// ---------------------------------------------------------------- 描画
const canvas = $<HTMLCanvasElement>("avatar");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 20);
const light = new THREE.DirectionalLight(0xffffff, Math.PI);
light.position.set(0.5, 1.5, 2);
scene.add(light, new THREE.AmbientLight(0xffffff, 0.6));

const avatar = new Avatar(scene, camera);

function resize(): void {
  const w = window.innerWidth;
  const h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  avatar.frame(); // 縦置き・横置きで構図を取り直す
}
window.addEventListener("resize", resize);
resize();

/** アバターを読み込む。失敗したら、何も出ていなければ既定のサンプルを出す */
function loadAvatar(url: string): void {
  avatar.load(url).catch((err: unknown) => {
    console.error("VRM を読み込めません", url, err);
    if (avatar.currentUrl === null && url !== asset(DEFAULT_MODEL)) {
      loadAvatar(asset(DEFAULT_MODEL));
      return;
    }
    statusEl.textContent = "アバターを読み込めません";
  });
}

// 確認用（WP08 のスクリーンショット・口パクの確かめ）。DevTools から __avatarDebug() で読む
(window as unknown as { __avatarDebug: () => unknown }).__avatarDebug = () => avatar.debugInfo();

let themeReceived = false;
setTimeout(() => {
  if (!themeReceived) loadAvatar(asset(DEFAULT_MODEL));
}, THEME_WAIT_MS);

/** テーマ（WP08）: アバターと背景を読み直す（同じものなら何もしない）。フロントは会話中には送ってこない */
function applyTheme(msg: ThemeMessage): void {
  themeReceived = true;
  if (msg.avatar) loadAvatar(asset(msg.avatar));
  const bg = msg.background || `backgrounds/${msg.site}_${msg.season}.jpg`; // 古いフロント（WP07 まで）は site と season だけ
  const url = `url("${asset(bg)}"), var(--bg-fallback)`;
  if (backgroundEl.style.backgroundImage !== url) backgroundEl.style.backgroundImage = url;
  console.info("テーマ", msg.name ?? msg.season);
}

// ---------------------------------------------------------------- 音声
const playbackCtx = new AudioContext();
let micStarted = false;

const link = new FrontLink(`ws://${location.hostname}:${WS_PORT}`, {
  onOpen: () => {
    statusEl.textContent = "接続中";
    sendHello();
  },
  onClose: (replaced) => {
    if (replaced) {
      statusEl.textContent = "休止中（別の画面がつながりました）";
      replacedEl.hidden = false;
      return;
    }
    statusEl.textContent = "未接続（再接続を試みています）";
  },
  onMessage: handleMessage,
  onBinary: handleBinary,
});

const player = new SpeechPlayer(playbackCtx, (utteranceId, lastSeq) => {
  link.send({ type: "playback_ended", utterance_id: utteranceId, last_seq: lastSeq });
});

let micAec = false;
let pendingSpeak: SpeakMessage | null = null;
// アバターの字幕は、鳴っている文だけを声に合わせて出す（届いた文を先に出さない）。
// 来訪者が次に話したら、それまでの 1 往復（発話と応答の全文）を上に薄く残す
const spokenTexts = new Map<string, string[]>();
let currentUserText = "";
let currentUtterance = "";
let shownSegment = "";
let lastSpeakAt = 0;
// speak が来ていない応答（合成失敗など）だけ caption の本文をそのまま出す
const CAPTION_FALLBACK_MS = 2000;

/** 上りバイナリに種別の 1 バイトを付ける（PCM と JPEG を同じ接続で送る） */
function framed(kind: number, data: ArrayBuffer): ArrayBuffer {
  const out = new Uint8Array(data.byteLength + 1);
  out[0] = kind;
  out.set(new Uint8Array(data), 1);
  return out.buffer;
}

let cameraStarted = false;

/** フロントから届いたカメラの取り方で、カメラを 1 回だけ開く（開けなければフロントに知らせる） */
async function onCameraConfig(msg: CameraMessage): Promise<void> {
  if (cameraStarted || !msg.enabled) return;
  cameraStarted = true;
  try {
    const info = await startCamera(msg, (jpeg) => link.sendBinary(framed(KIND_JPEG, jpeg)));
    console.info("カメラ開始", info);
    link.send({ type: "camera_status", ok: true, label: info.label, width: info.width, height: info.height });
  } catch (err) {
    console.error("カメラを開けません", err);
    link.send({ type: "camera_status", ok: false, label: "", width: 0, height: 0 });
  }
}

function sendHello(): void {
  if (!micStarted) return; // マイクが動いてから hello する（フロントは hello した接続の音声を聞く）
  link.send({ type: "hello", edge_id: edgeId(), sample_rate: 16000, caps: { aec: micAec } });
}

function edgeId(): string {
  const key = "idobata-edge-id";
  try {
    let id = localStorage.getItem(key);
    if (!id) {
      id = `edge-${Math.random().toString(36).slice(2, 10)}`;
      localStorage.setItem(key, id);
    }
    return id;
  } catch {
    return "edge-local";
  }
}

function handleMessage(msg: DownMessage): void {
  switch (msg.type) {
    case "state":
      avatar.setState(msg.value);
      if (msg.value === "listening" || msg.value === "thinking") sourcesCaption.textContent = "";
      break;
    case "sources": {
      sourcesCaption.textContent = msg.titles.length ? `出典: ${msg.titles.join(" / ")}` : "";
      const urls = msg.urls ?? [];
      if (msg.titles.length && urls.length) {
        const span = document.createElement("span");
        span.className = "source-urls";
        span.textContent = ` ${urls.join(" ")}`;
        sourcesCaption.appendChild(span);
      }
      break;
    }
    case "caption":
      if (msg.role === "user") {
        if (msg.final) onUserFinal(msg.text);
      } else if (performance.now() - lastSpeakAt > CAPTION_FALLBACK_MS) {
        assistantCaption.textContent = msg.text;
      }
      break;
    case "speak":
      pendingSpeak = msg;
      lastSpeakAt = performance.now();
      break;
    case "camera":
      void onCameraConfig(msg);
      break;
    case "party":
      avatar.setParty(msg.state);
      break;
    case "theme":
      applyTheme(msg);
      break;
  }
}

function handleBinary(data: ArrayBuffer): void {
  const speak = pendingSpeak;
  pendingSpeak = null;
  if (!speak) {
    console.warn("speak の無い音声を受信しました");
    return;
  }
  const texts = spokenTexts.get(speak.utterance_id) ?? [];
  texts[speak.seq] = speak.text;
  spokenTexts.set(speak.utterance_id, texts);
  player.enqueue(
    { utteranceId: speak.utterance_id, seq: speak.seq, text: speak.text, mouth: speak.mouth },
    data,
  );
}

/** 来訪者の発話が確定したら、ひとつ前のやりとりを上に移して、新しい発話を出す */
function onUserFinal(text: string): void {
  const reply = (spokenTexts.get(currentUtterance) ?? []).join("") || assistantCaption.textContent || "";
  if (currentUserText || reply) {
    previousUser.textContent = currentUserText;
    previousAssistant.textContent = reply;
  }
  currentUserText = text;
  userCaption.textContent = text;
  assistantCaption.textContent = "";
  shownSegment = "";
}

/** 鳴り始めた文に切り替える（文と文の間は直前の文を出したまま） */
function updateAssistantCaption(): void {
  const seg = player.currentSegment();
  if (!seg) return;
  const key = `${seg.utteranceId}/${seg.seq}`;
  if (key === shownSegment) return;
  shownSegment = key;
  currentUtterance = seg.utteranceId;
  assistantCaption.textContent = seg.text;
  // 古い発話の本文は持ち続けない（ひとつ前の 1 往復の分だけは onUserFinal まで残す）
  for (const id of spokenTexts.keys()) if (id !== seg.utteranceId) spokenTexts.delete(id);
}

async function start(): Promise<void> {
  startEl.hidden = true;
  await playbackCtx.resume();
  try {
    const mic = await startMic((pcm) => link.sendBinary(framed(KIND_PCM, pcm)));
    micAec = mic.aec;
    micStarted = true;
    console.info("マイク開始", mic);
  } catch (err) {
    console.error("マイクを開けません", err);
    statusEl.textContent = "マイクを開けません";
    startEl.hidden = false;
    return;
  }
  if (link.isOpen) sendHello();
}

link.connect();
$<HTMLButtonElement>("reclaim-button").addEventListener("click", () => {
  replacedEl.hidden = true;
  link.connect(); // hello すると、今度はこちらが音声を受け持つ（前の画面が休止する）
});
// 自動再生が許されている（キオスク起動のフラグ付き）ならそのまま始める。そうでなければボタンを出す
if (playbackCtx.state === "running") {
  void start();
} else {
  startEl.hidden = false;
  $<HTMLButtonElement>("start-button").addEventListener("click", () => void start());
}

// ---------------------------------------------------------------- ループ
const timer = new THREE.Timer();
renderer.setAnimationLoop((time) => {
  timer.update(time);
  const dt = Math.min(timer.getDelta(), 0.1);
  avatar.update(dt, player.currentViseme()?.v ?? null);
  updateAssistantCaption();
  renderer.render(scene, camera);
});
