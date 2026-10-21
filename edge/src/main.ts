// 井戸端AI エッジページ: ブラウザのマイクで聞き、フロントAI の応答を口パク付きで鳴らす。

import * as THREE from "three";

import { startMic, SpeechPlayer } from "./audio";
import { Avatar } from "./avatar";
import { FrontLink } from "./link";
import type { DownMessage, SpeakMessage } from "./protocol";

const WS_PORT = 8765;
const MODEL_URL = `${import.meta.env.BASE_URL}models/VRM1_Constraint_Twist_Sample.vrm`;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const statusEl = $<HTMLDivElement>("status");
const userCaption = $<HTMLParagraphElement>("caption-user");
const assistantCaption = $<HTMLParagraphElement>("caption-assistant");
const sourcesCaption = $<HTMLParagraphElement>("caption-sources");
const startEl = $<HTMLDivElement>("start");
const backgroundEl = $<HTMLDivElement>("background");

// ---------------------------------------------------------------- 描画
const canvas = $<HTMLCanvasElement>("avatar");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 20);
const light = new THREE.DirectionalLight(0xffffff, Math.PI);
light.position.set(0.5, 1.5, 2);
scene.add(light, new THREE.AmbientLight(0xffffff, 0.6));

function resize(): void {
  const w = window.innerWidth;
  const h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

const avatar = new Avatar(scene, camera);
avatar.load(MODEL_URL).catch((err: unknown) => {
  console.error("VRM を読み込めません", err);
  statusEl.textContent = "アバターを読み込めません";
});

// ---------------------------------------------------------------- 音声
const playbackCtx = new AudioContext();
let micStarted = false;

const link = new FrontLink(`ws://${location.hostname}:${WS_PORT}`, {
  onOpen: () => {
    statusEl.textContent = "接続中";
    sendHello();
  },
  onClose: () => {
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
// アバターの字幕は、文が鳴り始めたときにその文までを出す（届いた時点で先に出さない）
const spokenTexts = new Map<string, string[]>();
let shownSegment = "";
let lastSpeakAt = 0;
// speak が来ていない応答（合成失敗など）だけ caption の本文をそのまま出す
const CAPTION_FALLBACK_MS = 2000;

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
    case "sources":
      sourcesCaption.textContent = msg.titles.length ? `出典: ${msg.titles.join(" / ")}` : "";
      break;
    case "caption":
      if (msg.role === "user") {
        if (msg.final) userCaption.textContent = msg.text;
      } else if (performance.now() - lastSpeakAt > CAPTION_FALLBACK_MS) {
        assistantCaption.textContent = msg.text;
      }
      break;
    case "speak":
      pendingSpeak = msg;
      lastSpeakAt = performance.now();
      break;
    case "theme":
      backgroundEl.style.backgroundImage =
        `url("${import.meta.env.BASE_URL}backgrounds/${msg.site}_${msg.season}.jpg"), var(--bg-fallback)`;
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

function updateAssistantCaption(): void {
  const seg = player.currentSegment();
  if (!seg) return;
  const key = `${seg.utteranceId}/${seg.seq}`;
  if (key === shownSegment) return;
  shownSegment = key;
  const texts = spokenTexts.get(seg.utteranceId) ?? [];
  assistantCaption.textContent = texts.slice(0, seg.seq + 1).join("");
  // 古い発話の本文は持ち続けない
  for (const id of spokenTexts.keys()) if (id !== seg.utteranceId) spokenTexts.delete(id);
}

async function start(): Promise<void> {
  startEl.hidden = true;
  await playbackCtx.resume();
  try {
    const mic = await startMic((pcm) => link.sendBinary(pcm));
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
// 自動再生が許されている（キオスク起動のフラグ付き）ならそのまま始める。そうでなければボタンを出す
if (playbackCtx.state === "running") {
  void start();
} else {
  startEl.hidden = false;
  $<HTMLButtonElement>("start-button").addEventListener("click", () => void start());
}

// ---------------------------------------------------------------- ループ
const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  const dt = Math.min(clock.getDelta(), 0.1);
  avatar.update(dt, player.currentViseme()?.v ?? null);
  updateAssistantCaption();
  renderer.render(scene, camera);
});
