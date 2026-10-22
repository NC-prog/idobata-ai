// カメラ（WP02 A）: ブラウザでカメラを開き、縮めた JPEG を一定間隔でフロントへ送る。
// フロントは受け取ったフレームをメモリ上でだけ使う（保存しない）。顔識別はしない。

import type { CameraMessage } from "./protocol";

export interface CameraInfo {
  label: string;
  width: number;
  height: number;
}

/** 複数のカメラがあれば label に preferred を含むものを選ぶ（無ければ最初のもの） */
async function pickDevice(preferred: string): Promise<string | undefined> {
  const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  const hit = devices.find((d) => preferred && d.label.includes(preferred));
  return (hit ?? devices[0])?.deviceId || undefined;
}

export async function startCamera(
  cfg: CameraMessage,
  onFrame: (jpeg: ArrayBuffer) => void,
): Promise<CameraInfo> {
  const constraints = (deviceId?: string): MediaStreamConstraints => ({
    audio: false,
    video: {
      ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
      width: cfg.capture.width,
      height: cfg.capture.height,
      frameRate: cfg.capture.fps,
    },
  });
  // 一度開かないと label が読めないブラウザがあるので、開いてから選び直す
  let stream = await navigator.mediaDevices.getUserMedia(constraints());
  const wanted = await pickDevice(cfg.label);
  const current = stream.getVideoTracks()[0]?.getSettings().deviceId;
  if (wanted && wanted !== current) {
    stream.getTracks().forEach((t) => t.stop());
    stream = await navigator.mediaDevices.getUserMedia(constraints(wanted));
  }
  const track = stream.getVideoTracks()[0];
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.srcObject = stream;
  await video.play();

  const canvas = document.createElement("canvas");
  canvas.width = cfg.send.width;
  canvas.height = cfg.send.height;
  const ctx = canvas.getContext("2d")!;
  let busy = false;
  window.setInterval(() => {
    if (busy || video.readyState < 2) return;
    busy = true;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(
      (blob) => {
        busy = false;
        if (blob) void blob.arrayBuffer().then(onFrame);
      },
      "image/jpeg",
      cfg.send.quality,
    );
  }, 1000 / cfg.send.fps);
  const s = track.getSettings();
  return { label: track.label, width: s.width ?? 0, height: s.height ?? 0 };
}
