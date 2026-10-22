// VRM アバター 1 体の表示と、状態ごとの見た目（待機・傾聴・考え中・発話）。

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { VRMLoaderPlugin, VRMUtils, type VRM, type VRMHumanBoneName } from "@pixiv/three-vrm";

import type { AvatarState, PartyState, Viseme } from "./protocol";

const VISEMES: Viseme[] = ["aa", "ih", "ou", "ee", "oh"];

// 状態ごとの姿勢と表情（すべて「薄く乗せる」程度）
interface Pose {
  lean: number; // 前傾（spine の X 回転、rad）
  headYaw: number; // 顔の向き（左右）
  headPitch: number; // 顔の向き（上下。正で下向き）
  gazeX: number; // 視線（カメラからのずれ、m）
  gazeY: number;
  happy: number;
  relaxed: number;
}

const POSES: Record<AvatarState, Pose> = {
  idle: { lean: 0, headYaw: 0, headPitch: 0, gazeX: 0, gazeY: 0, happy: 0.1, relaxed: 0.15 },
  listening: { lean: 0.08, headYaw: 0, headPitch: 0.04, gazeX: 0, gazeY: 0, happy: 0.2, relaxed: 0 },
  thinking: { lean: 0.02, headYaw: 0.12, headPitch: -0.06, gazeX: 0.35, gazeY: 0.25, happy: 0, relaxed: 0.1 },
  speaking: { lean: 0.04, headYaw: 0, headPitch: 0, gazeX: 0, gazeY: 0, happy: 0.25, relaxed: 0 },
};

// 目の前の輪（WP02）による上乗せ。誰もいないときは少しうつむいて視線を外し、近くに人が来たら顔を上げ、
// 輪ができたら相手（カメラ）を見る。カメラが無いときは PARTY 扱い（上乗せ無し＝従来どおり）
const PARTY_OFFSET: Record<PartyState, Partial<Pose>> = {
  NONE: { headPitch: 0.08, gazeX: 0.25, gazeY: -0.35, happy: -0.05 },
  PRESENT: { headPitch: -0.03, gazeX: 0.1, gazeY: 0 },
  PARTY: {},
};

// 姿勢の切り替わりの速さ（1/秒）と、口形の追従の速さ
const POSE_RATE = 4;
const MOUTH_RATE = 28;
const MOUTH_OPEN = 0.85;
// 構図（WP08）: 胸から上。頭のてっぺんの上に画面の高さの FRAME_TOP_MARGIN を空け、胸の骨の FRAME_BELOW_CHEST m 下までを、
// 画面の上 FRAME_UPPER（下の残りは字幕が重なる）に収める。縦置きで幅が足りないときは、肩幅の FRAME_WIDTH 倍が入るまで引く
const FRAME_TOP_MARGIN = 0.07;
const FRAME_BELOW_CHEST = 0.08;
const FRAME_UPPER = 0.8;
const FRAME_WIDTH = 2.1;
// 使う表情（モデルに無いものは使わない。VRoid Studio 製の VRM 1.0 なら全部ある）
const EXPRESSIONS = ["happy", "relaxed", "blink", ...VISEMES] as const;

function approach(current: number, target: number, rate: number, dt: number): number {
  return current + (target - current) * Math.min(1, rate * dt);
}

export class Avatar {
  private vrm: VRM | null = null;
  private url: string | null = null;
  private loadSeq = 0;
  private available = new Set<string>();
  private readonly gazeTarget = new THREE.Object3D();
  private readonly camera: THREE.PerspectiveCamera;
  private state: AvatarState = "idle";
  private party: PartyState = "PARTY";
  private pose: Pose = { ...POSES.idle };
  private mouth: Record<Viseme, number> = { aa: 0, ih: 0, ou: 0, ee: 0, oh: 0 };
  private elapsed = 0;
  private nextBlink = 2;
  private blinkT = -1;

  constructor(private readonly scene: THREE.Scene, camera: THREE.PerspectiveCamera) {
    this.camera = camera;
    scene.add(this.gazeTarget);
  }

  /** 読み込み済み（または読み込み中）の VRM の URL */
  get currentUrl(): string | null {
    return this.url;
  }

  /**
   * VRM を読み込んで差し替える（テーマの切り替え、WP08）。読み終わるまでは前のモデルを出したまま。
   * 読み込み中に別の URL が来たら、古い方は読み終わっても捨てる。失敗したら前のモデルのまま例外を投げる
   */
  async load(url: string): Promise<void> {
    if (url === this.url) return;
    const prevUrl = this.url;
    this.url = url;
    const seq = ++this.loadSeq;
    const loader = new GLTFLoader();
    loader.register((parser) => new VRMLoaderPlugin(parser));
    let gltf;
    try {
      gltf = await loader.loadAsync(url);
    } catch (err) {
      if (seq === this.loadSeq) this.url = prevUrl;
      throw err;
    }
    const vrm = gltf.userData.vrm as VRM | undefined;
    if (!vrm) {
      if (seq === this.loadSeq) this.url = prevUrl;
      throw new Error(`VRM ではありません: ${url}`);
    }
    if (seq !== this.loadSeq) {
      VRMUtils.deepDispose(gltf.scene); // 読み込み中に次の切り替えが来た
      return;
    }
    VRMUtils.removeUnnecessaryVertices(gltf.scene);
    VRMUtils.combineSkeletons(gltf.scene);
    VRMUtils.rotateVRM0(vrm); // VRM 0.x を VRM 1.0 と同じ向き（+Z 向き）に揃える
    vrm.scene.traverse((obj) => {
      obj.frustumCulled = false;
    });
    const old = this.vrm;
    if (old) {
      this.scene.remove(old.scene);
      VRMUtils.deepDispose(old.scene);
    }
    this.scene.add(vrm.scene);
    if (vrm.lookAt) vrm.lookAt.target = this.gazeTarget;
    this.vrm = vrm;
    const names = Object.keys(vrm.expressionManager?.expressionMap ?? {});
    this.available = new Set(EXPRESSIONS.filter((e) => names.includes(e)));
    const missing = EXPRESSIONS.filter((e) => !this.available.has(e));
    if (missing.length) console.warn("VRM に無い表情（使いません）", missing);
    this.mouth = { aa: 0, ih: 0, ou: 0, ee: 0, oh: 0 };
    this.restPose();
    this.frame();
  }

  /** 確認用（WP08）: 使える表情と今の重み、視線の向き */
  debugInfo(): Record<string, unknown> {
    const em = this.vrm?.expressionManager;
    const weights: Record<string, number> = {};
    for (const name of this.available) weights[name] = em?.getValue(name) ?? 0;
    const look = this.vrm?.lookAt;
    return {
      url: this.url,
      title: this.vrm?.meta?.metaVersion === "1" ? this.vrm.meta.name : null,
      available: [...this.available],
      weights,
      lookAt: look ? { yaw: look.yaw, pitch: look.pitch } : null,
      camera: this.camera.position.toArray(),
    };
  }

  setState(state: AvatarState): void {
    this.state = state;
  }

  setParty(party: PartyState): void {
    this.party = party;
  }

  /**
   * 胸から上が入るようにカメラを置く（頭のてっぺん・胸・肩幅はモデルから取る）。画面の縦横が変わったときも呼ぶ。
   * カメラは水平に向ける（見上げ・見下ろしにしない）
   */
  frame(): void {
    const vrm = this.vrm;
    const head = this.bone("head");
    if (!vrm || !head) return;
    vrm.scene.updateMatrixWorld(true);
    const headPos = head.getWorldPosition(new THREE.Vector3());
    const chestBone = this.bone("upperChest") ?? this.bone("chest") ?? this.bone("spine");
    const chestY = chestBone ? chestBone.getWorldPosition(new THREE.Vector3()).y : headPos.y - 0.3;
    // 頭のてっぺん: 頭の骨より上にある頂点（髪を含む）の高さ。取れなければ頭の骨 + 0.2 m
    const box = new THREE.Box3().setFromObject(vrm.scene);
    const top = Number.isFinite(box.max.y) && box.max.y > headPos.y ? Math.min(box.max.y, headPos.y + 0.4) : headPos.y + 0.2;
    const l = this.bone("leftUpperArm")?.getWorldPosition(new THREE.Vector3());
    const r = this.bone("rightUpperArm")?.getWorldPosition(new THREE.Vector3());
    const shoulders = l && r ? l.distanceTo(r) : 0.3;

    const tanV = Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));
    const span = top - (chestY - FRAME_BELOW_CHEST);
    let height = span / (FRAME_UPPER - FRAME_TOP_MARGIN); // 画面の高さ（m、モデルの位置で）
    height = Math.max(height, (shoulders * FRAME_WIDTH) / this.camera.aspect);
    const distance = height / 2 / tanV;
    const centerY = top + height * FRAME_TOP_MARGIN - height / 2;
    this.camera.position.set(0, centerY, headPos.z + distance);
    this.camera.lookAt(0, centerY, headPos.z);
  }

  /** T ポーズから腕を下ろす */
  private restPose(): void {
    const set = (name: VRMHumanBoneName, z: number, x = 0) => {
      const b = this.bone(name);
      if (b) b.rotation.set(x, 0, z);
    };
    set("leftUpperArm", -1.2, 0.1);
    set("rightUpperArm", 1.2, 0.1);
    set("leftLowerArm", -0.15);
    set("rightLowerArm", 0.15);
  }

  private bone(name: VRMHumanBoneName): THREE.Object3D | null {
    return this.vrm?.humanoid.getNormalizedBoneNode(name) ?? null;
  }

  /** 毎フレーム。viseme は今鳴っている口形（無音なら null） */
  update(dt: number, viseme: Viseme | null): void {
    const vrm = this.vrm;
    if (!vrm) return;
    this.elapsed += dt;
    const target = POSES[this.state];
    // 話している・考えている間は輪の上乗せをしない（相手に向いたまま）
    const offset = this.state === "idle" || this.state === "listening" ? PARTY_OFFSET[this.party] : {};
    const p = this.pose;
    for (const key of Object.keys(p) as (keyof Pose)[]) {
      p[key] = approach(p[key], target[key] + (offset[key] ?? 0), POSE_RATE, dt);
    }

    // 呼吸（ゆるい上下）と、発話中の軽い頷き
    const breath = Math.sin(this.elapsed * 1.6) * 0.012;
    const nod = this.state === "speaking" ? Math.sin(this.elapsed * 3.1) * 0.035 : 0;
    const spine = this.bone("spine");
    const chest = this.bone("chest");
    const neck = this.bone("neck");
    const head = this.bone("head");
    if (spine) spine.rotation.x = p.lean + breath;
    if (chest) chest.rotation.x = breath * 0.6;
    if (neck) neck.rotation.set(p.headPitch * 0.4, p.headYaw * 0.4, 0);
    if (head) head.rotation.set(p.headPitch * 0.6 + nod, p.headYaw * 0.6, 0);

    // 視線: カメラ（来訪者）を見る。考え中は少し外す
    const cam = this.camera.position;
    this.gazeTarget.position.set(cam.x + p.gazeX, cam.y + p.gazeY, cam.z);

    const em = vrm.expressionManager;
    if (em) {
      const set = (name: string, w: number) => {
        if (this.available.has(name)) em.setValue(name, w);
      };
      set("happy", p.happy);
      set("relaxed", p.relaxed);
      set("blink", this.blinkWeight(dt));
      for (const v of VISEMES) {
        this.mouth[v] = approach(this.mouth[v], v === viseme ? MOUTH_OPEN : 0, MOUTH_RATE, dt);
        set(v, this.mouth[v]);
      }
    }
    vrm.update(dt);
  }

  /** 3〜6 秒おきのまばたき（0.15 秒） */
  private blinkWeight(dt: number): number {
    if (this.blinkT < 0) {
      this.nextBlink -= dt;
      if (this.nextBlink <= 0) this.blinkT = 0;
      return 0;
    }
    this.blinkT += dt;
    const dur = 0.15;
    if (this.blinkT >= dur) {
      this.blinkT = -1;
      this.nextBlink = 3 + Math.random() * 3;
      return 0;
    }
    return Math.sin((this.blinkT / dur) * Math.PI);
  }
}
