from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# PyBullet is required for this generator.
import pybullet as p
import pybullet_data

import imageio.v2 as imageio


# ----------------------------
# Taxonomy codes (modules)
# ----------------------------

TAXONOMIES = [
    "TR-ALI",
    "TR-ORDER",
    "TM-ID",
    "P-OCC",
    "R-COUNT",
    "PH-LATENT-FRICTION",
    "PH-LATENT-MASS",
    "PH-INVAR-ENERGY",
    "C-CHAIN",
    "PH-COUNTER",
]


# ----------------------------
# CLI / Config
# ----------------------------

@dataclass
class RenderConfig:
    fps: int = 30
    sim_hz: int = 240
    width: int = 640
    height: int = 360
    duration_s: float = 5.0
    gravity: float = -9.8


@dataclass
class CameraConfig:
    # A simple, consistent "lab camera"
    eye: Tuple[float, float, float] = (0.0, -2.2, 1.0)
    target: Tuple[float, float, float] = (0.0, 0.2, 0.2)
    up: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    fov: float = 60.0
    near: float = 0.05
    far: float = 10.0


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--pairs_per_taxonomy", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--sim_hz", type=int, default=240)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--duration_s", type=float, default=5.0)

    ap.add_argument("--taxonomies", type=str, default=",".join(TAXONOMIES),
                    help="Comma-separated subset of taxonomy codes to generate")

    return ap.parse_args()


# ----------------------------
# Bullet utilities
# ----------------------------

def connect_and_reset(cfg: RenderConfig) -> None:
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setTimeStep(1.0 / cfg.sim_hz)
    p.setGravity(0, 0, cfg.gravity)
    p.loadURDF("plane.urdf")


def disconnect() -> None:
    try:
        p.disconnect()
    except Exception:
        pass


def quat_from_euler(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    return p.getQuaternionFromEuler((roll, pitch, yaw))


def make_box(
    half_extents: Tuple[float, float, float],
    pos: Tuple[float, float, float],
    mass: float,
    rgba: Tuple[float, float, float, float],
    orn: Tuple[float, float, float, float] = (0, 0, 0, 1),
) -> int:
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=rgba)
    body = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos,
        baseOrientation=orn,
    )
    return body


def make_sphere(
    radius: float,
    pos: Tuple[float, float, float],
    mass: float,
    rgba: Tuple[float, float, float, float],
) -> int:
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
    body = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos,
    )
    return body


def set_dyn(
    body: int,
    *,
    friction: float = 0.4,
    restitution: float = 0.2,
    linear_damping: float = 0.0,
    angular_damping: float = 0.0,
) -> None:
    p.changeDynamics(
        body, -1,
        lateralFriction=friction,
        restitution=restitution,
        linearDamping=linear_damping,
        angularDamping=angular_damping,
        rollingFriction=0.0,
        spinningFriction=0.0,
    )


def render_frame(cam: CameraConfig, rcfg: RenderConfig) -> np.ndarray:
    view = p.computeViewMatrix(
        cameraEyePosition=cam.eye,
        cameraTargetPosition=cam.target,
        cameraUpVector=cam.up,
    )
    aspect = rcfg.width / rcfg.height
    proj = p.computeProjectionMatrixFOV(
        fov=cam.fov,
        aspect=aspect,
        nearVal=cam.near,
        farVal=cam.far,
    )
    _, _, rgba, _, _ = p.getCameraImage(
        width=rcfg.width,
        height=rcfg.height,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER,
    )
    img = np.reshape(rgba, (rcfg.height, rcfg.width, 4))[:, :, :3]
    return img


def simulate_and_render(
    *,
    rcfg: RenderConfig,
    cam: CameraConfig,
    out_mp4: str,
    build_scene: Callable[[], Dict[str, Any]],
    intervention: Optional[Callable[[float, int, Dict[str, Any]], None]] = None,
) -> None:
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    connect_and_reset(rcfg)

    ids = build_scene()

    steps = int(rcfg.duration_s * rcfg.sim_hz)
    frame_every = max(1, int(rcfg.sim_hz / rcfg.fps))

    writer = imageio.get_writer(out_mp4, fps=rcfg.fps, codec="libx264", quality=8)
    try:
        for step in range(steps):
            t = step / rcfg.sim_hz
            if intervention is not None:
                intervention(t, step, ids)
            p.stepSimulation()
            if step % frame_every == 0:
                writer.append_data(render_frame(cam, rcfg))
    finally:
        writer.close()
        disconnect()


# ----------------------------
# Dataset writing
# ----------------------------

def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def qa_row(
    *,
    example_id: str,
    pair_id: str,
    pair_role: str,
    question_type: str,
    prompt: str,
    options: Optional[List[str]],
    answer_index: int,
    video_path: str,
    module: str,
    critical_window: Tuple[float, float],
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "example_id": example_id,
        "pair_id": pair_id,
        "pair_role": pair_role,
        "question_id": "q1",
        "question_type": question_type,
        "prompt": prompt,
        "options": options,
        "answer_index": answer_index,
        "video_path": video_path,
        "module": module,
        "domain": module,
        "critical_window": [float(critical_window[0]), float(critical_window[1])],
        "source_type": "simulated",
        "generation": extra,
    }
    return row


# ----------------------------
# Taxonomy generators
# Each returns:
#   (videoA_path, videoB_path, rows_jsonl, pair_meta)
# ----------------------------

def gen_TR_ALI(pair_id: str, out_videos_dir: str, seed: int, rcfg: RenderConfig, cam: CameraConfig):
    rng = np.random.default_rng(seed)

    # Config: sliding sphere with optional micro-teleport
    radius = float(rng.uniform(0.05, 0.07))
    v = float(rng.uniform(0.7, 1.2))
    teleport_t = float(rng.uniform(2.0, 3.0))
    teleport_dt = 1.0 / rcfg.fps  # one frame
    delta = float(rng.uniform(0.06, 0.14))  # small but visible jump
    axis = rng.choice(["x", "y"])

    critical = (teleport_t - 0.15, teleport_t + 0.15)

    def build():
        ball = make_sphere(radius, (0.0, -1.0, radius), 1.0, (0.85, 0.2, 0.2, 1.0))
        set_dyn(ball, friction=0.02, restitution=0.2)
        p.resetBaseVelocity(ball, linearVelocity=(0.0, v, 0.0))
        return {"ball": ball}

    def intervene_B(t: float, step: int, ids: Dict[str, Any]):
        # Micro-teleport near teleport_t (one-frame duration)
        if teleport_t <= t < teleport_t + teleport_dt:
            pos, orn = p.getBasePositionAndOrientation(ids["ball"])
            if axis == "x":
                p.resetBasePositionAndOrientation(ids["ball"], (pos[0] + delta, pos[1], pos[2]), orn)
            else:
                p.resetBasePositionAndOrientation(ids["ball"], (pos[0], pos[1] + delta, pos[2]), orn)

    vA = os.path.join(out_videos_dir, "TR-ALI", f"pair_{pair_id}_A.mp4")
    vB = os.path.join(out_videos_dir, "TR-ALI", f"pair_{pair_id}_B.mp4")

    simulate_and_render(rcfg=rcfg, cam=cam, out_mp4=vA, build_scene=build, intervention=None)
    simulate_and_render(rcfg=rcfg, cam=cam, out_mp4=vB, build_scene=build, intervention=intervene_B)

    prompt = "Does any object move discontinuously (teleport/jump) at any point in the video?"
    # yesno: YES->1, NO->0 (must match evaluator convention)
    rows = [
        qa_row(
            example_id=f"{pair_id}_A", pair_id=pair_id, pair_role="A",
            question_type="yesno", prompt=prompt, options=None, answer_index=0,
            video_path=os.path.relpath(vA, os.path.dirname(out_videos_dir)),
            module="TR-ALI", critical_window=critical,
            extra={"seed": seed, "template": "TR-ALI", "variant": "A",
                   "params": {"radius": radius, "v": v, "teleport_t": teleport_t, "axis": axis, "delta": delta},
                   "intervention": None},
        ),
        qa_row(
            example_id=f"{pair_id}_B", pair_id=pair_id, pair_role="B",
            question_type="yesno", prompt=prompt, options=None, answer_index=1,
            video_path=os.path.relpath(vB, os.path.dirname(out_videos_dir)),
            module="TR-ALI", critical_window=critical,
            extra={"seed": seed, "template": "TR-ALI", "variant": "B",
                   "params": {"radius": radius, "v": v, "teleport_t": teleport_t, "axis": axis, "delta": delta},
                   "intervention": {"type": "teleport", "t": teleport_t, "axis": axis, "delta": delta}},
        )
    ]

    meta = {"pair_id": pair_id, "module": "TR-ALI", "critical_window": critical, "params": rows[0]["generation"]["params"]}
    return vA, vB, rows, meta


def gen_TR_ORDER(pair_id: str, out_videos_dir: str, seed: int, rcfg: RenderConfig, cam: CameraConfig):
    rng = np.random.default_rng(seed)

    radius = float(rng.uniform(0.05, 0.07))
    v = float(rng.uniform(0.8, 1.2))
    t0 = float(rng.uniform(0.8, 1.2))
    gap = float(rng.uniform(0.25, 0.5))  # time difference between launches

    critical = (t0 - 0.1, t0 + gap + 0.8)

    def build():
        # Static target block (mass=0)
        block = make_box((0.12, 0.12, 0.12), (0.0, 0.8, 0.12), 0.0, (0.6, 0.6, 0.6, 1.0))
        # Two colored balls
        red = make_sphere(radius, (-0.2, -1.0, radius), 1.0, (0.9, 0.1, 0.1, 1.0))
        blue = make_sphere(radius, (0.2, -1.0, radius), 1.0, (0.1, 0.2, 0.9, 1.0))
        for b in (red, blue):
            set_dyn(b, friction=0.02, restitution=0.8)
        return {"block": block, "red": red, "blue": blue}

    def launch(intervene_order: str):
        # intervene_order: "red_first" or "blue_first"
        def _fn(t: float, step: int, ids: Dict[str, Any]):
            # set velocities at specific times; keep them zero before launch
            if intervene_order == "red_first":
                t_red, t_blue = t0, t0 + gap
            else:
                t_blue, t_red = t0, t0 + gap

            if abs(t - t_red) < (1.0 / rcfg.sim_hz):
                p.resetBaseVelocity(ids["red"], linearVelocity=(0.0, v, 0.0))
            if abs(t - t_blue) < (1.0 / rcfg.sim_hz):
                p.resetBaseVelocity(ids["blue"], linearVelocity=(0.0, v, 0.0))
        return _fn

    vA = os.path.join(out_videos_dir, "TR-ORDER", f"pair_{pair_id}_A.mp4")
    vB = os.path.join(out_videos_dir, "TR-ORDER", f"pair_{pair_id}_B.mp4")

    simulate_and_render(rcfg=rcfg, cam=cam, out_mp4=vA, build_scene=build, intervention=launch("red_first"))
    simulate_and_render(rcfg=rcfg, cam=cam, out_mp4=vB, build_scene=build, intervention=launch("blue_first"))

    prompt = "Does the RED ball hit the gray block before the BLUE ball?"
    rows = [
        qa_row(
            example_id=f"{pair_id}_A", pair_id=pair_id, pair_role="A",
            question_type="yesno", prompt=prompt, options=None, answer_index=1,
            video_path=os.path.relpath(vA, os.path.dirname(out_videos_dir)),
            module="TR-ORDER", critical_window=critical,
            extra={"seed": seed, "template": "TR-ORDER", "variant": "A",
                   "params": {"radius": radius, "v": v, "t0": t0, "gap": gap},
                   "intervention": {"launch_order": "red_first"}},
        ),
        qa_row(
            example_id=f"{pair_id}_B", pair_id=pair_id, pair_role="B",
            question_type="yesno", prompt=prompt, options=None, answer_index=0,
            video_path=os.path.relpath(vB, os.path.dirname(out_videos_dir)),
            module="TR-ORDER", critical_window=critical,
            extra={"seed": seed, "template": "TR-ORDER", "variant": "B",
                   "params": {"radius": radius, "v": v, "t0": t0, "gap": gap},
                   "intervention": {"launch_order": "blue_first"}},
        ),
    ]
    meta = {"pair_id": pair_id, "module": "TR-ORDER", "critical_window": critical, "params": rows[0]["generation"]["params"]}
    return vA, vB, rows, meta


def gen_TM_ID(pair_id: str, out_videos_dir: str, seed: int, rcfg: RenderConfig, cam: CameraConfig):
    rng = np.random.default_rng(seed)

    # Two identical balls in two lanes; swap lanes behind occluder in B.
    radius = float(rng.uniform(0.05, 0.07))
    v = float(rng.uniform(0.9, 1.1))
    swap_t = float(rng.uniform(1.8, 2.4))
    critical = (swap_t - 0.2, swap_t + 0.2)

    # Occluder in front of camera: between eye and target (near y=0)
    occluder_y = 0.0
    occluder = {"half": (0.45, 0.02, 0.4), "pos": (0.0, occluder_y, 0.4)}

    lane_left_x = -0.22
    lane_right_x = 0.22
    y0 = -1.0

    def build():
        # Occluder wall
        wall = make_box(occluder["half"], occluder["pos"], 0.0, (0.2, 0.2, 0.2, 1.0))
        # Two identical balls
        b1 = make_sphere(radius, (lane_left_x, y0, radius), 1.0, (0.9, 0.7, 0.2, 1.0))
        b2 = make_sphere(radius, (lane_right_x, y0, radius), 1.0, (0.9, 0.7, 0.2, 1.0))
        for b in (b1, b2):
            set_dyn(b, friction=0.0, restitution=0.1)
            p.resetBaseVelocity(b, linearVelocity=(0.0, v, 0.0))
        return {"wall": wall, "b_leftstart": b1, "b_rightstart": b2}

    def intervene_B(t: float, step: int, ids: Dict[str, Any]):
        if abs(t - swap_t) < (1.0 / rcfg.sim_hz):
            p1, o1 = p.getBasePositionAndOrientation(ids["b_leftstart"])
            p2, o2 = p.getBasePositionAndOrientation(ids["b_rightstart"])
            # Swap x positions only (lane swap) while keeping y,z
            p.resetBasePositionAndOrientation(ids["b_leftstart"], (p2[0], p1[1], p1[2]), o1)
            p.resetBasePositionAndOrientation(ids["b_rightstart"], (p1[0], p2[1], p2[2]), o2)

    vA = os.path.join(out_videos_dir, "TM-ID", f"pair_{pair_id}_A.mp4")
    vB = os.path.join(out_videos_dir, "TM-ID", f"pair_{pair_id}_B.mp4")

    simulate_and_render(rcfg=rcfg, cam=cam, out_mp4=vA, build_scene=build, intervention=None)
    simulate_and_render(rcfg=rcfg, cam=cam, out_mp4=vB, build_scene=build, intervention=intervene_B)

    prompt = "Does the ball that STARTS in the LEFT lane also end in the LEFT lane?"
    rows = [
        qa_row(
            example_id=f"{pair_id}_A", pair_id=pair_id, pair_role="A",
            question_type="yesno", prompt=prompt, options=None, answer_index=1,
            video_path=os.path.relpath(vA, os.path.dirname(out_videos_dir)),
            module="TM-ID", critical_window=critical,
            extra={"seed": seed, "template": "TM-ID", "variant": "A",
                   "params": {"radius": radius, "v": v, "swap_t": swap_t},
                   "intervention": None},
        ),
        qa_row(
            example_id=f"{pair_id}_B", pair_id=pair_id, pair_role="B",
            question_type="yesno", prompt=prompt, options=None, answer_index=0,
            video_path=os.path.relpath(vB, os.path.dirname(out_videos_dir)),
            module="TM-ID", critical_window=critical,
            extra={"seed": seed, "template": "TM-ID", "variant": "B",
                   "params": {"radius": radius, "v": v, "swap_t": swap_t},
                   "intervention": {"type": "lane_swap", "t": swap_t}},
        ),
    ]
    meta = {"pair_id": pair_id, "module": "TM-ID", "critical_window": critical, "params": rows[0]["generation"]["params"]}
    return vA, vB, rows, meta


def gen_P_OCC(pair_id: str, out_videos_dir: str, seed: int, rcfg: RenderConfig, cam: CameraConfig):
    rng = np.random.default_rng(seed)

    radius = float(rng.uniform(0.05, 0.07))
    v = float(rng.uniform(0.9, 1.3))
    obstacle_present_in_A = True  # A has obstacle, B removes it
    critical = (1.0, 2.0)

    occluder = {"half": (0.5, 0.02, 0.5), "pos": (0.0, 0.0, 0.5)}
    obstacle = {"half": (0.12, 0.12, 0.12), "pos": (0.0, 0.35, 0.12)}

    def build(with_obstacle: bool):
        def _b():
            wall = make_box(occluder["half"], occluder["pos"], 0.0, (0.15, 0.15, 0.15, 1.0))
            ball = make_sphere(radius, (0.0, -1.0, radius), 1.0, (0.2, 0.8, 0.2, 1.0))
            set_dyn(ball, friction=0.0, restitution=0.9)
            p.resetBaseVelocity(ball, linearVelocity=(0.0, v, 0.0))
            obs_id = None
            if with_obstacle:
                obs_id = make_box(obstacle["half"], obstacle["pos"], 0.0, (0.5, 0.5, 0.5, 1.0))
            return {"wall": wall, "ball": ball, "obstacle": obs_id}
        return _b

    vA = os.path.join(out_videos_dir, "P-OCC", f"pair_{pair_id}_A.mp4")
    vB = os.path.join(out_videos_dir, "P-OCC", f"pair_{pair_id}_B.mp4")

    simulate_and_render(rcfg=rcfg, cam=cam, out_mp4=vA, build_scene=build(True), intervention=None)
    simulate_and_render(rcfg=rcfg, cam=cam, out_mp4=vB, build_scene=build(False), intervention=None)

    prompt = "Is there an obstacle behind the occluder that the ball COLLIDES with?"
    rows = [
        qa_row(
            example_id=f"{pair_id}_A", pair_id=pair_id, pair_role="A",
            question_type="yesno", prompt=prompt, options=None, answer_index=1,
            video_path=os.path.relpath(vA, os.path.dirname(out_videos_dir)),
            module="P-OCC", critical_window=critical,
            extra={"seed": seed, "template": "P-OCC", "variant": "A",
                   "params": {"radius": radius, "v": v, "obstacle": True},
                   "intervention": {"obstacle_present": True}},
        ),
        qa_row(
            example_id=f"{pair_id}_B", pair_id=pair_id, pair_role="B",
            question_type="yesno", prompt=prompt, options=None, answer_index=0,
            video_path=os.path.relpath(vB, os.path.dirname(out_videos_dir)),
            module="P-OCC", critical_window=critical,
            extra={"seed": seed, "template": "P-OCC", "variant": "B",
                   "params": {"radius": radius, "v": v, "obstacle": False},
                   "intervention": {"obstacle_present": False}},
        ),
    ]
    meta = {"pair_id": pair_id, "module": "P-OCC", "critical_window": critical, "params": rows[0]["generation"]["params"]}
    return vA, vB, rows, meta


def gen_R_COUNT(pair_id: str, out_videos_dir: str, seed: int, rcfg: RenderConfig, cam: CameraConfig):
    rng = np.random.default_rng(seed)

    radius = float(rng.uniform(0.05, 0.07))
    # We engineer 1 vs 2 wall hits by changing speed while keeping duration fixed.
    v1 = float(rng.uniform(0.6, 0.85))   # tends to yield 1 hit
    v2 = float(rng.uniform(1.05, 1.35))  # tends to yield 2 hits
    corridor_half = 0.55

    critical = (0.0, rcfg.duration_s)

    def build(v: float):
        def _b():
            # Two static walls in y
            wall1 = make_box((0.8, 0.02, 0.4), (0.0, -corridor_half, 0.4), 0.0, (0.25, 0.25, 0.25, 1.0))
            wall2 = make_box((0.8, 0.02, 0.4), (0.0, corridor_half, 0.4), 0.0, (0.25, 0.25, 0.25, 1.0))
            ball = make_sphere(radius, (0.0, 0.0, radius), 1.0, (0.9, 0.4, 0.9, 1.0))
            set_dyn(ball, friction=0.0, restitution=1.0)
            p.resetBaseVelocity(ball, linearVelocity=(0.0, v, 0.0))
            return {"ball": ball, "wall1": wall1, "wall2": wall2}
        return _b

    # Count contacts (to label), by doing a quick non-rendered sim in a separate run
    def count_hits(v: float) -> int:
        connect_and_reset(rcfg)
        ids = build(v)()
        steps = int(rcfg.duration_s * rcfg.sim_hz)
        hits = 0
        last_contact = False
        for step in range(steps):
            p.stepSimulation()
            cps1 = p.getContactPoints(bodyA=ids["ball"], bodyB=ids["wall1"])
            cps2 = p.getContactPoints(bodyA=ids["ball"], bodyB=ids["wall2"])
            contact = (len(cps1) > 0) or (len(cps2) > 0)
            if contact and not last_contact:
                hits += 1
            last_contact = contact
        disconnect()
        return hits

    hitsA = count_hits(v1)
    hitsB = count_hits(v2)
    # We want 1 vs 2 as much as possible; if not, keep labels but store actual.
    # Options allow 1/2/3; clamp >3 to 3 for simplicity.
    def clamp(h: int) -> int:
        return 3 if h >= 3 else max(1, h)

    ansA = clamp(hitsA) - 1
    ansB = clamp(hitsB) - 1

    vA = os.path.join(out_videos_dir, "R-COUNT", f"pair_{pair_id}_A.mp4")
    vB = os.path.join(out_videos_dir, "R-COUNT", f"pair_{pair_id}_B.mp4")

    simulate_and_render(rcfg=rcfg, cam=cam, out_mp4=vA, build_scene=build(v1), intervention=None)
    simulate_and_render(rcfg=rcfg, cam=cam, out_mp4=vB, build_scene=build(v2), intervention=None)

    prompt = "How many times does the ball HIT a wall during the video?"
    options = ["1", "2", "3+"]
    rows = [
        qa_row(
            example_id=f"{pair_id}_A", pair_id=pair_id, pair_role="A",
            question_type="mcq", prompt=prompt, options=options, answer_index=int(ansA),
            video_path=os.path.relpath(vA, os.path.dirname(out_videos_dir)),
            module="R-COUNT", critical_window=critical,
            extra={"seed": seed, "template": "R-COUNT", "variant": "A",
                   "params": {"radius": radius, "v": v1, "corridor_half": corridor_half, "hits": hitsA},
                   "intervention": {"v": v1}},
        ),
        qa_row(
            example_id=f"{pair_id}_B", pair_id=pair_id, pair_role="B",
            question_type="mcq", prompt=prompt, options=options, answer_index=int(ansB),
            video_path=os.path.relpath(vB, os.path.dirname(out_videos_dir)),
            module="R-COUNT", critical_window=critical,
            extra={"seed": seed, "template": "R-COUNT", "variant": "B",
                   "params": {"radius": radius, "v": v2, "corridor_half": corridor_half, "hits": hitsB},
                   "intervention": {"v": v2}},
        ),
    ]
    meta = {"pair_id": pair_id, "module": "R-COUNT", "critical_window": critical, "params": {"vA": v1, "vB": v2, "hitsA": hitsA, "hitsB": hitsB}}
    return vA, vB, rows, meta


def gen_PH_LATENT_FRICTION(pair_id: str, out_videos_dir: str, seed: int, rcfg: RenderConfig, cam: CameraConfig):
    rng = np.random.default_rng(seed)

    # Two blocks on two ground patches with different friction, pushed identically.
    mu_high = float(rng.uniform(0.6, 0.9))
    mu_low = float(rng.uniform(0.05, 0.2))
    v_push = float(rng.uniform(1.0, 1.4))
    push_t = float(rng.uniform(0.8, 1.2))
    critical = (push_t - 0.1, push_t + 2.5)

    patch_half = (0.5, 1.8, 0.02)
    left_patch_pos = (-0.35, 0.5, 0.02)
    right_patch_pos = (0.35, 0.5, 0.02)

    block_half = (0.07, 0.07, 0.07)
    left_block_pos = (-0.35, -0.8, 0.07)
    right_block_pos = (0.35, -0.8, 0.07)

    def build(mu_left: float, mu_right: float):
        def _b():
            # ground patches
            lp = make_box(patch_half, left_patch_pos, 0.0, (0.35, 0.35, 0.35, 1.0))
            rp = make_box(patch_half, right_patch_pos, 0.0, (0.35, 0.35, 0.35, 1.0))
            set_dyn(lp, friction=mu_left, restitution=0.0)
            set_dyn(rp, friction=mu_right, restitution=0.0)

            # blocks
            lb = make_box(block_half, left_block_pos, 1.0, (0.8, 0.6, 0.2, 1.0))
            rb = make_box(block_half, right_block_pos, 1.0, (0.8, 0.6, 0.2, 1.0))
            set_dyn(lb, friction=0.8, restitution=0.0)
            set_dyn(rb, friction=0.8, restitution=0.0)

            return {"lp": lp, "rp": rp, "lb": lb, "rb": rb}
        return _b

    def push_at(t_push: float):
        def _fn(t: float, step: int, ids: Dict[str, Any]):
            if abs(t - t_push) < (1.0 / rcfg.sim_hz):
                p.resetBaseVelocity(ids["lb"], linearVelocity=(0.0, v_push, 0.0))
                p.resetBaseVelocity(ids["rb"], linearVelocity=(0.0, v_push, 0.0))
        return _fn

    # A: left high friction; B: right high friction
    vA = os.path.join(out_videos_dir, "PH-LATENT-FRICTION", f"pair_{pair_id}_A.mp4")
    vB = os.path.join(out_videos_dir, "PH-LATENT-FRICTION", f"pair_{pair_id}_B.mp4")

    simulate_and_render(rcfg=rcfg, cam=cam, out_mp4=vA, build_scene=build(mu_high, mu_low), intervention=push_at(push_t))
    simulate_and_render(rcfg=rcfg, cam=cam, out_mp4=vB, build_scene=build(mu_low, mu_high), intervention=push_at(push_t))

    prompt = "Which surface has HIGHER friction?"
    options = ["Left", "Right", "Same"]
    rows = [
        qa_row(
            example_id=f"{pair_id}_A", pair_id=pair_id, pair_role="A",
            question_type="mcq", prompt=prompt, options=options, answer_index=0,
            video_path=os.path.relpath(vA, os.path.dirname(out_videos_dir)),
            module="PH-LATENT-FRICTION", critical_window=critical,
            extra={"seed": seed, "template": "PH-LATENT-FRICTION", "variant": "A",
                   "params": {"mu_high": mu_high, "mu_low": mu_low, "push_t": push_t, "v_push": v_push},
                   "intervention": {"mu_left": mu_high, "mu_right": mu_low}},
        ),
        qa_row(
            example_id=f"{pair_id}_B", pair_id=pair_id, pair_role="B",
            question_type="mcq", prompt=prompt, options=options, answer_index=1,
            video_path=os.path.relpath(vB, os.path.dirname(out_videos_dir)),
            module="PH-LATENT-FRICTION", critical_window=critical,
            extra={"seed": seed, "template": "PH-LATENT-FRICTION", "variant": "B",
                   "params": {"mu_high": mu_high, "mu_low": mu_low, "push_t": push_t, "v_push": v_push},
                   "intervention": {"mu_left": mu_low, "mu_right": mu_high}},
        ),
    ]
    meta = {"pair_id": pair_id, "module": "PH-LATENT-FRICTION", "critical_window": critical,
            "params": {"mu_high": mu_high, "mu_low": mu_low, "push_t": push_t, "v_push": v_push}}
    return vA, vB, rows, meta


def gen_PH_LATENT_MASS(pair_id: str, out_videos_dir: str, seed: int, rcfg: RenderConfig, cam: CameraConfig):
    rng = np.random.default_rng(seed)

    m_heavy = float(rng.uniform(2.0, 4.0))
    m_light = float(rng.uniform(0.8, 1.4))
    force = float(rng.uniform(10.0, 18.0))  # N
    force_dur = float(rng.uniform(0.08, 0.14))
    push_t = float(rng.uniform(0.8, 1.2))
    critical = (push_t - 0.1, push_t + 1.0)

    block_half = (0.07, 0.07, 0.07)
    left_pos = (-0.25, -0.8, 0.07)
    right_pos = (0.25, -0.8, 0.07)

    def build(m_left: float, m_right: float):
        def _b():
            # Low-friction surface
            # (plane already exists; just keep block friction tiny)
            lb = make_box(block_half, left_pos, m_left, (0.75, 0.75, 0.75, 1.0))
            rb = make_box(block_half, right_pos, m_right, (0.75, 0.75, 0.75, 1.0))
            set_dyn(lb, friction=0.02, restitution=0.0)
            set_dyn(rb, friction=0.02, restitution=0.0)
            return {"lb": lb, "rb": rb}
        return _b

    def push_fn(t0: float):
        def _fn(t: float, step: int, ids: Dict[str, Any]):
            if t0 <= t < t0 + force_dur:
                # Apply same force to both blocks in +y direction
                p.applyExternalForce(ids["lb"], -1, forceObj=(0.0, force, 0.0), posObj=(0,0,0), flags=p.WORLD_FRAME)
                p.applyExternalForce(ids["rb"], -1, forceObj=(0.0, force, 0.0), posObj=(0,0,0), flags=p.WORLD_FRAME)
        return _fn

    vA = os.path.join(out_videos_dir, "PH-LATENT-MASS", f"pair_{pair_id}_A.mp4")
    vB = os.path.join(out_videos_dir, "PH-LATENT-MASS", f"pair_{pair_id}_B.mp4")

    # A: left heavy; B: right heavy
    simulate_and_render(rcfg=rcfg, cam=cam, out_mp4=vA, build_scene=build(m_heavy, m_light), intervention=push_fn(push_t))
    simulate_and_render(rcfg=rcfg, cam=cam, out_mp4=vB, build_scene=build(m_light, m_heavy), intervention=push_fn(push_t))

    prompt = "Which block is HEAVIER?"
    options = ["Left", "Right", "Same"]
    rows = [
        qa_row(
            example_id=f"{pair_id}_A", pair_id=pair_id, pair_role="A",
            question_type="mcq", prompt=prompt, options=options, answer_index=0,
            video_path=os.path.relpath(vA, os.path.dirname(out_videos_dir)),
            module="PH-LATENT-MASS", critical_window=critical,
            extra={"seed": seed, "template": "PH-LATENT-MASS", "variant": "A",
                   "params": {"m_heavy": m_heavy, "m_light": m_light, "force": force, "force_dur": force_dur, "push_t": push_t},
                   "intervention": {"m_left": m_heavy, "m_right": m_light}},
        ),
        qa_row(
            example_id=f"{pair_id}_B", pair_id=pair_id, pair_role="B",
            question_type="mcq", prompt=prompt, options=options, answer_index=1,
            video_path=os.path.relpath(vB, os.path.dirname(out_videos_dir)),
            module="PH-LATENT-MASS", critical_window=critical,
            extra={"seed": seed, "template": "PH-LATENT-MASS", "variant": "B",
                   "params": {"m_heavy": m_heavy, "m_light": m_light, "force": force, "force_dur": force_dur, "push_t": push_t},
                   "intervention": {"m_left": m_light, "m_right": m_heavy}},
        ),
    ]
    meta = {"pair_id": pair_id, "module": "PH-LATENT-MASS", "critical_window": critical,
            "params": {"m_heavy": m_heavy, "m_light": m_light, "force": force, "force_dur": force_dur, "push_t": push_t}}
    return vA, vB, rows, meta


def gen_PH_INVAR_ENERGY(pair_id: str, out_videos_dir: str, seed: int, rcfg: RenderConfig, cam: CameraConfig):
    rng = np.random.default_rng(seed)

    radius = float(rng.uniform(0.05, 0.07))
    restitution = float(rng.uniform(0.55, 0.75))
    drop_h = float(rng.uniform(0.9, 1.2))
    boost_t = float(rng.uniform(1.5, 2.5))
    boost_dv = float(rng.uniform(0.8, 1.3))
    critical = (boost_t - 0.15, boost_t + 0.15)

    def build():
        ball = make_sphere(radius, (0.0, 0.0, drop_h), 1.0, (0.2, 0.6, 0.95, 1.0))
        set_dyn(ball, friction=0.05, restitution=restitution)
        return {"ball": ball}

    def intervene_B(t: float, step: int, ids: Dict[str, Any]):
        # Add upward velocity at boost_t (epsilon energy gain).
        if abs(t - boost_t) < (1.0 / rcfg.sim_hz):
            v_lin, v_ang = p.getBaseVelocity(ids["ball"])
            p.resetBaseVelocity(ids["ball"], linearVelocity=(v_lin[0], v_lin[1], v_lin[2] + boost_dv))

    vA = os.path.join(out_videos_dir, "PH-INVAR-ENERGY", f"pair_{pair_id}_A.mp4")
    vB = os.path.join(out_videos_dir, "PH-INVAR-ENERGY", f"pair_{pair_id}_B.mp4")

    simulate_and_render(rcfg=rcfg, cam=cam, out_mp4=vA, build_scene=build, intervention=None)
    simulate_and_render(rcfg=rcfg, cam=cam, out_mp4=vB, build_scene=build, intervention=intervene_B)

    prompt = "Does the ball gain energy (speed/height) without any external cause?"
    rows = [
        qa_row(
            example_id=f"{pair_id}_A", pair_id=pair_id, pair_role="A",
            question_type="yesno", prompt=prompt, options=None, answer_index=0,
            video_path=os.path.relpath(vA, os.path.dirname(out_videos_dir)),
            module="PH-INVAR-ENERGY", critical_window=critical,
            extra={"seed": seed, "template": "PH-INVAR-ENERGY", "variant": "A",
                   "params": {"radius": radius, "restitution": restitution, "drop_h": drop_h, "boost_t": boost_t, "boost_dv": boost_dv},
                   "intervention": None},
        ),
        qa_row(
            example_id=f"{pair_id}_B", pair_id=pair_id, pair_role="B",
            question_type="yesno", prompt=prompt, options=None, answer_index=1,
            video_path=os.path.relpath(vB, os.path.dirname(out_videos_dir)),
            module="PH-INVAR-ENERGY", critical_window=critical,
            extra={"seed": seed, "template": "PH-INVAR-ENERGY", "variant": "B",
                   "params": {"radius": radius, "restitution": restitution, "drop_h": drop_h, "boost_t": boost_t, "boost_dv": boost_dv},
                   "intervention": {"type": "velocity_boost", "t": boost_t, "dv_z": boost_dv}},
        ),
    ]
    meta = {"pair_id": pair_id, "module": "PH-INVAR-ENERGY", "critical_window": critical, "params": rows[0]["generation"]["params"]}
    return vA, vB, rows, meta


def gen_C_CHAIN(pair_id: str, out_videos_dir: str, seed: int, rcfg: RenderConfig, cam: CameraConfig):
    rng = np.random.default_rng(seed)

    # A: domino spacing small (chain completes)
    # B: domino spacing larger (chain breaks before last)
    spacing_A = float(rng.uniform(0.12, 0.16))
    spacing_B = float(rng.uniform(0.22, 0.28))
    critical = (1.0, 4.5)

    domino_half = (0.02, 0.06, 0.12)  # thin, tall
    base_x = 0.0
    base_y = 0.0
    z = domino_half[2]

    ball_r = 0.06
    ball_v = float(rng.uniform(1.0, 1.4))

    def build(spacing: float):
        def _b():
            # Three dominoes
            d1 = make_box(domino_half, (base_x, base_y + 0.4, z), 0.5, (0.7, 0.5, 0.3, 1.0))
            d2 = make_box(domino_half, (base_x, base_y + 0.4 + spacing, z), 0.5, (0.7, 0.5, 0.3, 1.0))
            d3 = make_box(domino_half, (base_x, base_y + 0.4 + 2*spacing, z), 0.5, (0.7, 0.5, 0.3, 1.0))
            for d in (d1, d2, d3):
                set_dyn(d, friction=0.6, restitution=0.05)
            # Ball that hits d1
            ball = make_sphere(ball_r, (0.0, -0.6, ball_r), 1.0, (0.2, 0.9, 0.9, 1.0))
            set_dyn(ball, friction=0.0, restitution=0.8)
            p.resetBaseVelocity(ball, linearVelocity=(0.0, ball_v, 0.0))
            return {"d1": d1, "d2": d2, "d3": d3, "ball": ball}
        return _b

    vA = os.path.join(out_videos_dir, "C-CHAIN", f"pair_{pair_id}_A.mp4")
    vB = os.path.join(out_videos_dir, "C-CHAIN", f"pair_{pair_id}_B.mp4")

    simulate_and_render(rcfg=rcfg, cam=cam, out_mp4=vA, build_scene=build(spacing_A), intervention=None)
    simulate_and_render(rcfg=rcfg, cam=cam, out_mp4=vB, build_scene=build(spacing_B), intervention=None)

    prompt = "Does the LAST domino fall down by the end of the video?"
    rows = [
        qa_row(
            example_id=f"{pair_id}_A", pair_id=pair_id, pair_role="A",
            question_type="yesno", prompt=prompt, options=None, answer_index=1,
            video_path=os.path.relpath(vA, os.path.dirname(out_videos_dir)),
            module="C-CHAIN", critical_window=critical,
            extra={"seed": seed, "template": "C-CHAIN", "variant": "A",
                   "params": {"spacing": spacing_A, "ball_v": ball_v},
                   "intervention": {"spacing": spacing_A}},
        ),
        qa_row(
            example_id=f"{pair_id}_B", pair_id=pair_id, pair_role="B",
            question_type="yesno", prompt=prompt, options=None, answer_index=0,
            video_path=os.path.relpath(vB, os.path.dirname(out_videos_dir)),
            module="C-CHAIN", critical_window=critical,
            extra={"seed": seed, "template": "C-CHAIN", "variant": "B",
                   "params": {"spacing": spacing_B, "ball_v": ball_v},
                   "intervention": {"spacing": spacing_B}},
        ),
    ]
    meta = {"pair_id": pair_id, "module": "C-CHAIN", "critical_window": critical,
            "params": {"spacing_A": spacing_A, "spacing_B": spacing_B, "ball_v": ball_v}}
    return vA, vB, rows, meta


def gen_PH_COUNTER(pair_id: str, out_videos_dir: str, seed: int, rcfg: RenderConfig, cam: CameraConfig):
    rng = np.random.default_rng(seed)

    # Two worlds, both show the ball hitting the target, but wedge counterfactual differs.
    # A: wedge is necessary to redirect the ball into target -> removing wedge => MISS (answer NO)
    # B: ball reaches target without wedge (wedge is irrelevant) -> removing wedge => HIT (answer YES)

    radius = float(rng.uniform(0.05, 0.07))
    v = float(rng.uniform(1.0, 1.4))
    wedge_half = (0.10, 0.04, 0.04)
    target_half = (0.06, 0.06, 0.12)

    # Keep target fixed
    target_pos = (0.32, 0.85, target_half[2])

    # World A: ball starts centerline and would miss target (x=0); wedge redirects to x=0.32
    ballA_start = (0.0, -0.9, radius)
    wedgeA_pos = (0.12, 0.35, wedge_half[2])
    wedgeA_yaw = math.radians(35)

    # World B: ball starts near target line; wedge placed away so it doesn't matter
    ballB_start = (0.30 + float(rng.uniform(-0.03, 0.03)), -0.9, radius)
    wedgeB_pos = (-0.25, 0.35, wedge_half[2])
    wedgeB_yaw = math.radians(-10)

    critical = (0.8, 4.0)

    def build(ball_start: Tuple[float, float, float], wedge_pos: Tuple[float, float, float], wedge_yaw: float):
        def _b():
            # Target post
            target = make_box(target_half, target_pos, 0.0, (0.9, 0.2, 0.2, 1.0))
            # Wedge (a thin box rotated)
            wedge = make_box(wedge_half, wedge_pos, 0.0, (0.2, 0.2, 0.2, 1.0), orn=quat_from_euler(0, 0, wedge_yaw))
            # Ball
            ball = make_sphere(radius, ball_start, 1.0, (0.2, 0.8, 0.2, 1.0))
            set_dyn(ball, friction=0.0, restitution=0.9)
            p.resetBaseVelocity(ball, linearVelocity=(0.0, v, 0.0))
            return {"ball": ball, "wedge": wedge, "target": target}
        return _b

    # Counterfactual evaluator: run the same build but omit wedge; detect whether ball contacts target
    def counterfactual_hits_target(ball_start: Tuple[float, float, float]) -> bool:
        connect_and_reset(rcfg)
        # Only target + ball (no wedge)
        target = make_box(target_half, target_pos, 0.0, (0.9, 0.2, 0.2, 1.0))
        ball = make_sphere(radius, ball_start, 1.0, (0.2, 0.8, 0.2, 1.0))
        set_dyn(ball, friction=0.0, restitution=0.9)
        p.resetBaseVelocity(ball, linearVelocity=(0.0, v, 0.0))
        steps = int(rcfg.duration_s * rcfg.sim_hz)

        hit = False
        for step in range(steps):
            p.stepSimulation()
            cps = p.getContactPoints(bodyA=ball, bodyB=target)
            if cps:
                hit = True
                break
        disconnect()
        return hit

    # Compute the ground-truth counterfactual answers
    # Question: "If the wedge were removed at the start, would the ball still hit the target?"
    cfA = counterfactual_hits_target(ballA_start)  # expected False
    cfB = counterfactual_hits_target(ballB_start)  # expected True

    vA = os.path.join(out_videos_dir, "PH-COUNTER", f"pair_{pair_id}_A.mp4")
    vB = os.path.join(out_videos_dir, "PH-COUNTER", f"pair_{pair_id}_B.mp4")

    simulate_and_render(rcfg=rcfg, cam=cam, out_mp4=vA, build_scene=build(ballA_start, wedgeA_pos, wedgeA_yaw), intervention=None)
    simulate_and_render(rcfg=rcfg, cam=cam, out_mp4=vB, build_scene=build(ballB_start, wedgeB_pos, wedgeB_yaw), intervention=None)

    prompt = "If the wedge were removed at the start, would the ball still HIT the red target?"
    rows = [
        qa_row(
            example_id=f"{pair_id}_A", pair_id=pair_id, pair_role="A",
            question_type="yesno", prompt=prompt, options=None, answer_index=(1 if cfA else 0),
            video_path=os.path.relpath(vA, os.path.dirname(out_videos_dir)),
            module="PH-COUNTER", critical_window=critical,
            extra={"seed": seed, "template": "PH-COUNTER", "variant": "A",
                   "params": {"radius": radius, "v": v, "ball_start": ballA_start, "wedge_pos": wedgeA_pos, "wedge_yaw": wedgeA_yaw, "cf_hits": cfA},
                   "intervention": {"counterfactual": "remove_wedge"}},
        ),
        qa_row(
            example_id=f"{pair_id}_B", pair_id=pair_id, pair_role="B",
            question_type="yesno", prompt=prompt, options=None, answer_index=(1 if cfB else 0),
            video_path=os.path.relpath(vB, os.path.dirname(out_videos_dir)),
            module="PH-COUNTER", critical_window=critical,
            extra={"seed": seed, "template": "PH-COUNTER", "variant": "B",
                   "params": {"radius": radius, "v": v, "ball_start": ballB_start, "wedge_pos": wedgeB_pos, "wedge_yaw": wedgeB_yaw, "cf_hits": cfB},
                   "intervention": {"counterfactual": "remove_wedge"}},
        ),
    ]
    meta = {"pair_id": pair_id, "module": "PH-COUNTER", "critical_window": critical,
            "params": {"cfA": cfA, "cfB": cfB, "ballA_start": ballA_start, "ballB_start": ballB_start}}
    return vA, vB, rows, meta


# Registry
GEN = {
    "TR-ALI": gen_TR_ALI,
    "TR-ORDER": gen_TR_ORDER,
    "TM-ID": gen_TM_ID,
    "P-OCC": gen_P_OCC,
    "R-COUNT": gen_R_COUNT,
    "PH-LATENT-FRICTION": gen_PH_LATENT_FRICTION,
    "PH-LATENT-MASS": gen_PH_LATENT_MASS,
    "PH-INVAR-ENERGY": gen_PH_INVAR_ENERGY,
    "C-CHAIN": gen_C_CHAIN,
    "PH-COUNTER": gen_PH_COUNTER,
}


# ----------------------------
# Main
# ----------------------------

def main():
    args = parse_args()

    taxonomies = [t.strip() for t in args.taxonomies.split(",") if t.strip()]
    for t in taxonomies:
        if t not in GEN:
            raise SystemExit(f"Unknown taxonomy '{t}'. Known: {sorted(GEN.keys())}")

    out_root = args.out_root
    out_videos = os.path.join(out_root, "videos")
    out_ann = os.path.join(out_root, "annotations")
    out_pair_meta = os.path.join(out_ann, "pair_meta")
    os.makedirs(out_videos, exist_ok=True)
    os.makedirs(out_pair_meta, exist_ok=True)

    jsonl_path = os.path.join(out_ann, "miragev.jsonl")
    # If re-running, append; if you want fresh output, delete the file first.
    if not os.path.exists(jsonl_path):
        open(jsonl_path, "w", encoding="utf-8").close()

    rcfg = RenderConfig(
        fps=args.fps,
        sim_hz=args.sim_hz,
        width=args.width,
        height=args.height,
        duration_s=args.duration_s,
    )
    cam = CameraConfig()

    all_rows: List[Dict[str, Any]] = []
    for module in taxonomies:
        gen_fn = GEN[module]
        for i in tqdm(range(args.pairs_per_taxonomy), desc=f"Generating {module}"):
            pair_id = f"{module}_{i:04d}"
            seed = int(args.seed + (hash(pair_id) % 1_000_000_000))

            vA, vB, rows, meta = gen_fn(pair_id, out_videos, seed, rcfg, cam)

            # Save per-pair meta for debugging/analysis
            write_json(os.path.join(out_pair_meta, f"pair_{pair_id}.json"), meta)

            append_jsonl(jsonl_path, rows)

    print(f"Done. Wrote JSONL: {jsonl_path}")
    print(f"Videos under: {out_videos}")


if __name__ == "__main__":
    main()
