#!/usr/bin/env python3
"""
Replay a converted BridgeSim/MetaDrive scenario and export first-person and/or
third-person videos.

Supports scenario directory input:
    /path/to/sd_xxxxxxxx

And direct pickle input:
    /path/to/sd_xxxxxxxx/sd_xxxxxxxx_0/sd_xxxxxxxx.pkl
"""

import argparse
import logging
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.envs.scenario_env import ScenarioEnv
from metadrive.policy.replay_policy import ReplayEgoCarPolicy


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "bridgesim" / "results"


CAMERA_PARAMS = {
    "CAM_F0": {
        "distortion": np.array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
        "intrinsics": np.array([[1.545e03, 0.0, 9.600e02], [0.0, 1.545e03, 5.600e02], [0.0, 0.0, 1.0]]),
        "sensor2lidar_rotation": np.array(
            [[-0.00785972, -0.02271912, 0.99971099], [-0.99994262, 0.00745516, -0.00769211],
             [-0.00727825, -0.99971409, -0.02277642]]
        ),
        "sensor2lidar_translation": np.array([1.65506747, -0.01168732, 1.49112208]),
    },
    "CAM_L0": {
        "distortion": np.array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
        "intrinsics": np.array([[1.545e03, 0.0, 9.600e02], [0.0, 1.545e03, 5.600e02], [0.0, 0.0, 1.0]]),
        "sensor2lidar_rotation": np.array(
            [[0.81776776, -0.0057693, 0.57551942], [-0.57553938, -0.01377628, 0.81765802],
             [0.0032112, -0.99988846, -0.01458626]]
        ),
        "sensor2lidar_translation": np.array([1.63069485, 0.11956747, 1.48117884]),
    },
    "CAM_L1": {
        "distortion": np.array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
        "intrinsics": np.array([[1.545e03, 0.0, 9.600e02], [0.0, 1.545e03, 5.600e02], [0.0, 0.0, 1.0]]),
        "sensor2lidar_rotation": np.array(
            [[0.93120104, 0.00261563, -0.36449662], [0.36447127, -0.02048653, 0.93098926],
             [-0.00503215, -0.99978671, -0.0200304]]
        ),
        "sensor2lidar_translation": np.array([1.29939471, 0.63819702, 1.36736822]),
    },
    "CAM_L2": {
        "distortion": np.array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
        "intrinsics": np.array([[1.545e03, 0.0, 9.600e02], [0.0, 1.545e03, 5.600e02], [0.0, 0.0, 1.0]]),
        "sensor2lidar_rotation": np.array(
            [[0.63520782, 0.01497516, -0.77219607], [0.77232489, -0.00580669, 0.63520119],
             [0.00502834, -0.99987101, -0.01525415]]
        ),
        "sensor2lidar_translation": np.array([-0.49561003, 0.54750373, 1.3472672]),
    },
    "CAM_R0": {
        "distortion": np.array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
        "intrinsics": np.array([[1.545e03, 0.0, 9.600e02], [0.0, 1.545e03, 5.600e02], [0.0, 0.0, 1.0]]),
        "sensor2lidar_rotation": np.array(
            [[-0.82454901, 0.01165722, 0.56567043], [-0.56528395, 0.02532491, -0.82450755],
             [-0.02393702, -0.9996113, -0.01429199]]
        ),
        "sensor2lidar_translation": np.array([1.61828343, -0.15532203, 1.49007665]),
    },
    "CAM_R1": {
        "distortion": np.array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
        "intrinsics": np.array([[1.545e03, 0.0, 9.600e02], [0.0, 1.545e03, 5.600e02], [0.0, 0.0, 1.0]]),
        "sensor2lidar_rotation": np.array(
            [[-0.92684778, 0.02177016, -0.37480562], [0.37497631, 0.00421964, -0.92702479],
             [-0.01859993, -0.9997541, -0.01207426]]
        ),
        "sensor2lidar_translation": np.array([1.27299407, -0.60973112, 1.37217911]),
    },
    "CAM_R2": {
        "distortion": np.array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
        "intrinsics": np.array([[1.545e03, 0.0, 9.600e02], [0.0, 1.545e03, 5.600e02], [0.0, 0.0, 1.0]]),
        "sensor2lidar_rotation": np.array(
            [[-0.62253245, 0.03706878, -0.78171558], [0.78163434, -0.02000083, -0.62341618],
             [-0.03874424, -0.99911254, -0.01652307]]
        ),
        "sensor2lidar_translation": np.array([-0.48771615, -0.493167, 1.35027683]),
    },
    "CAM_B0": {
        "distortion": np.array([-0.356123, 0.172545, -0.00213, 0.000464, -0.05231]),
        "intrinsics": np.array([[1.545e03, 0.0, 9.600e02], [0.0, 1.545e03, 5.600e02], [0.0, 0.0, 1.0]]),
        "sensor2lidar_rotation": np.array(
            [[0.00802542, 0.01047463, -0.99991293], [0.99989075, -0.01249671, 0.00789433],
             [-0.01241293, -0.99986705, -0.01057378]]
        ),
        "sensor2lidar_translation": np.array([-0.47463312, 0.02368552, 1.4341838]),
    },
}


ALL_CAM_IDS = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0"]


def rotation_matrix_to_euler_angles(rotation):
    sy = np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arctan2(-rotation[2, 0], sy)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        pitch = np.arctan2(-rotation[2, 0], sy)
        yaw = 0.0
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def calculate_fov_from_intrinsics(intrinsics, image_width, image_height):
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    fov_horizontal = 2 * np.arctan(image_width / (2 * fx))
    fov_vertical = 2 * np.arctan(image_height / (2 * fy))
    return np.degrees(fov_horizontal), np.degrees(fov_vertical)


def get_metadrive_cam_params(cam_params, image_width=1920, image_height=1120):
    translation = cam_params["sensor2lidar_translation"]
    rotation = cam_params["sensor2lidar_rotation"]
    intrinsics = cam_params["intrinsics"]

    x_md = -translation[1]
    y_md = translation[0]
    z_md = translation[2]

    roll, pitch, yaw = rotation_matrix_to_euler_angles(rotation)
    yaw_md = yaw + 90.0
    pitch_md = pitch
    roll_md = 0.0

    fov_h, _ = calculate_fov_from_intrinsics(intrinsics, image_width, image_height)

    return {
        "pos": (x_md, y_md, z_md),
        "hpr": (yaw_md, pitch_md, roll_md),
        "fov": fov_h,
        "width": image_width,
        "height": image_height,
    }


def resolve_scenario_paths(input_path: str):
    path = Path(input_path).resolve()
    if path.is_file() and path.suffix == ".pkl":
        pkl_path = path
        scenario_dir = path.parent.parent
        scenario_name = scenario_dir.name
        return scenario_dir, pkl_path, scenario_name

    if path.is_dir():
        scenario_name = path.name
        scenario_subfolder = path / f"{scenario_name}_0"
        pkl_candidates = sorted(scenario_subfolder.glob("*.pkl"))
        if not pkl_candidates:
            raise FileNotFoundError(f"No .pkl found in {scenario_subfolder}")
        return path, pkl_candidates[0], scenario_name

    raise FileNotFoundError(f"Scenario path not found: {input_path}")


def create_video_writer(output_path: Path, fps: int, width: int, height: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    return writer


def ensure_numpy_image(frame):
    """Convert cv2/cupy-like outputs to a CPU numpy array."""
    if hasattr(frame, "get"):
        frame = frame.get()
    return np.asarray(frame)


def resize_for_preview(frame, max_width=1280, max_height=720):
    """Resize preview frames to fit comfortably on screen while preserving aspect ratio."""
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return frame
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def annotate_preview(frame, title, frame_idx, total_frames):
    """Overlay a compact HUD on preview windows."""
    preview = frame.copy()
    text = f"{title}  frame {frame_idx + 1}/{total_frames}  q/esc: quit"
    cv2.putText(preview, text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(preview, text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 1, cv2.LINE_AA)
    return preview


def show_preview(window_name, frame, frame_idx, total_frames, max_width=1280, max_height=720):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    preview = annotate_preview(frame, window_name, frame_idx, total_frames)
    preview = resize_for_preview(preview, max_width=max_width, max_height=max_height)
    cv2.imshow(window_name, preview)


def should_quit_preview():
    key = cv2.waitKey(1) & 0xFF
    return key in (27, ord("q"))


def replay_views(
    scenario_path: str,
    output_root: str,
    view: str,
    fps: int,
    max_frames: int = None,
    cameras=None,
    first_person_size=(1920, 1120),
    third_person_size=(1280, 720),
    live: bool = True,
    save_video: bool = True,
    preview_max_width: int = 1280,
    preview_max_height: int = 720,
):
    scenario_dir, scenario_pkl, scenario_name = resolve_scenario_paths(scenario_path)
    logger.info("Loading scenario: %s", scenario_pkl)

    with open(scenario_pkl, "rb") as f:
        scenario = pickle.load(f)

    metadata = scenario.get("metadata", {})
    frame_info = metadata.get("frame_info", [])
    log_name = metadata.get("log_name", scenario_name)
    scenario_length = int(scenario.get("length", len(frame_info)))

    if not frame_info:
        raise RuntimeError("No frame_info in scenario metadata. Re-convert with updated converter.")

    wants_first_person = view in {"first_person", "both", "all"}
    wants_third_person = view in {"third_person", "both", "all"}
    wants_top_down = view in {"top_down", "all"}

    cam_ids = cameras if cameras else ["CAM_F0"]
    fp_width, fp_height = first_person_size
    tp_width, tp_height = third_person_size

    md_cam_configs = {}
    if wants_first_person:
        for cam_id in cam_ids:
            if cam_id not in CAMERA_PARAMS:
                raise ValueError(f"Unknown camera id: {cam_id}")
            md_cam_configs[cam_id] = get_metadrive_cam_params(CAMERA_PARAMS[cam_id], fp_width, fp_height)

    if not live and not save_video:
        raise ValueError("At least one of live preview or video saving must be enabled.")

    env_config = {
        "use_render": wants_third_person,
        "render_pipeline": False,
        # MainCamera is more stable on CPU in this mixed replay/export use case.
        "image_on_cuda": False,
        "agent_policy": ReplayEgoCarPolicy,
        "manual_control": False,
        "num_scenarios": 1,
        "horizon": min(scenario_length, max_frames) if max_frames else scenario_length,
        "data_directory": str(scenario_dir),
        "image_observation": True,
        "show_interface": False,
        "show_logo": False,
        "show_fps": False,
        "window_size": third_person_size,
        "vehicle_config": {
            "image_source": "rgb_camera",
            "show_navi_mark": False,
        },
        "sensors": {
            "rgb_camera": (RGBCamera, fp_width, fp_height),
        },
    }

    output_dir = Path(output_root).resolve() / log_name
    if save_video:
        output_dir.mkdir(parents=True, exist_ok=True)

    env = ScenarioEnv(env_config)
    fp_writers = {}
    tp_writer = None
    td_writer = None

    try:
        env.reset(seed=0)
        if wants_third_person:
            env.switch_to_third_person_view()
        ego = env.agent
        sensor = env.engine.get_sensor("rgb_camera") if wants_first_person else None

        render_horizon = env_config["horizon"]
        logger.info("Rendering %s frames...", render_horizon)
        stop_requested = False

        for current_step in tqdm(range(render_horizon), desc="Replaying"):
            frame_start = time.perf_counter()
            env.step([0, 0])

            if live and wants_third_person:
                env.render(text={"frame": f"{current_step + 1}/{render_horizon}", "view": "third_person"})

            if wants_first_person:
                for cam_id, cfg in md_cam_configs.items():
                    if hasattr(sensor.lens, "setFov"):
                        sensor.lens.setFov(70)
                    img = sensor.perceive(
                        to_float=False,
                        new_parent_node=ego.origin,
                        position=cfg["pos"],
                        hpr=cfg["hpr"],
                    )
                    img = ensure_numpy_image(img)

                    if save_video and cam_id not in fp_writers:
                        fp_writers[cam_id] = create_video_writer(
                            output_dir / f"first_person_{cam_id}.mp4",
                            fps=fps,
                            width=img.shape[1],
                            height=img.shape[0],
                        )
                    if save_video:
                        fp_writers[cam_id].write(img)

                    if live:
                        show_preview(
                            window_name=f"First Person {cam_id}",
                            frame=img,
                            frame_idx=current_step,
                            total_frames=render_horizon,
                            max_width=preview_max_width,
                            max_height=preview_max_height,
                        )

            if wants_third_person:
                frame = env.main_camera.perceive(to_float=False)
                if frame is not None:
                    frame = ensure_numpy_image(frame)
                    if save_video and tp_writer is None:
                        tp_writer = create_video_writer(
                            output_dir / "third_person.mp4",
                            fps=fps,
                            width=frame.shape[1],
                            height=frame.shape[0],
                        )
                    if save_video:
                        tp_writer.write(frame)

            if wants_top_down:
                top_down_frame = env.render(
                    mode="top_down",
                    window=live,
                    screen_record=False,
                    target_agent_heading_up=True,
                    semantic_map=True,
                    draw_target_vehicle_trajectory=True,
                    scaling=8.0,
                    film_size=(3000, 3000),
                    screen_size=(900, 900),
                    num_stack=1,
                    history_smooth=0,
                    text={"frame": f"{current_step + 1}/{render_horizon}", "view": "top_down"},
                )
                if top_down_frame is not None:
                    top_down_frame = ensure_numpy_image(top_down_frame)
                    if save_video and td_writer is None:
                        td_writer = create_video_writer(
                            output_dir / "top_down.mp4",
                            fps=fps,
                            width=top_down_frame.shape[1],
                            height=top_down_frame.shape[0],
                        )
                    if save_video:
                        td_writer.write(top_down_frame)

            if live and wants_first_person and should_quit_preview():
                stop_requested = True

            target_dt = 1.0 / max(fps, 1)
            remaining = target_dt - (time.perf_counter() - frame_start)
            if remaining > 0:
                time.sleep(remaining)

            if stop_requested:
                logger.info("Replay stopped early by user input.")
                break

        if save_video:
            logger.info("Done! Outputs saved to: %s", output_dir)
            if wants_first_person:
                for cam_id in fp_writers:
                    logger.info("First-person video: %s", output_dir / f"first_person_{cam_id}.mp4")
            if tp_writer is not None:
                logger.info("Third-person video: %s", output_dir / "third_person.mp4")
            elif wants_third_person:
                logger.warning("Third-person video was requested but no frames were captured.")
            if td_writer is not None:
                logger.info("Top-down video: %s", output_dir / "top_down.mp4")
            elif wants_top_down:
                logger.warning("Top-down video was requested but no frames were captured.")
        else:
            logger.info("Done! Live replay finished without saving videos.")

    finally:
        for writer in fp_writers.values():
            writer.release()
        if tp_writer is not None:
            tp_writer.release()
        if td_writer is not None:
            td_writer.release()
        cv2.destroyAllWindows()
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Replay a converted BridgeSim scenario with live preview and optional videos")
    parser.add_argument("--scenario-path", required=True, help="Scenario directory or scenario .pkl path")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=f"Directory to save replay videos (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--view",
        default="both",
        choices=["first_person", "third_person", "top_down", "both", "all"],
        help="Which view(s) to export",
    )
    parser.add_argument("--fps", type=int, default=10, help="Output video FPS (default: 10)")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to replay")
    parser.add_argument(
        "--cameras",
        type=str,
        default="CAM_F0",
        help="Comma-separated NavSim cameras for first-person export (default: CAM_F0)",
    )
    parser.add_argument("--fp-width", type=int, default=1920, help="First-person video width")
    parser.add_argument("--fp-height", type=int, default=1120, help="First-person video height")
    parser.add_argument("--tp-width", type=int, default=1280, help="Third-person video width")
    parser.add_argument("--tp-height", type=int, default=720, help="Third-person video height")
    parser.add_argument(
        "--live",
        action="store_true",
        default=True,
        help="Enable live preview windows (default: on)",
    )
    parser.add_argument("--no-live", dest="live", action="store_false", help="Disable live preview windows")
    parser.add_argument(
        "--save-video",
        action="store_true",
        default=True,
        help="Save replay videos under the output root (default: on)",
    )
    parser.add_argument("--no-save-video", dest="save_video", action="store_false", help="Disable saving videos")
    parser.add_argument("--preview-max-width", type=int, default=1280, help="Max width for first-person preview windows")
    parser.add_argument("--preview-max-height", type=int, default=720, help="Max height for first-person preview windows")

    args = parser.parse_args()
    cameras = [cam.strip() for cam in args.cameras.split(",") if cam.strip()]

    replay_views(
        scenario_path=args.scenario_path,
        output_root=args.output_root,
        view=args.view,
        fps=args.fps,
        max_frames=args.max_frames,
        cameras=cameras,
        first_person_size=(args.fp_width, args.fp_height),
        third_person_size=(args.tp_width, args.tp_height),
        live=args.live,
        save_video=args.save_video,
        preview_max_width=args.preview_max_width,
        preview_max_height=args.preview_max_height,
    )


if __name__ == "__main__":
    main()
