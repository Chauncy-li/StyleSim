from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


_TL_GREEN = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
_TL_YELLOW = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
_TL_RED = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
_TL_UNKNOWN = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

_STATIC_CZONE = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
_STATIC_BARRIER = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
_STATIC_CONE = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
_STATIC_GENERIC = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

_PACIFICA_WHEEL_BASE = 3.089
_PACIFICA_FRONT_LENGTH = 4.049
_PACIFICA_REAR_LENGTH = 1.127
_PACIFICA_COG_TO_REAR = 1.67
_PACIFICA_REAR_AXLE_TO_CENTER = ((_PACIFICA_FRONT_LENGTH + _PACIFICA_REAR_LENGTH) * 0.5) - _PACIFICA_REAR_LENGTH
_STATIC_SIZE_DEFAULTS = {
    "czone_sign": (0.6, 0.6),
    "barrier": (0.5, 1.5),
    "traffic_cone": (0.4, 0.4),
    "generic_object": (1.0, 1.0),
}


@dataclass
class LaneRecord:
    lane_id: str
    vector: np.ndarray
    mask: np.ndarray
    speed_limit: float
    has_speed_limit: bool
    global_polyline: np.ndarray
    anchor_distance: float
    route_score: float
    entry_ids: Tuple[str, ...]
    exit_ids: Tuple[str, ...]
    left_neighbor_ids: Tuple[str, ...]
    right_neighbor_ids: Tuple[str, ...]


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _world_to_local_points(points_xy: np.ndarray, anchor_xy: np.ndarray, anchor_heading: float) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float64)
    dx = points_xy[..., 0] - anchor_xy[0]
    dy = points_xy[..., 1] - anchor_xy[1]
    cos_h = math.cos(anchor_heading)
    sin_h = math.sin(anchor_heading)
    local_x = cos_h * dx + sin_h * dy
    local_y = cos_h * dy - sin_h * dx
    return np.stack([local_x, local_y], axis=-1)


def _world_to_local_vectors(vectors_xy: np.ndarray, anchor_heading: float) -> np.ndarray:
    vectors_xy = np.asarray(vectors_xy, dtype=np.float64)
    cos_h = math.cos(anchor_heading)
    sin_h = math.sin(anchor_heading)
    local_x = cos_h * vectors_xy[..., 0] + sin_h * vectors_xy[..., 1]
    local_y = cos_h * vectors_xy[..., 1] - sin_h * vectors_xy[..., 0]
    return np.stack([local_x, local_y], axis=-1)


def _translate_pose_longitudinally(pose_xyh: np.ndarray, offset_m: float) -> np.ndarray:
    pose_xyh = np.asarray(pose_xyh, dtype=np.float64)
    shifted = pose_xyh.copy()
    shifted[..., 0] = pose_xyh[..., 0] + math.cos(float(pose_xyh[..., 2])) * float(offset_m)
    shifted[..., 1] = pose_xyh[..., 1] + math.sin(float(pose_xyh[..., 2])) * float(offset_m)
    return shifted


def _translate_points_along_heading(points_xy: np.ndarray, headings: np.ndarray, offset_m: float) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float64)
    headings = np.asarray(headings, dtype=np.float64).reshape(-1)
    shifted = np.asarray(points_xy, dtype=np.float64).copy()
    shifted[:, 0] = points_xy[:, 0] + np.cos(headings) * float(offset_m)
    shifted[:, 1] = points_xy[:, 1] + np.sin(headings) * float(offset_m)
    return shifted


def _resample_polyline(points_xy: np.ndarray, num_points: int) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float64)
    if len(points_xy) == 0:
        return np.zeros((num_points, 2), dtype=np.float64)
    if len(points_xy) == 1:
        return np.repeat(points_xy, num_points, axis=0)

    deltas = np.diff(points_xy, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cumulative[-1]
    if total < 1e-6:
        return np.repeat(points_xy[:1], num_points, axis=0)

    samples = np.linspace(0.0, total, num_points)
    xs = np.interp(samples, cumulative, points_xy[:, 0])
    ys = np.interp(samples, cumulative, points_xy[:, 1])
    return np.stack([xs, ys], axis=1)


def _resample_scalar(values: np.ndarray, num_points: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        if len(values) == 0:
            return np.zeros((num_points,), dtype=np.float64)
        if len(values) == 1:
            return np.repeat(values, num_points)
        samples = np.linspace(0.0, len(values) - 1, num_points)
        return np.interp(samples, np.arange(len(values)), values)

    if values.ndim == 2:
        cols = [_resample_scalar(values[:, i], num_points) for i in range(values.shape[1])]
        return np.stack(cols, axis=1)

    raise ValueError(f"Unsupported scalar resample shape: {values.shape}")


def _compute_polyline_vectors(points_xy: np.ndarray) -> np.ndarray:
    vectors = np.diff(points_xy, axis=0)
    if len(vectors) == 0:
        return np.zeros_like(points_xy)
    return np.vstack([vectors, vectors[-1:]])


def _compute_normals(points_xy: np.ndarray) -> np.ndarray:
    tangents = _compute_polyline_vectors(points_xy)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.clip(norms, 1e-6, None)
    return np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)


def _encode_track_type(track_type: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    upper = str(track_type).upper()
    if "VEHICLE" in upper:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32), "vehicle"
    if "PEDESTRIAN" in upper or "HUMAN" in upper:
        return np.array([0.0, 1.0, 0.0], dtype=np.float32), "ped_bike"
    if "BICYCLE" in upper or "CYCLIST" in upper:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32), "ped_bike"
    return None, None


def _encode_static_track_type(track_type: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    upper = str(track_type).upper()
    if "CZONE" in upper or ("SIGN" in upper and "STOP_SIGN" not in upper):
        return _STATIC_CZONE.copy(), "czone_sign"
    if "BARRIER" in upper:
        return _STATIC_BARRIER.copy(), "barrier"
    if "CONE" in upper:
        return _STATIC_CONE.copy(), "traffic_cone"
    if "GENERIC" in upper or "OBJECT" in upper:
        return _STATIC_GENERIC.copy(), "generic_object"
    return None, None


def _is_lane_feature(feature_type: str) -> bool:
    upper = str(feature_type).upper()
    return upper.startswith("LANE")


def _encode_traffic_light(state: Any) -> np.ndarray:
    text = str(state).upper()
    if "GO" in text or "GREEN" in text:
        return _TL_GREEN.copy()
    if "CAUTION" in text or "YELLOW" in text:
        return _TL_YELLOW.copy()
    if "STOP" in text or "RED" in text:
        return _TL_RED.copy()
    return _TL_UNKNOWN.copy()


def _get_timestep_array(scenario_data: Dict[str, Any], length: int) -> np.ndarray:
    timestep = scenario_data.get("metadata", {}).get("timestep")
    if timestep is not None:
        arr = np.asarray(timestep, dtype=np.float64)
        if arr.shape[0] == length:
            return arr
    return np.arange(length, dtype=np.float64) * 0.1


def _safe_valid(track: Dict[str, Any], frame_idx: int) -> bool:
    valid = np.asarray(track["state"].get("valid", []))
    if frame_idx >= len(valid):
        return False
    return bool(valid[frame_idx] > 0.5)


def _scalar_from_state(track: Dict[str, Any], key: str, frame_idx: int, default: float) -> float:
    values = track["state"].get(key)
    if values is None:
        return default
    arr = np.asarray(values)
    if frame_idx >= len(arr):
        return default
    value = arr[frame_idx]
    if np.ndim(value) == 0:
        return float(value)
    return float(np.asarray(value).reshape(-1)[0])


def _build_future_heading(positions_xy: np.ndarray, headings: np.ndarray) -> np.ndarray:
    if len(positions_xy) <= 1:
        return headings.copy()

    diffs = np.diff(positions_xy, axis=0)
    new_heading = headings.copy()
    valid = np.linalg.norm(diffs, axis=1) > 1e-4
    new_heading[:-1][valid] = np.arctan2(diffs[valid, 1], diffs[valid, 0])
    if valid.any():
        new_heading[-1] = new_heading[:-1][valid][-1]
    return new_heading


def _polyline_heading_at(points_xy: np.ndarray, index: int) -> float:
    points_xy = np.asarray(points_xy, dtype=np.float64)
    if len(points_xy) <= 1:
        return 0.0
    lo = max(0, int(index) - 1)
    hi = min(len(points_xy) - 1, int(index) + 1)
    if hi == lo:
        if hi < len(points_xy) - 1:
            hi += 1
        elif lo > 0:
            lo -= 1
    delta = points_xy[hi] - points_xy[lo]
    return float(math.atan2(delta[1], delta[0]))


def _dedupe_consecutive(values: Sequence[int]) -> List[int]:
    ordered: List[int] = []
    for value in values:
        if not ordered or ordered[-1] != value:
            ordered.append(int(value))
    return ordered


def _feature_id_list(feature: Dict[str, Any], *keys: str) -> Tuple[str, ...]:
    for key in keys:
        raw = feature.get(key)
        if raw is None:
            continue
        if isinstance(raw, (str, bytes)):
            raw = [raw]
        return tuple(str(item) for item in raw if item is not None and str(item) != "")
    return tuple()


def _positive_median(values: np.ndarray) -> Optional[float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr) & (arr > 1e-3)]
    if arr.size == 0:
        return None
    return float(np.median(arr))


def _ensure_xy_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[1] == 1:
        arr = np.concatenate([arr, np.zeros((arr.shape[0], 1), dtype=np.float64)], axis=1)
    elif arr.shape[1] > 2:
        arr = arr[:, :2]
    return arr


def _estimate_world_velocity(position_xy: np.ndarray, time_axis: np.ndarray) -> np.ndarray:
    position_xy = np.asarray(position_xy, dtype=np.float64)
    if len(position_xy) <= 1:
        return np.zeros((len(position_xy), 2), dtype=np.float64)

    edge_order = 2 if len(position_xy) >= 3 else 1
    velocity = np.zeros_like(position_xy)
    for dim in range(2):
        velocity[:, dim] = np.gradient(position_xy[:, dim], time_axis, edge_order=edge_order)
    return velocity


def _estimate_world_acceleration(velocity_xy: np.ndarray, time_axis: np.ndarray) -> np.ndarray:
    velocity_xy = _ensure_xy_array(velocity_xy)
    if len(velocity_xy) <= 1:
        return np.zeros_like(velocity_xy)

    edge_order = 2 if len(velocity_xy) >= 3 else 1
    acceleration = np.zeros_like(velocity_xy)
    for dim in range(2):
        acceleration[:, dim] = np.gradient(velocity_xy[:, dim], time_axis, edge_order=edge_order)
    return acceleration


class BridgeSimNuPlanFeatureBuilder:
    """
    Reconstruct the vectorized nuPlan-style planner inputs from BridgeSim scenario
    pickles and the live MetaDrive environment.
    """

    def __init__(
        self,
        agent_num: int = 32,
        predicted_neighbor_num: int = 10,
        static_objects_num: int = 5,
        lane_num: int = 70,
        lane_len: int = 20,
        route_num: int = 25,
        route_len: int = 20,
        past_steps: int = 21,
        future_steps: int = 80,
        max_ped_bike: int = 10,
        bridge_ego_reference: str = "rear_axle",
    ) -> None:
        self.agent_num = int(agent_num)
        self.predicted_neighbor_num = int(predicted_neighbor_num)
        self.static_objects_num = int(static_objects_num)
        self.lane_num = int(lane_num)
        self.lane_len = int(lane_len)
        self.route_num = int(route_num)
        self.route_len = int(route_len)
        self.past_steps = int(past_steps)
        self.future_steps = int(future_steps)
        self.max_ped_bike = int(max_ped_bike)
        bridge_reference_offsets = {
            "rear_axle": 0.0,
            "center": _PACIFICA_REAR_AXLE_TO_CENTER,
            "cog": _PACIFICA_COG_TO_REAR,
        }
        if bridge_ego_reference not in bridge_reference_offsets:
            raise ValueError(
                f"Unsupported bridge_ego_reference={bridge_ego_reference!r}. "
                f"Choose from {sorted(bridge_reference_offsets)}."
            )
        self.bridge_ego_reference = str(bridge_ego_reference)
        self.bridge_to_rear_axle_offset = float(bridge_reference_offsets[self.bridge_ego_reference])

    def build_inference_features(
        self,
        scenario_data: Dict[str, Any],
        ego_state: Dict[str, Any],
        frame_id: int,
        env: Optional[Any] = None,
        ego_history: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, np.ndarray]:
        bridge_anchor_pose = np.array(
            [
                float(ego_state["position"][0]),
                float(ego_state["position"][1]),
                float(ego_state["heading"]),
            ],
            dtype=np.float64,
        )
        anchor_pose = self._bridge_pose_to_rear_axle_pose(bridge_anchor_pose)
        features, _, _ = self._build_core_features(
            scenario_data=scenario_data,
            anchor_pose=anchor_pose,
            frame_id=frame_id,
            env=env,
            future_route_world=None,
            ego_history=ego_history,
        )
        return features

    def build_training_sample(
        self,
        scenario_data: Dict[str, Any],
        frame_id: int,
    ) -> Dict[str, np.ndarray]:
        sdc_id = scenario_data["metadata"]["sdc_id"]
        ego_track = scenario_data["tracks"][sdc_id]
        bridge_anchor_pose = np.array(
            [
                float(ego_track["state"]["position"][frame_id][0]),
                float(ego_track["state"]["position"][frame_id][1]),
                float(ego_track["state"]["heading"][frame_id]),
            ],
            dtype=np.float64,
        )
        anchor_pose = self._bridge_pose_to_rear_axle_pose(bridge_anchor_pose)

        future_positions_bridge = np.asarray(
            ego_track["state"]["position"][frame_id + 1: frame_id + 1 + self.future_steps, :2],
            dtype=np.float64,
        )
        future_headings = np.asarray(
            ego_track["state"]["heading"][frame_id + 1: frame_id + 1 + self.future_steps],
            dtype=np.float64,
        )
        future_world = np.asarray(
            self._bridge_positions_to_rear_axle(future_positions_bridge, future_headings),
            dtype=np.float64,
        )
        features, selected_track_ids, _ = self._build_core_features(
            scenario_data=scenario_data,
            anchor_pose=anchor_pose,
            frame_id=frame_id,
            env=None,
            future_route_world=future_world,
            ego_history=None,
        )

        ego_future = self._build_ego_future(scenario_data, sdc_id, anchor_pose, frame_id)
        neighbor_future, neighbor_future_mask = self._build_neighbor_future(
            scenario_data=scenario_data,
            anchor_pose=anchor_pose,
            frame_id=frame_id,
            selected_track_ids=selected_track_ids[: self.predicted_neighbor_num],
        )

        features.update(
            {
                "ego_agent_future": ego_future.astype(np.float32),
                "neighbor_agents_future": neighbor_future.astype(np.float32),
                "neighbor_agents_future_mask": neighbor_future_mask.astype(np.bool_),
                "code_lat": np.asarray(-1, dtype=np.int64),
                "code_lon": np.asarray(-1, dtype=np.int64),
            }
        )
        return features

    def valid_export_frames(self, scenario_data: Dict[str, Any]) -> Sequence[int]:
        sdc_id = scenario_data["metadata"]["sdc_id"]
        total_len = len(scenario_data["tracks"][sdc_id]["state"]["position"])
        start = self.past_steps - 1
        end = total_len - self.future_steps - 1
        if end < start:
            return []
        return list(range(start, end + 1))

    def _build_core_features(
        self,
        scenario_data: Dict[str, Any],
        anchor_pose: np.ndarray,
        frame_id: int,
        env: Optional[Any],
        future_route_world: Optional[np.ndarray],
        ego_history: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, np.ndarray], List[str], List[LaneRecord]]:
        sdc_id = scenario_data["metadata"]["sdc_id"]
        ego_agent_past, ego_current_state = self._build_ego_history(
            scenario_data,
            sdc_id,
            anchor_pose,
            frame_id,
            ego_history=ego_history,
        )
        (
            neighbor_agents_past,
            neighbor_agents_past_mask,
            selected_track_ids,
        ) = self._build_neighbor_history(scenario_data, sdc_id, anchor_pose, frame_id)
        lanes, lanes_mask, lane_speed_limit, lane_has_speed_limit, lane_records = self._build_lane_features(
            scenario_data,
            anchor_pose,
            frame_id,
            future_route_world,
        )
        route_lanes, route_lanes_mask, route_speed_limit, route_has_speed_limit = self._build_route_features(
            scenario_data=scenario_data,
            anchor_pose=anchor_pose,
            frame_id=frame_id,
            env=env,
            lane_records=lane_records,
            future_route_world=future_route_world,
        )

        static_objects = self._build_static_objects(
            scenario_data=scenario_data,
            anchor_pose=anchor_pose,
            frame_id=frame_id,
        )

        features = {
            "ego_current_state": ego_current_state.astype(np.float32),
            "ego_agent_past": ego_agent_past.astype(np.float32),
            "neighbor_agents_past": neighbor_agents_past.astype(np.float32),
            "neighbor_agents_past_mask": neighbor_agents_past_mask.astype(np.bool_),
            "static_objects": static_objects,
            "lanes": lanes.astype(np.float32),
            "lanes_mask": lanes_mask.astype(np.bool_),
            "lanes_speed_limit": lane_speed_limit.astype(np.float32),
            "lanes_has_speed_limit": lane_has_speed_limit.astype(np.bool_),
            "route_lanes": route_lanes.astype(np.float32),
            "route_lanes_mask": route_lanes_mask.astype(np.bool_),
            "route_lanes_speed_limit": route_speed_limit.astype(np.float32),
            "route_lanes_has_speed_limit": route_has_speed_limit.astype(np.bool_),
        }
        return features, selected_track_ids, lane_records

    def _bridge_pose_to_rear_axle_pose(self, pose_xyh: np.ndarray) -> np.ndarray:
        if abs(self.bridge_to_rear_axle_offset) < 1e-8:
            return np.asarray(pose_xyh, dtype=np.float64)
        return _translate_pose_longitudinally(np.asarray(pose_xyh, dtype=np.float64), -self.bridge_to_rear_axle_offset)

    def _bridge_positions_to_rear_axle(self, positions_xy: np.ndarray, headings: np.ndarray) -> np.ndarray:
        if abs(self.bridge_to_rear_axle_offset) < 1e-8:
            return np.asarray(positions_xy, dtype=np.float64)
        return _translate_points_along_heading(
            np.asarray(positions_xy, dtype=np.float64),
            np.asarray(headings, dtype=np.float64),
            -self.bridge_to_rear_axle_offset,
        )

    def rear_axle_predictions_to_bridge_frame(
        self,
        rear_axle_xy: np.ndarray,
        heading: np.ndarray,
    ) -> np.ndarray:
        rear_axle_xy = np.asarray(rear_axle_xy, dtype=np.float64)
        heading = np.asarray(heading, dtype=np.float64).reshape(-1)
        if abs(self.bridge_to_rear_axle_offset) < 1e-8:
            return rear_axle_xy.astype(np.float32)

        bridge_delta = np.stack(
            [
                np.cos(heading) * self.bridge_to_rear_axle_offset - self.bridge_to_rear_axle_offset,
                np.sin(heading) * self.bridge_to_rear_axle_offset,
            ],
            axis=1,
        )
        return (rear_axle_xy + bridge_delta).astype(np.float32)

    def _build_ego_history(
        self,
        scenario_data: Dict[str, Any],
        sdc_id: str,
        anchor_pose: np.ndarray,
        frame_id: int,
        ego_history: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if ego_history:
            return self._build_live_ego_history(anchor_pose=anchor_pose, ego_history=ego_history)

        track = scenario_data["tracks"][sdc_id]
        position_bridge = np.asarray(track["state"]["position"], dtype=np.float64)
        heading = np.asarray(track["state"]["heading"], dtype=np.float64)
        time_axis = _get_timestep_array(scenario_data, len(position_bridge))
        position = self._bridge_positions_to_rear_axle(position_bridge[:, :2], heading)
        velocity = _estimate_world_velocity(position, time_axis)
        accel_world = _estimate_world_acceleration(velocity, time_axis)
        indices = [max(0, frame_id - self.past_steps + 1 + i) for i in range(self.past_steps)]

        pose_local = np.zeros((self.past_steps, 3), dtype=np.float64)
        pose_local[:, :2] = _world_to_local_points(position[indices, :2], anchor_pose[:2], anchor_pose[2])
        pose_local[:, 2] = _wrap_to_pi(heading[indices] - anchor_pose[2])

        vel_local = _world_to_local_vectors(velocity[indices], anchor_pose[2])
        acc_local = _world_to_local_vectors(accel_world[indices], anchor_pose[2])

        ego_agent_past = np.concatenate([pose_local, vel_local, acc_local], axis=1)
        ego_current_state = self._calculate_ego_current_state(ego_agent_past, time_axis[indices])
        return ego_agent_past, ego_current_state

    def _calculate_ego_current_state(self, ego_agent_past: np.ndarray, time_axis: np.ndarray) -> np.ndarray:
        current_state = ego_agent_past[-1]
        prev_index = len(time_axis) - 2
        while prev_index >= 0 and abs(float(time_axis[-1] - time_axis[prev_index])) < 1e-6:
            prev_index -= 1
        if prev_index < 0:
            prev_index = max(0, len(time_axis) - 2)

        prev_state = ego_agent_past[prev_index]
        dt = float(time_axis[-1] - time_axis[prev_index]) if len(time_axis) >= 2 else 0.1
        dt = max(dt, 1e-3)

        cur_velocity = float(current_state[3])
        angle_diff = float(_wrap_to_pi(current_state[2] - prev_state[2]))
        yaw_rate = angle_diff / dt

        if abs(cur_velocity) < 0.2:
            steering_angle = 0.0
            yaw_rate = 0.0
        else:
            steering_angle = math.atan(yaw_rate * _PACIFICA_WHEEL_BASE / max(abs(cur_velocity), 1e-3))
            steering_angle = float(np.clip(steering_angle, -2.0 * np.pi / 3.0, 2.0 * np.pi / 3.0))
            yaw_rate = float(np.clip(yaw_rate, -0.95, 0.95))

        output = np.zeros((10,), dtype=np.float32)
        output[:2] = current_state[:2]
        output[2] = math.cos(float(current_state[2]))
        output[3] = math.sin(float(current_state[2]))
        output[4:8] = current_state[3:7]
        output[8] = steering_angle
        output[9] = yaw_rate
        return output

    def _build_live_ego_history(
        self,
        anchor_pose: np.ndarray,
        ego_history: Sequence[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        samples = list(ego_history)
        if not samples:
            raise ValueError("ego_history must contain at least one state for live inference.")

        target_frames = self.past_steps
        if len(samples) >= target_frames:
            sampled = samples[-target_frames:]
        else:
            sampled = [samples[0]] * (target_frames - len(samples)) + samples

        bridge_positions = np.asarray(
            [[float(state["position"][0]), float(state["position"][1])] for state in sampled],
            dtype=np.float64,
        )
        headings = np.asarray([float(state["heading"]) for state in sampled], dtype=np.float64)
        rear_positions = self._bridge_positions_to_rear_axle(bridge_positions, headings)

        time_axis = np.asarray(
            [float(state.get("sim_time", idx * 0.1)) for idx, state in enumerate(sampled)],
            dtype=np.float64,
        )
        if np.any(np.diff(time_axis) <= 1e-9):
            time_axis = np.arange(len(sampled), dtype=np.float64) * 0.1

        velocity = _estimate_world_velocity(rear_positions, time_axis)
        accel_world = _estimate_world_acceleration(velocity, time_axis)

        pose_local = np.zeros((self.past_steps, 3), dtype=np.float64)
        pose_local[:, :2] = _world_to_local_points(rear_positions, anchor_pose[:2], anchor_pose[2])
        pose_local[:, 2] = _wrap_to_pi(headings - anchor_pose[2])

        vel_local = _world_to_local_vectors(velocity, anchor_pose[2])
        acc_local = _world_to_local_vectors(accel_world, anchor_pose[2])

        ego_agent_past = np.concatenate([pose_local, vel_local, acc_local], axis=1)
        ego_current_state = self._calculate_ego_current_state(ego_agent_past, time_axis)
        return ego_agent_past, ego_current_state

    def _build_static_type_size_priors(self, scenario_data: Dict[str, Any]) -> Dict[str, Tuple[float, float]]:
        priors: Dict[str, Tuple[float, float]] = {}
        grouped: Dict[str, List[Tuple[float, float]]] = {}

        for track in scenario_data["tracks"].values():
            _, type_label = _encode_static_track_type(track.get("type", "UNKNOWN"))
            if type_label is None:
                continue

            width_values = np.asarray(track["state"].get("width", []), dtype=np.float64).reshape(-1)
            length_values = np.asarray(track["state"].get("length", []), dtype=np.float64).reshape(-1)
            valid_values = np.asarray(track["state"].get("valid", []), dtype=np.float64).reshape(-1)
            if valid_values.size > 0:
                valid_mask = valid_values > 0.5
                if valid_mask.shape[0] == width_values.shape[0]:
                    width_values = width_values[valid_mask]
                if valid_mask.shape[0] == length_values.shape[0]:
                    length_values = length_values[valid_mask]

            width_median = _positive_median(width_values)
            length_median = _positive_median(length_values)
            if width_median is None and length_median is None:
                continue

            default_width, default_length = _STATIC_SIZE_DEFAULTS[type_label]
            grouped.setdefault(type_label, []).append(
                (
                    width_median if width_median is not None else default_width,
                    length_median if length_median is not None else default_length,
                )
            )

        for type_label, values in grouped.items():
            widths = np.asarray([item[0] for item in values], dtype=np.float64)
            lengths = np.asarray([item[1] for item in values], dtype=np.float64)
            priors[type_label] = (
                float(np.median(widths)) if widths.size > 0 else _STATIC_SIZE_DEFAULTS[type_label][0],
                float(np.median(lengths)) if lengths.size > 0 else _STATIC_SIZE_DEFAULTS[type_label][1],
            )

        return priors

    def _resolve_static_extent(
        self,
        track: Dict[str, Any],
        frame_id: int,
        type_label: str,
        size_priors: Dict[str, Tuple[float, float]],
    ) -> Tuple[float, float]:
        width = _scalar_from_state(track, "width", frame_id, 0.0)
        length = _scalar_from_state(track, "length", frame_id, 0.0)
        default_width, default_length = size_priors.get(type_label, _STATIC_SIZE_DEFAULTS[type_label])

        if width <= 1e-3:
            width_values = np.asarray(track["state"].get("width", []), dtype=np.float64).reshape(-1)
            valid_values = np.asarray(track["state"].get("valid", []), dtype=np.float64).reshape(-1)
            if valid_values.size > 0 and valid_values.shape[0] == width_values.shape[0]:
                width_values = width_values[valid_values > 0.5]
            width = _positive_median(width_values) or default_width

        if length <= 1e-3:
            length_values = np.asarray(track["state"].get("length", []), dtype=np.float64).reshape(-1)
            valid_values = np.asarray(track["state"].get("valid", []), dtype=np.float64).reshape(-1)
            if valid_values.size > 0 and valid_values.shape[0] == length_values.shape[0]:
                length_values = length_values[valid_values > 0.5]
            length = _positive_median(length_values) or default_length

        return float(width), float(length)

    def _build_static_objects(
        self,
        scenario_data: Dict[str, Any],
        anchor_pose: np.ndarray,
        frame_id: int,
    ) -> np.ndarray:
        size_priors = self._build_static_type_size_priors(scenario_data)
        candidates: List[Tuple[float, np.ndarray]] = []

        for track in scenario_data["tracks"].values():
            one_hot, type_label = _encode_static_track_type(track.get("type", "UNKNOWN"))
            if one_hot is None or not _safe_valid(track, frame_id):
                continue

            position = np.asarray(track["state"].get("position"), dtype=np.float64)
            heading = np.asarray(track["state"].get("heading"), dtype=np.float64).reshape(-1)
            if position.ndim != 2 or position.shape[0] <= frame_id or heading.shape[0] <= frame_id:
                continue

            local_xy = _world_to_local_points(position[frame_id: frame_id + 1, :2], anchor_pose[:2], anchor_pose[2])[0]
            local_heading = float(_wrap_to_pi(heading[frame_id] - anchor_pose[2]))
            width, length = self._resolve_static_extent(
                track=track,
                frame_id=frame_id,
                type_label=type_label,
                size_priors=size_priors,
            )

            vector = np.zeros((10,), dtype=np.float32)
            vector[0] = float(local_xy[0])
            vector[1] = float(local_xy[1])
            vector[2] = math.cos(local_heading)
            vector[3] = math.sin(local_heading)
            vector[4] = width
            vector[5] = length
            vector[6:] = one_hot
            candidates.append((float(np.linalg.norm(local_xy)), vector))

        static_objects = np.zeros((self.static_objects_num, 10), dtype=np.float32)
        candidates.sort(key=lambda item: item[0])
        for idx, (_, vector) in enumerate(candidates[: self.static_objects_num]):
            static_objects[idx] = vector
        return static_objects

    def _build_neighbor_history(
        self,
        scenario_data: Dict[str, Any],
        sdc_id: str,
        anchor_pose: np.ndarray,
        frame_id: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        candidates = []

        for track_id, track in scenario_data["tracks"].items():
            if track_id == sdc_id or not _safe_valid(track, frame_id):
                continue

            one_hot, class_group = _encode_track_type(track.get("type", "UNKNOWN"))
            if one_hot is None:
                continue

            position = np.asarray(track["state"]["position"], dtype=np.float64)
            heading = np.asarray(track["state"]["heading"], dtype=np.float64)
            velocity = np.asarray(track["state"].get("velocity"), dtype=np.float64)
            if velocity.ndim == 1:
                velocity = velocity[:, None]
            if velocity.shape[1] > 2:
                velocity = velocity[:, :2]

            history = np.zeros((self.past_steps, 11), dtype=np.float32)
            mask = np.zeros((self.past_steps,), dtype=np.bool_)
            fill_idx = frame_id
            for step in range(self.past_steps - 1, -1, -1):
                src_idx = frame_id - (self.past_steps - 1 - step)
                if src_idx < 0:
                    src_idx = 0
                if _safe_valid(track, src_idx):
                    fill_idx = src_idx
                if not _safe_valid(track, fill_idx):
                    continue

                local_pos = _world_to_local_points(position[fill_idx: fill_idx + 1, :2], anchor_pose[:2], anchor_pose[2])[0]
                local_heading = float(_wrap_to_pi(heading[fill_idx] - anchor_pose[2]))
                local_vel = _world_to_local_vectors(velocity[fill_idx: fill_idx + 1], anchor_pose[2])[0]
                width = _scalar_from_state(track, "width", fill_idx, 0.0)
                length = _scalar_from_state(track, "length", fill_idx, 0.0)

                history[step, 0] = local_pos[0]
                history[step, 1] = local_pos[1]
                history[step, 2] = math.cos(local_heading)
                history[step, 3] = math.sin(local_heading)
                history[step, 4] = local_vel[0]
                history[step, 5] = local_vel[1]
                history[step, 6] = width
                history[step, 7] = length
                history[step, 8:] = one_hot
                mask[step] = width > 1e-6 or length > 1e-6

            if not mask[-1]:
                continue

            distance = float(np.linalg.norm(history[-1, :2]))
            candidates.append(
                {
                    "track_id": track_id,
                    "history": history,
                    "mask": mask,
                    "class_group": class_group,
                    "distance": distance,
                }
            )

        ped_bike = sorted([c for c in candidates if c["class_group"] == "ped_bike"], key=lambda x: x["distance"])
        vehicles = sorted([c for c in candidates if c["class_group"] == "vehicle"], key=lambda x: x["distance"])

        if len(candidates) <= self.agent_num:
            selected = sorted(candidates, key=lambda x: x["distance"])[: self.agent_num]
        else:
            selected = ped_bike[: self.max_ped_bike] + vehicles
            remaining_slots = self.agent_num - len(selected)
            if remaining_slots > 0:
                selected.extend(ped_bike[self.max_ped_bike: self.max_ped_bike + remaining_slots])
            selected = sorted(selected, key=lambda x: x["distance"])[: self.agent_num]

        agents = np.zeros((self.agent_num, self.past_steps, 11), dtype=np.float32)
        masks = np.zeros((self.agent_num, self.past_steps), dtype=np.bool_)
        selected_ids: List[str] = []

        for idx, record in enumerate(selected):
            agents[idx] = record["history"]
            masks[idx] = record["mask"]
            selected_ids.append(record["track_id"])

        return agents, masks, selected_ids

    def _build_lane_features(
        self,
        scenario_data: Dict[str, Any],
        anchor_pose: np.ndarray,
        frame_id: int,
        future_route_world: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[LaneRecord]]:
        lane_records: List[LaneRecord] = []
        dynamic_map = scenario_data.get("dynamic_map_states", {})
        future_route_world = (
            np.asarray(future_route_world, dtype=np.float64)
            if future_route_world is not None and len(future_route_world) > 0
            else None
        )

        for lane_id, feature in scenario_data.get("map_features", {}).items():
            if not _is_lane_feature(feature.get("type", "")):
                continue

            polyline = np.asarray(feature.get("polyline"), dtype=np.float64)
            if polyline.ndim != 2 or polyline.shape[0] < 2:
                continue
            polyline = polyline[:, :2]
            center = _resample_polyline(polyline, self.lane_len)
            tangents = _compute_polyline_vectors(center)
            normals = _compute_normals(center)

            width = feature.get("width")
            if width is None:
                left_width = np.full((self.lane_len,), 1.75, dtype=np.float64)
                right_width = np.full((self.lane_len,), 1.75, dtype=np.float64)
            else:
                width_arr = np.asarray(width, dtype=np.float64)
                width_arr = _resample_scalar(width_arr, self.lane_len)
                if width_arr.ndim == 1:
                    left_width = width_arr * 0.5
                    right_width = width_arr * 0.5
                else:
                    left_width = width_arr[:, 0]
                    right_width = width_arr[:, 1] if width_arr.shape[1] > 1 else width_arr[:, 0]

            left_boundary = center + normals * left_width[:, None]
            right_boundary = center - normals * right_width[:, None]

            local_center = _world_to_local_points(center, anchor_pose[:2], anchor_pose[2])
            local_left = _world_to_local_points(left_boundary, anchor_pose[:2], anchor_pose[2])
            local_right = _world_to_local_points(right_boundary, anchor_pose[:2], anchor_pose[2])
            local_tangents = _compute_polyline_vectors(local_center)

            tl = _TL_UNKNOWN
            tl_state = dynamic_map.get(str(lane_id))
            if tl_state is not None:
                states = np.asarray(tl_state.get("state", {}).get("object_state", []), dtype=object)
                if frame_id < len(states):
                    tl = _encode_traffic_light(states[frame_id])
            tl_points = np.repeat(tl[None, :], self.lane_len, axis=0)

            vector = np.zeros((self.lane_len, 12), dtype=np.float32)
            vector[:, 0:2] = local_center.astype(np.float32)
            vector[:, 2:4] = local_tangents.astype(np.float32)
            vector[:, 4:6] = (local_left - local_center).astype(np.float32)
            vector[:, 6:8] = (local_right - local_center).astype(np.float32)
            vector[:, 8:12] = tl_points.astype(np.float32)

            speed_limit = 0.0
            has_speed_limit = False
            for key in ("speed_limit_mps", "speed_limit_kmh", "speed_limit_mph"):
                if key in feature:
                    raw_value = feature[key]
                    speed_limit = float(np.asarray(raw_value).reshape(-1)[0])
                    if key.endswith("kmh"):
                        speed_limit /= 3.6
                    elif key.endswith("mph"):
                        speed_limit *= 0.44704
                    has_speed_limit = True
                    break

            local_distance = float(np.linalg.norm(local_center[:, :2], axis=1).min())
            route_score = local_distance
            if future_route_world is not None and len(future_route_world) > 0:
                diff = center[:, None, :] - future_route_world[None, :, :]
                route_score = float(np.linalg.norm(diff, axis=2).min())

            entry_ids = _feature_id_list(feature, "entry_lanes", "entry")
            exit_ids = _feature_id_list(feature, "exit_lanes", "exit")
            left_neighbor_ids = _feature_id_list(feature, "left_neighbor", "left_neighbors")
            right_neighbor_ids = _feature_id_list(feature, "right_neighbor", "right_neighbors")

            lane_records.append(
                LaneRecord(
                    lane_id=str(lane_id),
                    vector=vector,
                    mask=np.ones((self.lane_len,), dtype=np.bool_),
                    speed_limit=float(speed_limit),
                    has_speed_limit=bool(has_speed_limit),
                    global_polyline=center,
                    anchor_distance=local_distance,
                    route_score=route_score,
                    entry_ids=entry_ids,
                    exit_ids=exit_ids,
                    left_neighbor_ids=left_neighbor_ids,
                    right_neighbor_ids=right_neighbor_ids,
                )
            )

        lane_records.sort(key=lambda record: float(np.linalg.norm(record.vector[:, :2], axis=1).min()))
        lanes = np.zeros((self.lane_num, self.lane_len, 12), dtype=np.float32)
        masks = np.zeros((self.lane_num, self.lane_len), dtype=np.bool_)
        speed_limits = np.zeros((self.lane_num, 1), dtype=np.float32)
        has_speed_limits = np.zeros((self.lane_num, 1), dtype=np.bool_)

        for idx, record in enumerate(lane_records[: self.lane_num]):
            lanes[idx] = record.vector
            masks[idx] = record.mask
            speed_limits[idx, 0] = record.speed_limit
            has_speed_limits[idx, 0] = record.has_speed_limit

        return lanes, masks, speed_limits, has_speed_limits, lane_records

    def _build_route_features(
        self,
        scenario_data: Dict[str, Any],
        anchor_pose: np.ndarray,
        frame_id: int,
        env: Optional[Any],
        lane_records: Sequence[LaneRecord],
        future_route_world: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        route_vectors: List[np.ndarray] = []
        route_speed_limits: List[float] = []
        route_has_speed_limits: List[bool] = []

        lane_by_id = {record.lane_id: record for record in lane_records}
        lane_to_group, group_to_lanes, outgoing_groups = self._build_lane_groups(lane_records)

        live_route_polylines: List[np.ndarray] = []
        if env is not None:
            live_route_polylines = self._build_live_route_global_polylines(env)

        route_group_sequence: List[int] = []
        route_source = "fallback"
        if live_route_polylines:
            route_group_sequence = self._infer_route_groups_from_live_navigation(
                live_route_polylines=live_route_polylines,
                lane_records=lane_records,
                lane_to_group=lane_to_group,
            )
            if route_group_sequence:
                route_source = "live"

        if not route_group_sequence and future_route_world is not None and len(future_route_world) > 0:
            route_group_sequence = self._infer_route_groups_from_future_path(
                future_route_world=future_route_world,
                lane_records=lane_records,
                lane_to_group=lane_to_group,
            )
            if route_group_sequence:
                route_source = "future"

        route_group_sequence = self._connect_route_group_sequence(route_group_sequence, outgoing_groups)
        if route_source != "live":
            route_group_sequence = self._extend_route_group_sequence(
                route_group_sequence=route_group_sequence,
                lane_records=lane_records,
                lane_to_group=lane_to_group,
                group_to_lanes=group_to_lanes,
                outgoing_groups=outgoing_groups,
            )

        route_records = self._collect_route_lane_records(
            lane_records=lane_records,
            lane_to_group=lane_to_group,
            route_group_sequence=route_group_sequence,
        )

        if route_records:
            for record in route_records[: self.route_num]:
                route_vectors.append(record.vector.copy())
                route_speed_limits.append(record.speed_limit)
                route_has_speed_limits.append(record.has_speed_limit)
        elif env is not None:
            route_vectors = self._build_live_route_vectors(env, anchor_pose)
            route_speed_limits = [0.0] * len(route_vectors)
            route_has_speed_limits = [False] * len(route_vectors)

        if not route_vectors:
            sorted_records = sorted(lane_records, key=lambda record: record.route_score)
            for record in sorted_records[: self.route_num]:
                route_vectors.append(record.vector.copy())
                route_speed_limits.append(record.speed_limit)
                route_has_speed_limits.append(record.has_speed_limit)

        route_lanes = np.zeros((self.route_num, self.route_len, 12), dtype=np.float32)
        route_masks = np.zeros((self.route_num, self.route_len), dtype=np.bool_)
        route_speed = np.zeros((self.route_num, 1), dtype=np.float32)
        route_has_speed = np.zeros((self.route_num, 1), dtype=np.bool_)

        for idx, vector in enumerate(route_vectors[: self.route_num]):
            route_lanes[idx] = vector
            route_masks[idx] = np.any(np.abs(vector[:, :8]) > 0.0, axis=1)
            route_speed[idx, 0] = route_speed_limits[idx]
            route_has_speed[idx, 0] = route_has_speed_limits[idx]

        return route_lanes, route_masks, route_speed, route_has_speed

    def _build_lane_groups(
        self,
        lane_records: Sequence[LaneRecord],
    ) -> Tuple[Dict[str, int], Dict[int, Tuple[str, ...]], Dict[int, Tuple[int, ...]]]:
        lane_by_id = {record.lane_id: record for record in lane_records}
        lateral_adjacency: Dict[str, set] = {lane_id: set() for lane_id in lane_by_id}

        for record in lane_records:
            for neighbor_id in record.left_neighbor_ids + record.right_neighbor_ids:
                if neighbor_id not in lane_by_id:
                    continue
                lateral_adjacency[record.lane_id].add(neighbor_id)
                lateral_adjacency[neighbor_id].add(record.lane_id)

        lane_to_group: Dict[str, int] = {}
        group_to_lanes: Dict[int, Tuple[str, ...]] = {}

        next_group_id = 0
        for lane_id in lane_by_id:
            if lane_id in lane_to_group:
                continue
            stack = [lane_id]
            component: List[str] = []
            while stack:
                current = stack.pop()
                if current in lane_to_group:
                    continue
                lane_to_group[current] = next_group_id
                component.append(current)
                for neighbor_id in lateral_adjacency.get(current, ()):
                    if neighbor_id not in lane_to_group:
                        stack.append(neighbor_id)

            component.sort(key=lambda current_id: lane_by_id[current_id].anchor_distance)
            group_to_lanes[next_group_id] = tuple(component)
            next_group_id += 1

        outgoing_groups: Dict[int, set] = {group_id: set() for group_id in group_to_lanes}
        for record in lane_records:
            src_group = lane_to_group[record.lane_id]
            for exit_id in record.exit_ids:
                dst_group = lane_to_group.get(exit_id)
                if dst_group is None or dst_group == src_group:
                    continue
                outgoing_groups[src_group].add(dst_group)

        return lane_to_group, group_to_lanes, {group_id: tuple(sorted(targets)) for group_id, targets in outgoing_groups.items()}

    def _infer_route_groups_from_live_navigation(
        self,
        live_route_polylines: Sequence[np.ndarray],
        lane_records: Sequence[LaneRecord],
        lane_to_group: Dict[str, int],
    ) -> List[int]:
        group_sequence: List[int] = []
        for polyline in live_route_polylines:
            lane_id = self._match_global_polyline_to_lane_id(polyline, lane_records)
            if lane_id is None:
                continue
            group_sequence.append(lane_to_group[lane_id])
        return _dedupe_consecutive(group_sequence)

    def _infer_route_groups_from_future_path(
        self,
        future_route_world: np.ndarray,
        lane_records: Sequence[LaneRecord],
        lane_to_group: Dict[str, int],
    ) -> List[int]:
        if len(future_route_world) == 0:
            return []

        sample_count = min(16, len(future_route_world))
        sample_indices = sorted(set(np.linspace(0, len(future_route_world) - 1, num=sample_count, dtype=int).tolist()))

        group_sequence: List[int] = []
        for point_idx in sample_indices:
            point_xy = np.asarray(future_route_world[point_idx], dtype=np.float64)
            heading = None
            if len(future_route_world) > 1:
                if point_idx < len(future_route_world) - 1:
                    delta = np.asarray(future_route_world[point_idx + 1], dtype=np.float64) - point_xy
                else:
                    delta = point_xy - np.asarray(future_route_world[point_idx - 1], dtype=np.float64)
                if np.linalg.norm(delta) > 1e-4:
                    heading = float(math.atan2(delta[1], delta[0]))

            lane_id = self._match_point_to_lane_id(
                point_xy=point_xy,
                lane_records=lane_records,
                heading=heading,
                distance_threshold=8.0,
            )
            if lane_id is None:
                continue
            group_sequence.append(lane_to_group[lane_id])

        return _dedupe_consecutive(group_sequence)

    def _match_point_to_lane_id(
        self,
        point_xy: np.ndarray,
        lane_records: Sequence[LaneRecord],
        heading: Optional[float],
        distance_threshold: float,
    ) -> Optional[str]:
        point_xy = np.asarray(point_xy, dtype=np.float64)
        best_lane_id: Optional[str] = None
        best_score = float("inf")

        for record in lane_records:
            polyline = np.asarray(record.global_polyline, dtype=np.float64)
            if polyline.ndim != 2 or polyline.shape[0] == 0:
                continue
            deltas = polyline - point_xy[None, :]
            distances = np.linalg.norm(deltas, axis=1)
            nearest_index = int(np.argmin(distances))
            nearest_distance = float(distances[nearest_index])
            if nearest_distance > distance_threshold:
                continue

            score = nearest_distance
            if heading is not None:
                lane_heading = _polyline_heading_at(polyline, nearest_index)
                heading_error = abs(float(_wrap_to_pi(lane_heading - heading)))
                score += 1.5 * heading_error

            if score < best_score:
                best_score = score
                best_lane_id = record.lane_id

        return best_lane_id

    def _match_global_polyline_to_lane_id(
        self,
        polyline: np.ndarray,
        lane_records: Sequence[LaneRecord],
    ) -> Optional[str]:
        polyline = np.asarray(polyline, dtype=np.float64)
        if polyline.ndim != 2 or polyline.shape[0] < 2:
            return None

        sampled = _resample_polyline(polyline[:, :2], self.route_len)
        best_lane_id: Optional[str] = None
        best_score = float("inf")

        for record in lane_records:
            record_polyline = _resample_polyline(np.asarray(record.global_polyline, dtype=np.float64), self.route_len)
            score = float(np.linalg.norm(record_polyline - sampled, axis=1).mean())
            if score < best_score:
                best_score = score
                best_lane_id = record.lane_id

        if best_score > 8.0:
            return None
        return best_lane_id

    def _connect_route_group_sequence(
        self,
        route_group_sequence: Sequence[int],
        outgoing_groups: Dict[int, Tuple[int, ...]],
    ) -> List[int]:
        if not route_group_sequence:
            return []

        connected: List[int] = [int(route_group_sequence[0])]
        for target_group in route_group_sequence[1:]:
            target_group = int(target_group)
            current_group = connected[-1]
            if target_group == current_group:
                continue
            path = self._shortest_group_path(current_group, target_group, outgoing_groups, max_depth=6)
            if path is None:
                break
            connected.extend(path[1:])
        return _dedupe_consecutive(connected)

    def _shortest_group_path(
        self,
        start_group: int,
        target_group: int,
        outgoing_groups: Dict[int, Tuple[int, ...]],
        max_depth: int,
    ) -> Optional[List[int]]:
        if start_group == target_group:
            return [start_group]

        queue: List[int] = [start_group]
        parent: Dict[int, Optional[int]] = {start_group: None}
        depth: Dict[int, int] = {start_group: 0}

        for current in queue:
            current_depth = depth[current]
            if current_depth >= max_depth:
                continue
            for next_group in outgoing_groups.get(current, ()):
                if next_group in parent:
                    continue
                parent[next_group] = current
                depth[next_group] = current_depth + 1
                if next_group == target_group:
                    path = [target_group]
                    while parent[path[-1]] is not None:
                        path.append(parent[path[-1]])
                    path.reverse()
                    return path
                queue.append(next_group)

        return None

    def _extend_route_group_sequence(
        self,
        route_group_sequence: Sequence[int],
        lane_records: Sequence[LaneRecord],
        lane_to_group: Dict[str, int],
        group_to_lanes: Dict[int, Tuple[str, ...]],
        outgoing_groups: Dict[int, Tuple[int, ...]],
    ) -> List[int]:
        if not lane_records:
            return []

        lane_by_id = {record.lane_id: record for record in lane_records}

        if route_group_sequence:
            extended = list(route_group_sequence)
        else:
            seed_group = lane_to_group[lane_records[0].lane_id]
            extended = [seed_group]

        visited = set(extended)
        max_groups = max(self.route_num, len(extended))

        while len(extended) < max_groups:
            current_group = extended[-1]
            candidates = [group_id for group_id in outgoing_groups.get(current_group, ()) if group_id not in visited]
            if not candidates:
                break
            next_group = min(candidates, key=lambda group_id: self._group_route_score(group_id, group_to_lanes, lane_by_id))
            extended.append(next_group)
            visited.add(next_group)

        return extended

    def _group_route_score(
        self,
        group_id: int,
        group_to_lanes: Dict[int, Tuple[str, ...]],
        lane_by_id: Dict[str, LaneRecord],
    ) -> float:
        scores = [lane_by_id[lane_id].route_score for lane_id in group_to_lanes.get(group_id, ())]
        if not scores:
            return float("inf")
        return float(min(scores))

    def _collect_route_lane_records(
        self,
        lane_records: Sequence[LaneRecord],
        lane_to_group: Dict[str, int],
        route_group_sequence: Sequence[int],
    ) -> List[LaneRecord]:
        if not route_group_sequence:
            return []
        selected_groups = set(int(group_id) for group_id in route_group_sequence)
        return [record for record in lane_records if lane_to_group.get(record.lane_id) in selected_groups]

    def _ordered_live_route_lanes(self, env: Any) -> Tuple[Optional[Any], List[Any]]:
        agent = getattr(env, "agent", None)
        navigation = getattr(agent, "navigation", None)
        if agent is None or navigation is None:
            return None, []

        current_lanes = list(getattr(navigation, "current_ref_lanes", None) or [])
        next_lanes = list(getattr(navigation, "next_ref_lanes", None) or [])
        fallback_lane = getattr(agent, "lane", None)
        if fallback_lane is not None and fallback_lane not in current_lanes:
            current_lanes = [fallback_lane] + current_lanes

        ordered_lanes: List[Any] = []
        seen = set()
        for lane in current_lanes + next_lanes:
            if lane is None:
                continue
            lane_key = id(lane)
            if lane_key in seen:
                continue
            seen.add(lane_key)
            ordered_lanes.append(lane)
        return agent, ordered_lanes

    def _build_live_route_global_polylines(self, env: Any) -> List[np.ndarray]:
        agent, ordered_lanes = self._ordered_live_route_lanes(env)
        if agent is None:
            return []

        ego_position = np.asarray(agent.position, dtype=np.float64)
        polylines: List[np.ndarray] = []
        for lane in ordered_lanes:
            try:
                start_long = max(0.0, float(lane.local_coordinates(ego_position[:2])[0]))
            except Exception:
                start_long = 0.0
            lane_length = float(getattr(lane, "length", 0.0))
            if lane_length <= 1e-3:
                continue

            longs = np.linspace(start_long, lane_length, self.route_len)
            center = np.asarray([lane.position(float(s), 0.0) for s in longs], dtype=np.float64)
            if center.ndim != 2 or center.shape[0] < 2:
                continue
            polylines.append(center[:, :2])
        return polylines

    def _build_live_route_vectors(self, env: Any, anchor_pose: np.ndarray) -> List[np.ndarray]:
        agent, ordered_lanes = self._ordered_live_route_lanes(env)
        if agent is None:
            return []

        ego_position = np.asarray(agent.position, dtype=np.float64)
        vectors: List[np.ndarray] = []
        for lane in ordered_lanes:
            try:
                start_long = max(0.0, float(lane.local_coordinates(ego_position[:2])[0]))
            except Exception:
                start_long = 0.0
            lane_length = float(getattr(lane, "length", 0.0))
            if lane_length <= 1e-3:
                continue

            longs = np.linspace(start_long, lane_length, self.route_len)
            center = np.asarray([lane.position(float(s), 0.0) for s in longs], dtype=np.float64)
            if center.ndim != 2 or center.shape[0] < 2:
                continue
            center = center[:, :2]
            local_center = _world_to_local_points(center, anchor_pose[:2], anchor_pose[2])
            local_tangent = _compute_polyline_vectors(local_center)
            width = float(getattr(lane, "width", 3.5))
            normals = _compute_normals(local_center)
            left_offsets = normals * (0.5 * width)
            right_offsets = -normals * (0.5 * width)

            vector = np.zeros((self.route_len, 12), dtype=np.float32)
            vector[:, 0:2] = local_center.astype(np.float32)
            vector[:, 2:4] = local_tangent.astype(np.float32)
            vector[:, 4:6] = left_offsets.astype(np.float32)
            vector[:, 6:8] = right_offsets.astype(np.float32)
            vector[:, 8:12] = _TL_UNKNOWN[None, :]
            vectors.append(vector)
        return vectors

    def _build_ego_future(
        self,
        scenario_data: Dict[str, Any],
        sdc_id: str,
        anchor_pose: np.ndarray,
        frame_id: int,
    ) -> np.ndarray:
        track = scenario_data["tracks"][sdc_id]
        positions_bridge = np.asarray(
            track["state"]["position"][frame_id + 1: frame_id + 1 + self.future_steps, :2],
            dtype=np.float64,
        )
        headings = np.asarray(track["state"]["heading"][frame_id + 1: frame_id + 1 + self.future_steps], dtype=np.float64)
        positions = self._bridge_positions_to_rear_axle(positions_bridge, headings)
        local_positions = _world_to_local_points(positions, anchor_pose[:2], anchor_pose[2])
        local_heading = _wrap_to_pi(headings - anchor_pose[2])
        local_heading = _build_future_heading(local_positions, local_heading)
        return np.concatenate([local_positions, local_heading[:, None]], axis=1)

    def _build_neighbor_future(
        self,
        scenario_data: Dict[str, Any],
        anchor_pose: np.ndarray,
        frame_id: int,
        selected_track_ids: Sequence[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        future = np.zeros((self.predicted_neighbor_num, self.future_steps, 3), dtype=np.float32)
        future_mask = np.ones((self.predicted_neighbor_num, self.future_steps), dtype=np.bool_)

        for agent_idx, track_id in enumerate(selected_track_ids[: self.predicted_neighbor_num]):
            track = scenario_data["tracks"].get(track_id)
            if track is None:
                continue
            positions = np.asarray(track["state"]["position"], dtype=np.float64)
            headings = np.asarray(track["state"]["heading"], dtype=np.float64)
            valid = np.asarray(track["state"].get("valid", []), dtype=np.float64)

            local_positions = np.zeros((self.future_steps, 2), dtype=np.float64)
            local_headings = np.zeros((self.future_steps,), dtype=np.float64)
            for step in range(self.future_steps):
                src_idx = frame_id + 1 + step
                if src_idx >= len(valid) or valid[src_idx] <= 0.5:
                    continue
                local_positions[step] = _world_to_local_points(
                    positions[src_idx: src_idx + 1, :2], anchor_pose[:2], anchor_pose[2]
                )[0]
                local_headings[step] = float(_wrap_to_pi(headings[src_idx] - anchor_pose[2]))
                future_mask[agent_idx, step] = False

            if (~future_mask[agent_idx]).any():
                local_headings = _build_future_heading(local_positions, local_headings)
                future[agent_idx, :, :2] = local_positions.astype(np.float32)
                future[agent_idx, :, 2] = local_headings.astype(np.float32)

        return future, future_mask
