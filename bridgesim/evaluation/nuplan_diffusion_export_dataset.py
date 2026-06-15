"""
Export BridgeSim scenarios into the .npz schema used by the external
nuPlan diffusion planner training code.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
METADRIVE_ROOT = ROOT / "metadrive"
for path in (ROOT, METADRIVE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from bridgesim.evaluation.nuplan_diffusion.feature_builder import BridgeSimNuPlanFeatureBuilder


def _feature_rows(values: np.ndarray, valid_mask: Optional[np.ndarray]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        return values[None, :]

    feature_dim = values.shape[-1]
    rows = values.reshape(-1, feature_dim)
    if valid_mask is None:
        return rows

    mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    if rows.shape[0] != mask.shape[0]:
        raise ValueError(f"Mask shape mismatch: rows={rows.shape}, mask={mask.shape}")
    return rows[mask]


@dataclass
class _StatBucket:
    total: np.ndarray
    total_sq: np.ndarray
    count: int


class NormalizationAccumulator:
    def __init__(self) -> None:
        self._buckets: Dict[str, _StatBucket] = {}

    def update(self, name: str, values: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> None:
        rows = _feature_rows(values, valid_mask)
        if rows.size == 0:
            return
        if name not in self._buckets:
            self._buckets[name] = _StatBucket(
                total=np.zeros((rows.shape[1],), dtype=np.float64),
                total_sq=np.zeros((rows.shape[1],), dtype=np.float64),
                count=0,
            )
        bucket = self._buckets[name]
        bucket.total += rows.sum(axis=0)
        bucket.total_sq += (rows ** 2).sum(axis=0)
        bucket.count += rows.shape[0]

    def finalize(self) -> Dict[str, Dict[str, List[float]]]:
        result: Dict[str, Dict[str, List[float]]] = {}
        for name, bucket in self._buckets.items():
            mean = bucket.total / max(bucket.count, 1)
            var = bucket.total_sq / max(bucket.count, 1) - mean ** 2
            std = np.sqrt(np.maximum(var, 1e-6))
            result[name] = {
                "mean": mean.astype(np.float32).tolist(),
                "std": std.astype(np.float32).tolist(),
            }
        return result


def _load_scenario(scenario_dir: Path) -> Dict:
    inner_dir = scenario_dir / f"{scenario_dir.name}_0"
    pkl_path = inner_dir / f"{scenario_dir.name}.pkl"
    with open(pkl_path, "rb") as handle:
        return pickle.load(handle)


def _scenario_dirs(root: Path) -> List[Path]:
    return sorted([path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export BridgeSim scenarios into nuPlan diffusion training .npz files")
    parser.add_argument("--scenario-root", required=True, help="Path to BridgeData root")
    parser.add_argument("--output-dir", required=True, help="Directory to write .npz files and split manifests")
    parser.add_argument("--scenario-limit", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--agent-num", type=int, default=32)
    parser.add_argument("--predicted-neighbor-num", type=int, default=10)
    parser.add_argument("--static-objects-num", type=int, default=5)
    parser.add_argument("--lane-num", type=int, default=70)
    parser.add_argument("--lane-len", type=int, default=20)
    parser.add_argument("--route-num", type=int, default=25)
    parser.add_argument("--route-len", type=int, default=20)
    parser.add_argument("--future-len", type=int, default=80)
    parser.add_argument("--time-len", type=int, default=21)
    parser.add_argument("--bridge-ego-reference", default="rear_axle", choices=["center", "rear_axle", "cog"])
    return parser.parse_args()


def _future_to_state_normalizer_inputs(ego_future: np.ndarray, neighbor_future: np.ndarray, neighbor_mask: np.ndarray):
    ego = np.concatenate(
        [ego_future[:, :2], np.stack([np.cos(ego_future[:, 2]), np.sin(ego_future[:, 2])], axis=-1)],
        axis=-1,
    )
    neighbor = np.concatenate(
        [
            neighbor_future[:, :, :2],
            np.stack([np.cos(neighbor_future[:, :, 2]), np.sin(neighbor_future[:, :, 2])], axis=-1),
        ],
        axis=-1,
    )
    valid_mask = ~neighbor_mask
    return ego.astype(np.float32), neighbor.astype(np.float32), valid_mask


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> int:
    args = _parse_args()

    builder = BridgeSimNuPlanFeatureBuilder(
        agent_num=args.agent_num,
        predicted_neighbor_num=args.predicted_neighbor_num,
        static_objects_num=args.static_objects_num,
        lane_num=args.lane_num,
        lane_len=args.lane_len,
        route_num=args.route_num,
        route_len=args.route_len,
        future_steps=args.future_len,
        past_steps=args.time_len,
        bridge_ego_reference=args.bridge_ego_reference,
    )

    scenario_root = Path(args.scenario_root)
    output_dir = Path(args.output_dir)
    npz_dir = output_dir / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

    scenario_dirs = _scenario_dirs(scenario_root)
    if args.scenario_limit is not None:
        scenario_dirs = scenario_dirs[: args.scenario_limit]

    normalization = NormalizationAccumulator()
    exported_files: List[str] = []

    for scenario_idx, scenario_dir in enumerate(scenario_dirs, start=1):
        print(f"[nuplan_export] [{scenario_idx}/{len(scenario_dirs)}] {scenario_dir.name}", flush=True)
        try:
            scenario_data = _load_scenario(scenario_dir)
        except Exception as exc:
            print(f"[nuplan_export] skip {scenario_dir.name}: {exc}", flush=True)
            continue

        frame_ids = builder.valid_export_frames(scenario_data)[:: max(args.frame_stride, 1)]
        for frame_id in frame_ids:
            sample = builder.build_training_sample(scenario_data, frame_id)
            file_name = f"{scenario_dir.name}__frame_{frame_id:05d}.npz"
            file_path = npz_dir / file_name
            np.savez_compressed(file_path, **sample)
            exported_files.append(file_name)

            normalization.update("ego_current_state", sample["ego_current_state"], None)
            normalization.update("neighbor_agents_past", sample["neighbor_agents_past"], sample["neighbor_agents_past_mask"])
            normalization.update("lanes", sample["lanes"], sample["lanes_mask"])
            normalization.update("lanes_speed_limit", sample["lanes_speed_limit"], sample["lanes_mask"].any(axis=1))
            normalization.update("route_lanes", sample["route_lanes"], sample["route_lanes_mask"])
            normalization.update("route_lanes_speed_limit", sample["route_lanes_speed_limit"], sample["route_lanes_mask"].any(axis=1))
            normalization.update("static_objects", sample["static_objects"], np.any(np.abs(sample["static_objects"]) > 0.0, axis=1))

            ego_norm, neighbor_norm, neighbor_valid = _future_to_state_normalizer_inputs(
                sample["ego_agent_future"],
                sample["neighbor_agents_future"],
                sample["neighbor_agents_future_mask"],
            )
            normalization.update("ego", ego_norm, None)
            normalization.update("neighbor", neighbor_norm, neighbor_valid)

    rng = random.Random(args.seed)
    rng.shuffle(exported_files)
    val_count = int(round(len(exported_files) * args.val_ratio))
    val_files = exported_files[:val_count]
    train_files = exported_files[val_count:]

    _write_json(output_dir / "all.json", exported_files)
    _write_json(output_dir / "train.json", train_files)
    _write_json(output_dir / "val.json", val_files)
    _write_json(output_dir / "normalization_bridgesim.json", normalization.finalize())

    manifest = {
        "scenario_root": str(scenario_root),
        "output_dir": str(output_dir),
        "num_scenarios": len(scenario_dirs),
        "num_samples": len(exported_files),
        "frame_stride": args.frame_stride,
        "val_ratio": args.val_ratio,
        "agent_num": args.agent_num,
        "predicted_neighbor_num": args.predicted_neighbor_num,
        "lane_num": args.lane_num,
        "route_num": args.route_num,
        "future_len": args.future_len,
        "time_len": args.time_len,
        "bridge_ego_reference": args.bridge_ego_reference,
    }
    _write_json(output_dir / "export_manifest.json", manifest)

    print(
        f"[nuplan_export] wrote {len(exported_files)} samples from {len(scenario_dirs)} scenarios to {output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
