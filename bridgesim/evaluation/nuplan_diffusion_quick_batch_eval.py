"""
Quick diverse-subset batch evaluator for the external nuPlan diffusion planner.

This mirrors the existing quick_batch_eval.py flow while staying fully isolated
from the current unified evaluator / model registry.
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _extract_features(scenario_dir: Path) -> Optional[Dict]:
    inner_dirs = [d for d in scenario_dir.iterdir() if d.is_dir() and d.name.endswith("_0")]
    if not inner_dirs:
        return None
    pkl_path = inner_dirs[0] / f"{scenario_dir.name}.pkl"
    if not pkl_path.exists():
        return None

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    sdc_id = data["metadata"]["sdc_id"]
    tracks = data["tracks"]

    type_counts: Dict[str, int] = {}
    for track in tracks.values():
        track_type = str(track.get("type", "UNKNOWN"))
        type_counts[track_type] = type_counts.get(track_type, 0) + 1

    ego = tracks.get(sdc_id, {})
    pos = ego.get("state", {}).get("position")
    valid = ego.get("state", {}).get("valid")
    ego_dist = 0.0
    if pos is not None and valid is not None:
        pos_arr = np.asarray(pos)
        valid_arr = np.asarray(valid, dtype=bool)
        valid_pos = pos_arr[valid_arr]
        if len(valid_pos) >= 2:
            ego_dist = float(np.sum(np.linalg.norm(np.diff(valid_pos[:, :2], axis=0), axis=1)))

    has_tl = len(data.get("dynamic_map_states", {})) > 0
    return {
        "path": scenario_dir,
        "n_vehicles": type_counts.get("VEHICLE", 0),
        "n_pedestrians": type_counts.get("PEDESTRIAN", 0),
        "n_cyclists": type_counts.get("CYCLIST", 0),
        "has_tl": int(has_tl),
        "ego_dist": ego_dist,
    }


def scan_scenarios(scenario_root: Path) -> List[Dict]:
    dirs = sorted([d for d in scenario_root.iterdir() if d.is_dir() and not d.name.startswith(".")])
    print(f"[nuplan_quick] Scanning {len(dirs)} scenarios for features...", flush=True)
    features = []
    for d in dirs:
        try:
            result = _extract_features(d)
            if result is not None:
                features.append(result)
        except Exception:
            pass
    print(f"[nuplan_quick] Successfully scanned {len(features)} scenarios.", flush=True)
    return features


def _feature_matrix(scenarios: List[Dict]) -> np.ndarray:
    mat = np.array(
        [
            [s["n_vehicles"], s["n_pedestrians"], s["n_cyclists"], s["has_tl"], s["ego_dist"]]
            for s in scenarios
        ],
        dtype=float,
    )
    col_min = mat.min(axis=0)
    col_max = mat.max(axis=0)
    span = np.where(col_max - col_min < 1e-9, 1.0, col_max - col_min)
    return (mat - col_min) / span


def greedy_diverse_sample(scenarios: List[Dict], n: int, seed: int = 42) -> List[Dict]:
    if n >= len(scenarios):
        return list(scenarios)

    mat = _feature_matrix(scenarios)
    rng = np.random.default_rng(seed)

    selected_indices = [int(rng.integers(len(scenarios)))]
    remaining = set(range(len(scenarios))) - set(selected_indices)

    while len(selected_indices) < n and remaining:
        sel_mat = mat[selected_indices]
        remaining_list = list(remaining)
        rem_mat = mat[remaining_list]
        dists = np.min(
            np.sqrt(((rem_mat[:, None, :] - sel_mat[None, :, :]) ** 2).sum(axis=2)),
            axis=1,
        )
        best_idx = remaining_list[int(np.argmax(dists))]
        selected_indices.append(best_idx)
        remaining.discard(best_idx)

    return [scenarios[i] for i in selected_indices]


def _build_cmd(scenario_path: Path, args) -> List[str]:
    evaluator_path = Path(__file__).with_name("nuplan_diffusion_evaluator.py")
    cmd = [
        sys.executable,
        str(evaluator_path),
        "--checkpoint",
        args.checkpoint,
        "--nuplan-args",
        args.nuplan_args,
        "--scenario-path",
        str(scenario_path),
        "--output-dir",
        args.output_dir,
        "--traffic-mode",
        args.traffic_mode,
        "--controller",
        args.controller,
        "--replan-rate",
        str(args.replan_rate),
        "--sim-dt",
        str(args.sim_dt),
        "--ego-replay-frames",
        str(args.ego_replay_frames),
        "--eval-mode",
        args.eval_mode,
    ]
    if args.nuplan_root:
        cmd.extend(["--nuplan-root", args.nuplan_root])
    if args.nuplan_devkit_root:
        cmd.extend(["--nuplan-devkit-root", args.nuplan_devkit_root])
    if args.device:
        cmd.extend(["--device", args.device])
    if args.eval_frames is not None:
        cmd.extend(["--eval-frames", str(args.eval_frames)])
    if args.score_start_frame is not None:
        cmd.extend(["--score-start-frame", str(args.score_start_frame)])
    if args.bridge_ego_reference:
        cmd.extend(["--bridge-ego-reference", args.bridge_ego_reference])
    if not args.save_perframe:
        cmd.append("--no-save-perframe")
    if args.enable_vis:
        cmd.append("--enable-vis")
    return cmd


def run_scenario(scenario_path: Path, args) -> Dict:
    cmd = _build_cmd(scenario_path, args)
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).parent.parent.parent,
            timeout=args.timeout_seconds,
        )
        duration = time.time() - start
        if result.returncode == 0:
            summary_csv = Path(args.output_dir) / scenario_path.name / "driving_score_summary.csv"
            if summary_csv.exists():
                with open(summary_csv, newline="") as f:
                    for row in csv_mod.DictReader(f):
                        if row.get("frame_id") == "AVERAGE":
                            return {"status": "success", "duration": duration, "scores": dict(row)}
            return {"status": "error", "duration": duration, "error": "no summary CSV"}
        return {"status": "failed", "duration": duration, "exit_code": result.returncode}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "duration": time.time() - start}
    except Exception as exc:
        return {"status": "error", "duration": time.time() - start, "error": str(exc)}


COLUMNS = ["DS", "EPDMS_no_ep", "RC", "NC", "DAC", "DDC", "TL", "TTC", "LK", "HC", "EC"]


def print_summary(results: List[Tuple[str, Dict]]) -> None:
    totals = {c: [] for c in COLUMNS}
    for name, result in results:
        if result["status"] != "success":
            print(f"  SKIP {name}: {result.get('error', result['status'])}")
            continue
        for col in COLUMNS:
            try:
                totals[col].append(float(result["scores"][col]))
            except (KeyError, ValueError, TypeError):
                pass

    print(f"\n{'=' * 60}")
    print("NuPlan Diffusion Quick Eval Complete! Summary (Averages)")
    print(f"{'=' * 60}")
    for col in COLUMNS:
        vals = totals[col]
        print(f"  {col:<12} {np.mean(vals):.6f}" if vals else f"  {col:<12} N/A")
    ok_count = sum(1 for _, result in results if result["status"] == "success")
    print(f"{'=' * 60}")
    print(f"  Scenarios evaluated: {ok_count} / {len(results)}")


def export_csv(results: List[Tuple[str, Dict]], output_dir: str) -> None:
    out_path = Path(output_dir) / "quick_batch_driving_score_summary.csv"
    cols = ["scenario"] + COLUMNS
    totals = {c: [] for c in COLUMNS}
    rows = []

    for name, result in results:
        if result["status"] != "success":
            continue
        scores = result.get("scores", {})
        row = {"scenario": name}
        for col in COLUMNS:
            val = scores.get(col, "")
            row[col] = val
            try:
                totals[col].append(float(val))
            except (ValueError, TypeError):
                pass
        rows.append(row)

    avg_row = {"scenario": "AVERAGE"}
    for col in COLUMNS:
        vals = totals[col]
        avg_row[col] = f"{np.mean(vals):.6f}" if vals else ""

    with open(out_path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(avg_row)
    print(f"[nuplan_quick] Results saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick diverse-subset batch evaluator for nuPlan diffusion")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--nuplan-args", required=True)
    parser.add_argument("--nuplan-root", default=None)
    parser.add_argument("--nuplan-devkit-root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--scenario-root", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--num-scenarios", type=int, default=10)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=int, default=1800)

    parser.add_argument("--traffic-mode", default="log_replay", choices=["no_traffic", "log_replay", "IDM"])
    parser.add_argument("--controller", default="pure_pursuit", choices=["pid", "pure_pursuit"])
    parser.add_argument("--replan-rate", type=int, default=1)
    parser.add_argument("--sim-dt", type=float, default=0.1)
    parser.add_argument("--ego-replay-frames", type=int, default=0)
    parser.add_argument("--eval-frames", type=int, default=None)
    parser.add_argument("--score-start-frame", type=int, default=None)
    parser.add_argument("--eval-mode", default="closed_loop", choices=["closed_loop", "open_loop"])
    parser.add_argument("--bridge-ego-reference", default="rear_axle", choices=["center", "rear_axle", "cog"])
    parser.add_argument("--save-perframe", action="store_true", default=True)
    parser.add_argument("--no-save-perframe", dest="save_perframe", action="store_false")
    parser.add_argument("--enable-vis", action="store_true")

    args = parser.parse_args()

    scenario_root = Path(args.scenario_root)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    all_scenarios = scan_scenarios(scenario_root)
    if not all_scenarios:
        print("[nuplan_quick] No scenarios found.")
        sys.exit(1)

    selected = greedy_diverse_sample(all_scenarios, args.num_scenarios, seed=args.selection_seed)

    print(f"\n[nuplan_quick] Selected {len(selected)} scenarios:")
    for scenario in selected:
        print(
            f"  {scenario['path'].name}  veh={scenario['n_vehicles']:3d} "
            f"ped={scenario['n_pedestrians']:3d} cyc={scenario['n_cyclists']} "
            f"tl={bool(scenario['has_tl'])} dist={scenario['ego_dist']:.0f}m"
        )
    print()

    results: List[Tuple[str, Dict]] = []
    for idx, scenario in enumerate(selected):
        name = scenario["path"].name
        print(f"[{idx + 1:2d}/{len(selected)}] {name} ...", end=" ", flush=True)
        result = run_scenario(scenario["path"], args)
        duration = result.get("duration", 0.0)
        if result["status"] == "success":
            ds = result.get("scores", {}).get("DS", "?")
            rc = result.get("scores", {}).get("RC", "?")
            print(f"OK  DS={ds}  RC={rc}  ({duration:.0f}s)")
        else:
            print(f"FAIL ({result['status']})  ({duration:.0f}s)")
        results.append((name, result))

    print_summary(results)
    export_csv(results, args.output_dir)


if __name__ == "__main__":
    main()
