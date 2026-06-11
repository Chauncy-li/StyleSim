"""
Run batch evaluation for multiple models and export one comparison table.

This script wraps `batch_evaluator.py` / `BatchEvaluator` so we can:
1. run several model adapters sequentially on the same scenario root
2. keep per-model outputs in separate folders
3. collect one `model_comparison.csv` with the final averaged metrics
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

from bridgesim.evaluation.batch_evaluator import BatchEvaluator
from bridgesim.evaluation.models.model_configs import (
    CKPT_BASE,
    get_model_info,
    normalize_model_type,
)


SUMMARY_COLUMNS = ["DS", "EPDMS_no_ep", "RC", "NC", "DAC", "DDC", "TL", "TTC", "LK", "HC", "EC"]

PRESETS = {
    # Purely local defaults that should not need extra backbone downloads.
    "stable_local": ["uniad", "vad", "tcp", "lead_navsim", "egomlp"],
    # Models with local checkpoints that often trigger first-run backbone downloads.
    "network_backed": ["rap", "transfuser", "ltf", "drivor"],
    # Try every model family for which the user already has a local checkpoint tree.
    "downloaded_ckpts": [
        "uniad", "vad", "tcp", "lead_navsim", "egomlp",
        "rap", "transfuser", "ltf", "drivor",
        "diffusiondrive", "diffusiondrivev2",
    ],
}

MODEL_OVERRIDES: Dict[str, Dict] = {
    "egomlp": {
        "config_key": "ego_mlp",
        "cli_model_type": "egomlp",
        "notes": "Blind baseline using only ego state.",
    },
    "tcp": {
        "planner_type": "only_traj",
    },
    "rap": {
        "image_source": "metadrive",
        "notes": "May download a DINO backbone on first run.",
    },
    "transfuser": {
        "notes": "May download pretrained timm backbones on first run.",
    },
    "ltf": {
        "notes": "Shares TransFuser-style pretrained backbones on first run.",
    },
    "drivor": {
        "notes": "May download DINOv2/timm backbones on first run.",
    },
    "diffusiondrive": {
        "plan_anchor_path": str(Path(CKPT_BASE) / "navsimv2" / "DiffusionDrive" / "kmeans_navsim_traj_20.npy"),
        "notes": "Requires local plan anchors and may download pretrained backbones.",
    },
    "diffusiondrivev2": {
        "plan_anchor_path": str(Path(CKPT_BASE) / "navsimv2" / "DiffusionDriveV2" / "kmeans_navsim_traj_20.npy"),
        "notes": "Requires local plan anchors and may download pretrained backbones.",
    },
}


def _scenario_count(scenario_root: Path) -> int:
    return len([d for d in sorted(scenario_root.iterdir()) if d.is_dir() and not d.name.startswith(".")])


def _effective_scenario_count(scenario_root: Path, scenario_limit: int = None) -> int:
    total = _scenario_count(scenario_root)
    if scenario_limit is None:
        return total
    return min(total, scenario_limit)


def _resolve_requested_models(args) -> List[str]:
    if args.models:
        models = args.models
    else:
        models = PRESETS[args.preset]

    resolved = []
    seen = set()
    for model in models:
        lowered = model.lower()
        if lowered not in seen:
            resolved.append(lowered)
            seen.add(lowered)
    return resolved


def _build_model_spec(model_name: str) -> Dict:
    overrides = MODEL_OVERRIDES.get(model_name, {})
    config_key = overrides.get("config_key", normalize_model_type(model_name))
    info = dict(get_model_info(config_key))

    spec = {
        "requested_model": model_name,
        "config_key": config_key,
        "cli_model_type": overrides.get("cli_model_type", model_name),
        "checkpoint": info.get("checkpoint"),
        "config": info.get("config"),
        "plan_anchor_path": overrides.get("plan_anchor_path", info.get("plan_anchor_path")),
        "planner_type": overrides.get("planner_type", "only_traj"),
        "image_source": overrides.get("image_source", "metadrive"),
        "num_cameras": overrides.get("num_cameras", 4),
        "image_size": overrides.get("image_size", [512, 288]),
        "num_poses": overrides.get("num_poses", 8),
        "use_lidar": overrides.get("use_lidar", False),
        "description": info.get("description", ""),
        "notes": overrides.get("notes", ""),
    }
    return spec


def _check_required_paths(spec: Dict) -> List[str]:
    missing = []
    for key in ("checkpoint", "config", "plan_anchor_path"):
        value = spec.get(key)
        if not value:
            continue
        # External IDs like nvidia/Alpamayo-R1-10B are not local paths.
        if "/" in str(value) and not str(value).startswith("/"):
            continue
        if not Path(value).exists():
            missing.append(f"{key}: {value}")
    return missing


def _read_model_average(summary_csv: Path) -> Tuple[Dict[str, str], int]:
    if not summary_csv.exists():
        return {}, 0

    rows = []
    average = {}
    with open(summary_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("scenario") == "AVERAGE":
                average = row
            else:
                rows.append(row)
    return average, len(rows)


def _has_meaningful_average(average_row: Dict[str, str]) -> bool:
    if not average_row:
        return False
    return any((average_row.get(metric) or "").strip() for metric in SUMMARY_COLUMNS)


def _write_comparison_csv(rows: List[Dict], output_path: Path) -> None:
    fieldnames = [
        "model",
        "status",
        "successful_scenarios",
        "total_scenarios",
        "elapsed_sec",
        "output_dir",
        "notes",
        "error",
        *SUMMARY_COLUMNS,
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _print_summary(rows: List[Dict]) -> None:
    print("\n" + "=" * 88)
    print("Multi-Model Comparison")
    print("=" * 88)
    header = f"{'model':<16} {'status':<10} {'ok/total':<10} {'DS':>8} {'RC':>8} {'EPDMS':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['model']:<16} "
            f"{row['status']:<10} "
            f"{str(row.get('successful_scenarios', '')) + '/' + str(row.get('total_scenarios', '')):<10} "
            f"{row.get('DS', ''):>8} "
            f"{row.get('RC', ''):>8} "
            f"{row.get('EPDMS_no_ep', ''):>10}"
        )
    print("=" * 88)


def main():
    parser = argparse.ArgumentParser(description="Run batch evaluation for multiple models and export one score table.")
    parser.add_argument("--scenario-root", type=str, default=None, help="Root directory containing scenario folders")
    parser.add_argument("--output-dir", type=str, default=None, help="Root directory for all model outputs")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Explicit list of models to run. Overrides --preset.")
    parser.add_argument("--preset", type=str, default="stable_local", choices=sorted(PRESETS.keys()),
                        help="Predefined model set to run when --models is not provided")
    parser.add_argument("--list-models", action="store_true",
                        help="Print preset/model information and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve paths and print what would run, but do not launch evaluations")
    parser.add_argument("--traffic-mode", type=str, default="log_replay", choices=["no_traffic", "log_replay", "IDM"])
    parser.add_argument("--controller", type=str, default="pure_pursuit", choices=["pid", "pure_pursuit"])
    parser.add_argument("--replan-rate", type=int, default=1)
    parser.add_argument("--sim-dt", type=float, default=0.1)
    parser.add_argument("--ego-replay-frames", type=int, default=0)
    parser.add_argument("--eval-frames", type=int, default=None)
    parser.add_argument("--score-start-frame", type=int, default=None)
    parser.add_argument("--eval-mode", type=str, default="closed_loop", choices=["closed_loop", "open_loop"])
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--scenario-limit", type=int, default=None,
                        help="Optional limit on the number of scenarios to evaluate per model. "
                             "Useful for smoke tests such as 10 scenarios.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--enable-vis", action="store_true")
    parser.add_argument("--save-perframe", action="store_true", default=True)
    parser.add_argument("--no-save-perframe", dest="save_perframe", action="store_false")
    args = parser.parse_args()

    if args.list_models:
        print("Presets:")
        for preset_name, models in PRESETS.items():
            print(f"  {preset_name}: {', '.join(models)}")
        print("\nModel notes:")
        for model_name in sorted(set(sum(PRESETS.values(), []))):
            spec = _build_model_spec(model_name)
            note = spec["notes"] or spec["description"]
            print(f"  {model_name}: {note}")
        return

    if not args.scenario_root or not args.output_dir:
        parser.error("--scenario-root and --output-dir are required unless --list-models is used")

    scenario_root = Path(args.scenario_root).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    requested_models = _resolve_requested_models(args)
    total_scenarios = _effective_scenario_count(scenario_root, args.scenario_limit)
    comparison_rows: List[Dict] = []

    manifest = {
        "scenario_root": str(scenario_root),
        "output_dir": str(output_root),
        "preset": args.preset,
        "models": requested_models,
        "traffic_mode": args.traffic_mode,
        "controller": args.controller,
        "replan_rate": args.replan_rate,
        "sim_dt": args.sim_dt,
        "ego_replay_frames": args.ego_replay_frames,
        "eval_frames": args.eval_frames,
        "score_start_frame": args.score_start_frame,
        "eval_mode": args.eval_mode,
        "scenario_limit": args.scenario_limit,
    }

    for model_name in requested_models:
        spec = _build_model_spec(model_name)
        model_output_dir = output_root / spec["cli_model_type"]
        missing = _check_required_paths(spec)

        row = {
            "model": spec["cli_model_type"],
            "status": "pending",
            "successful_scenarios": 0,
            "total_scenarios": total_scenarios,
            "elapsed_sec": "",
            "output_dir": str(model_output_dir),
            "notes": spec["notes"],
            "error": "",
        }

        if missing:
            row["status"] = "skipped"
            row["error"] = "; ".join(missing)
            comparison_rows.append(row)
            continue

        if args.dry_run:
            row["status"] = "dry_run"
            comparison_rows.append(row)
            continue

        start_time = time.time()
        try:
            evaluator = BatchEvaluator(
                model_type=spec["cli_model_type"],
                checkpoint_path=spec["checkpoint"],
                scenario_root=str(scenario_root),
                output_root=str(model_output_dir),
                config_path=spec["config"],
                planner_type=spec["planner_type"],
                image_source=spec["image_source"],
                plan_anchor_path=spec["plan_anchor_path"],
                traffic_mode=args.traffic_mode,
                max_workers=args.max_workers,
                resume=args.resume,
                save_perframe=args.save_perframe,
                controller_type=args.controller,
                replan_rate=args.replan_rate,
                sim_dt=args.sim_dt,
                ego_replay_frames=args.ego_replay_frames,
                eval_frames=args.eval_frames,
                score_start_frame=args.score_start_frame,
                eval_mode=args.eval_mode,
                enable_vis=args.enable_vis,
                num_cameras=spec["num_cameras"],
                image_size=spec["image_size"],
                num_poses=spec["num_poses"],
                use_lidar=spec["use_lidar"],
                scenario_limit=args.scenario_limit,
            )
            evaluator.run()
            row["elapsed_sec"] = f"{time.time() - start_time:.1f}"

            summary_csv = model_output_dir / "batch_driving_score_summary.csv"
            average_row, ok_count = _read_model_average(summary_csv)
            row["successful_scenarios"] = ok_count
            if _has_meaningful_average(average_row) and ok_count > 0:
                row["status"] = "success"
                for metric in SUMMARY_COLUMNS:
                    row[metric] = average_row.get(metric, "")
            else:
                row["status"] = "error"
                row["error"] = f"Missing or unreadable summary CSV: {summary_csv}"
        except Exception as exc:
            row["status"] = "error"
            row["elapsed_sec"] = f"{time.time() - start_time:.1f}"
            row["error"] = str(exc)

        comparison_rows.append(row)

    comparison_csv = output_root / "model_comparison.csv"
    _write_comparison_csv(comparison_rows, comparison_csv)

    manifest_path = output_root / "model_comparison_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest | {"results": comparison_rows}, f, indent=2)

    _print_summary(comparison_rows)
    print(f"\nComparison CSV written to: {comparison_csv}")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
