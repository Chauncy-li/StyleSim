"""
Standalone evaluator for the external nuPlan diffusion planner.

This file is intentionally separate from unified_evaluator.py so existing BridgeSim
evaluation flows remain untouched.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bridgesim.evaluation.core.base_evaluator import BaseEvaluator, _silence
from bridgesim.evaluation.nuplan_diffusion.adapter import NuPlanDiffusionBridgeAdapter


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate external nuPlan diffusion planner in BridgeSim")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained nuPlan diffusion checkpoint")
    parser.add_argument("--nuplan-args", dest="config", default=None, help="Path to nuPlan args.json")
    parser.add_argument("--nuplan-root", default=None, help="Repo root containing the external nuplan_baseline package")
    parser.add_argument("--nuplan-devkit-root", default=None, help="Optional explicit nuplan-devkit root")
    parser.add_argument("--device", default=None, help="cuda / cpu. Defaults to cuda when available.")

    parser.add_argument("--scenario-path", required=True, help="Path to one BridgeSim scenario directory")
    parser.add_argument("--output-dir", required=True, help="Directory for evaluation outputs")
    parser.add_argument("--traffic-mode", default="log_replay", choices=["no_traffic", "log_replay", "IDM"])
    parser.add_argument("--enable-vis", action="store_true")
    parser.add_argument("--no-save-perframe", dest="save_perframe", action="store_false")
    parser.set_defaults(save_perframe=True)
    parser.add_argument("--eval-mode", default="closed_loop", choices=["closed_loop", "open_loop"])
    parser.add_argument("--controller", default="pure_pursuit", choices=["pure_pursuit", "pid"])
    parser.add_argument("--replan-rate", type=int, default=1)
    parser.add_argument("--sim-dt", type=float, default=0.1)
    parser.add_argument("--ego-replay-frames", type=int, default=0)
    parser.add_argument("--eval-frames", type=int, default=None)
    parser.add_argument("--score-start-frame", type=int, default=None)

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
    return parser


def main() -> int:
    args = build_argparser().parse_args()

    adapter = NuPlanDiffusionBridgeAdapter(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        nuplan_root=args.nuplan_root,
        nuplan_devkit_root=args.nuplan_devkit_root,
        device=args.device,
        agent_num=args.agent_num,
        predicted_neighbor_num=args.predicted_neighbor_num,
        static_objects_num=args.static_objects_num,
        lane_num=args.lane_num,
        lane_len=args.lane_len,
        route_num=args.route_num,
        route_len=args.route_len,
        future_len=args.future_len,
        time_len=args.time_len,
        bridge_ego_reference=args.bridge_ego_reference,
    )

    print("Loading external nuPlan diffusion planner...", flush=True)
    with _silence():
        adapter.load_model()
    print("Model loaded. Starting BridgeSim evaluation.", flush=True)

    evaluator = BaseEvaluator(
        model_adapter=adapter,
        scenario_path=args.scenario_path,
        output_dir=args.output_dir,
        traffic_mode=args.traffic_mode,
        enable_vis=args.enable_vis,
        save_perframe=args.save_perframe,
        eval_mode=args.eval_mode,
        controller_type=args.controller,
        replan_rate=args.replan_rate,
        sim_dt=args.sim_dt,
        ego_replay_frames=args.ego_replay_frames,
        eval_frames=args.eval_frames,
        score_start_frame=args.score_start_frame,
    )
    evaluator.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
