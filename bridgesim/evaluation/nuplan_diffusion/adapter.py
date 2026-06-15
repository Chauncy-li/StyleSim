from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from bridgesim.evaluation.models.base_adapter import BaseModelAdapter
from bridgesim.evaluation.nuplan_diffusion.bootstrap import ensure_nuplan_imports
from bridgesim.evaluation.nuplan_diffusion.checkpoint import load_checkpoint_safely
from bridgesim.evaluation.nuplan_diffusion.feature_builder import BridgeSimNuPlanFeatureBuilder


class NuPlanDiffusionBridgeAdapter(BaseModelAdapter):
    """
    Isolated BridgeSim adapter for the external nuPlan diffusion planner.
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        *,
        nuplan_root: Optional[str] = None,
        nuplan_devkit_root: Optional[str] = None,
        device: Optional[str] = None,
        agent_num: int = 32,
        predicted_neighbor_num: int = 10,
        static_objects_num: int = 5,
        lane_num: int = 70,
        lane_len: int = 20,
        route_num: int = 25,
        route_len: int = 20,
        future_len: int = 80,
        time_len: int = 21,
        bridge_ego_reference: str = "rear_axle",
        **kwargs: Any,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path, config_path=config_path, **kwargs)
        self.nuplan_root = nuplan_root
        self.nuplan_devkit_root = nuplan_devkit_root
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._repo_root = None
        self._config = None
        self._observation_normalizer = None
        self._env = None
        self._ego_history: List[Dict[str, Any]] = []

        self.feature_builder = BridgeSimNuPlanFeatureBuilder(
            agent_num=agent_num,
            predicted_neighbor_num=predicted_neighbor_num,
            static_objects_num=static_objects_num,
            lane_num=lane_num,
            lane_len=lane_len,
            route_num=route_num,
            route_len=route_len,
            future_steps=future_len,
            past_steps=time_len,
            bridge_ego_reference=bridge_ego_reference,
        )

    def perceive(self, env, frame_id: int):
        self._env = env
        # This planner is vector-only, but returning None lets the base evaluator
        # capture front-camera frames for visualization output.
        return None

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        # Use the cam_f0 naming expected by BaseEvaluator.render_cam_f0_vis().
        return {
            "CAM_F0": {
                "x": 0.80,
                "y": 0.0,
                "z": 1.60,
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "fov": 70,
                "width": 1600,
                "height": 900,
            }
        }

    def load_model(self):
        repo_root, _ = ensure_nuplan_imports(self.nuplan_root, self.nuplan_devkit_root)
        self._repo_root = repo_root

        from nuplan_baseline.model.diff_planner.diffusion_planner import Diffusion_Planner
        from nuplan_baseline.utils.config import Config

        args_path = self._resolve_args_path()
        self._config = Config(args_path, guidance_fn=None, device=self.device)
        self._observation_normalizer = self._config.observation_normalizer

        self.model = Diffusion_Planner(self._config)
        self.model = load_checkpoint_safely(self.model, self.checkpoint_path, self.device)
        self.model.eval()
        self.model.to(self.device)
        return self.model

    def _resolve_args_path(self) -> str:
        if self.config_path is not None:
            return self.config_path

        checkpoint_path = Path(self.checkpoint_path)
        candidates = []
        if checkpoint_path.is_dir():
            candidates.append(checkpoint_path / "args.json")
        else:
            candidates.append(checkpoint_path.parent / "args.json")
            candidates.append(checkpoint_path.parent.parent / "args.json")

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        raise ValueError(
            "Could not infer nuPlan args.json. Pass --nuplan-args explicitly or place args.json next to the checkpoint."
        )

    def _to_tensors(self, features: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        tensors: Dict[str, torch.Tensor] = {}
        for key, value in features.items():
            array = np.asarray(value)
            if array.dtype == np.bool_:
                tensor = torch.tensor(array, dtype=torch.bool, device=self.device).unsqueeze(0)
            else:
                tensor = torch.tensor(array, dtype=torch.float32, device=self.device).unsqueeze(0)
            tensors[key] = tensor
        return tensors

    def prepare_input(
        self,
        images: Dict[str, np.ndarray],
        ego_state: Dict[str, Any],
        scenario_data: Dict[str, Any],
        frame_id: int,
    ) -> Any:
        if frame_id == 0 or (self._ego_history and frame_id <= int(self._ego_history[-1]["frame_id"])):
            self._ego_history = []

        self._ego_history.append(
            {
                "frame_id": int(frame_id),
                "position": np.asarray(ego_state["position"][:2], dtype=np.float64).copy(),
                "heading": float(ego_state["heading"]),
                "sim_time": float(frame_id) * 0.1,
            }
        )
        if len(self._ego_history) > max(self.feature_builder.past_steps * 3, self.feature_builder.past_steps + 8):
            self._ego_history = self._ego_history[-max(self.feature_builder.past_steps * 3, self.feature_builder.past_steps + 8):]

        features = self.feature_builder.build_inference_features(
            scenario_data=scenario_data,
            ego_state=ego_state,
            frame_id=frame_id,
            env=self._env,
            ego_history=self._ego_history,
        )
        model_input = self._to_tensors(features)
        if self._observation_normalizer is not None:
            model_input = self._observation_normalizer(model_input)
        return model_input

    def run_inference(self, model_input: Any) -> Any:
        with torch.no_grad():
            _, outputs = self.model(model_input)
        return outputs

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        prediction = model_output["prediction"][0, 0].detach().cpu().numpy().astype(np.float32)
        heading = np.arctan2(prediction[:, 3], prediction[:, 2]).astype(np.float32)
        bridge_xy = self.feature_builder.rear_axle_predictions_to_bridge_frame(prediction[:, :2], heading)

        # nuPlan local frame is [forward, left]; BridgeSim expects [left, forward].
        trajectory = np.stack([bridge_xy[:, 1], bridge_xy[:, 0]], axis=1).astype(np.float32)
        return {
            "trajectory": trajectory,
            "heading": heading,
            "trajectory_nuplan_rear_xy": prediction[:, :2].astype(np.float32),
            "trajectory_bridge_xy": bridge_xy.astype(np.float32),
        }

    def get_waypoint_dt(self) -> float:
        return 0.1

    def get_trajectory_time_horizon(self) -> float:
        return float(self.feature_builder.future_steps) * 0.1
