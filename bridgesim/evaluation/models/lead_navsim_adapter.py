"""
Model adapter for LEAD NavSim (LTFv6) model.

This adapter uses the self-contained ltfv6.py model definition bundled with the checkpoint.
LEAD NavSim uses 4 cameras with 1920x270 resolution and 4 discrete commands.
"""

import sys
import types
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any

from bridgesim.evaluation.models.base_adapter import BaseModelAdapter
from bridgesim.evaluation.utils.constants import NAVSIM_CMD_MAPPING, DEFAULT_CMD
from bridgesim.utils.camera_utils import NAVSIM_CAM_CONFIGS


class LEADNavsimAdapter(BaseModelAdapter):
    """
    Adapter for LEAD NavSim (LTFv6) model.

    Uses 4 cameras (front, front-left, front-right, back) with 1920x270 resolution.
    This is the NavSim-trained version of LEAD with Latent TransFuser backbone.
    """

    def __init__(self, checkpoint_path: str, **kwargs):
        """
        Initialize LEAD NavSim adapter.

        Args:
            checkpoint_path: Path to checkpoint (.pth file)
        """
        super().__init__(checkpoint_path, config_path=None, **kwargs)
        self.config = None
        # Expected image dimensions from config
        self.image_width = 1920
        self.image_height = 270

    def _load_ltfv6_module(self, checkpoint_dir: Path):
        """
        Load the bundled ``ltfv6.py`` in a Python-3.9-safe way.

        The checkpoint file uses PEP 604 unions inside jaxtyping annotations, which
        are evaluated eagerly on Python 3.9. Prepending ``from __future__ import
        annotations`` keeps those annotations lazy without modifying the checkpoint
        bundle itself.
        """
        module_path = checkpoint_dir / "ltfv6.py"
        module_name = f"_bridgesim_ltfv6_{abs(hash(str(module_path)))}"

        if module_name in sys.modules:
            return sys.modules[module_name]

        source = module_path.read_text(encoding="utf-8")
        if "from __future__ import annotations" not in source:
            source = "from __future__ import annotations\n" + source

        # Disable checkpoint-bundled runtime type enforcement during import.
        # These decorators are helpful for training/dev, but on Python 3.9 they
        # eagerly evaluate jaxtyping unions that this evaluator does not need.
        source = source.replace(
            "import jaxtyping as jt",
            "import jaxtyping as jt\n"
            "def _bridgesim_noop_jaxtyped(obj=None, *args, **kwargs):\n"
            "    if callable(obj):\n"
            "        return obj\n"
            "    def decorator(fn):\n"
            "        return fn\n"
            "    return decorator\n"
            "jt.jaxtyped = _bridgesim_noop_jaxtyped",
        )
        source = source.replace(
            "from beartype import beartype",
            "def beartype(obj=None, *args, **kwargs):\n"
            "    if callable(obj):\n"
            "        return obj\n"
            "    def decorator(fn):\n"
            "        return fn\n"
            "    return decorator",
        )
        source = source.replace(
            'self.image_encoder = timm.create_model(config.image_architecture, pretrained=True, features_only=True)',
            'self.image_encoder = timm.create_model(config.image_architecture, pretrained=False, features_only=True)',
        )
        source = source.replace('raise Exception(f"Unknown GPU name: {name}")', 'return ""')

        module = types.ModuleType(module_name)
        module.__file__ = str(module_path)
        module.__package__ = ""

        sys.modules[module_name] = module
        try:
            exec(compile(source, str(module_path), "exec"), module.__dict__)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        return module

    def load_model(self):
        """Load LEAD NavSim model from checkpoint."""
        print("Loading LEAD NavSim (LTFv6) model...")

        # Get the directory containing the checkpoint (should have ltfv6.py and config.json)
        checkpoint_dir = Path(self.checkpoint_path).parent

        # Load the bundled ltfv6.py while preserving Python 3.9 compatibility.
        ltfv6_module = self._load_ltfv6_module(checkpoint_dir)
        load_tf = ltfv6_module.load_tf

        # Load model using the bundled loader
        self.model = load_tf(self.checkpoint_path, torch.device(self.device))

        # Store config reference
        self.config = self.model.config

        # Force float32 mode (override auto-detected bfloat16 for A100/L40S)
        # This ensures consistent dtype between model weights and inputs
        type(self.config).torch_float_type = property(lambda self: torch.float32)
        self.model = self.model.float()
        print("Forced float32 mode for inference")

        # Update image dimensions from config if available
        if hasattr(self.config, 'final_image_width'):
            self.image_width = self.config.final_image_width
        if hasattr(self.config, 'final_image_height'):
            self.image_height = self.config.final_image_height

        print(f"Image dimensions: {self.image_width}x{self.image_height}")
        print("LEAD NavSim model loaded successfully.")

    def get_camera_configs(self) -> Dict[str, Dict[str, float]]:
        """LEAD NavSim uses 4 cameras."""
        return {k: NAVSIM_CAM_CONFIGS[k] for k in ('CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_B0')}

    def _preprocess_rgb(self, camera_images: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        Preprocess multi-camera RGB images for LEAD NavSim model (4 cameras).

        The model expects a stitched panoramic image of shape (3, H, W) where
        W = 1920 (4 cameras * 480 each) and H = 270.
        """
        # Camera order for NavSim 4-camera setup
        cam_order = ['CAM_L0', 'CAM_F0', 'CAM_R0', 'CAM_B0']

        processed_cams = []
        cam_width = self.image_width // 4  # 480 per camera

        for cam_name in cam_order:
            if cam_name not in camera_images:
                # If camera is missing, use zeros
                processed_cams.append(np.zeros((self.image_height, cam_width, 3), dtype=np.uint8))
                continue

            img = camera_images[cam_name]

            # Convert BGR to RGB if needed
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize to target dimensions per camera
            img = cv2.resize(img, (cam_width, self.image_height), interpolation=cv2.INTER_LINEAR)
            processed_cams.append(img)

        # Concatenate all 4 cameras horizontally: 4 * 480 = 1920 width
        rgb = np.concatenate(processed_cams, axis=1)

        # Convert to tensor: (H, W, 3) -> (3, H, W)
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()

        return rgb_tensor

    def _get_command_onehot(self, command: int) -> np.ndarray:
        """Convert driving command to one-hot encoding using NAVSIM_CMD_MAPPING."""
        return NAVSIM_CMD_MAPPING.get(command, DEFAULT_CMD).copy()

    def prepare_input(self,
                     images: Dict[str, np.ndarray],
                     ego_state: Dict[str, Any],
                     scenario_data: Dict[str, Any],
                     frame_id: int) -> Any:
        """Prepare input for LEAD NavSim model."""
        # 1. Preprocess RGB images
        rgb = self._preprocess_rgb(images).unsqueeze(0).to(self.device)

        # 2. Get speed (m/s)
        speed = np.linalg.norm(ego_state['velocity'][:2])

        # Apply kick-off speed when stationary to help model predict forward motion
        KICKOFF_SPEED = 2.0  # m/s
        if speed < 0.5:
            speed = KICKOFF_SPEED

        # 3. Get acceleration (m/s²)
        if 'acceleration' in ego_state:
            acceleration = np.linalg.norm(ego_state['acceleration'][:2])
        else:
            acceleration = 0.0

        # 4. Get command one-hot
        command = ego_state.get('command', 1)  # Default to straight
        cmd_onehot = self._get_command_onehot(command)

        # 5. Build input tensors
        input_data = {
            'rgb': rgb,
            'speed': torch.tensor([[speed]], dtype=torch.float32, device=self.device),
            'acceleration': torch.tensor([[acceleration]], dtype=torch.float32, device=self.device),
            'command': torch.from_numpy(cmd_onehot).unsqueeze(0).to(self.device),
        }

        return input_data

    def run_inference(self, model_input: Any) -> Any:
        """Run LEAD NavSim model inference."""
        with torch.no_grad():
            prediction = self.model(model_input)
        return prediction

    def parse_output(self, model_output: Any, ego_state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Parse LEAD NavSim output."""
        # Extract waypoints from prediction
        if model_output.pred_future_waypoints is not None:
            # Convert to float32 for numpy compatibility (model may output bfloat16)
            waypoints = model_output.pred_future_waypoints[0].float().cpu().numpy()  # (n_waypoints, 2)

            # Swap columns: model outputs [forward, lateral] but evaluator expects [lateral, forward]
            waypoints_swapped = np.column_stack([-waypoints[:, 1], waypoints[:, 0]])
        else:
            # Fallback: use empty trajectory
            waypoints_swapped = np.zeros((8, 2), dtype=np.float32)

        return {'trajectory': waypoints_swapped}
