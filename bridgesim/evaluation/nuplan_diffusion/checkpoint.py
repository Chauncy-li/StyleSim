from __future__ import annotations

from typing import Any

import torch


def load_checkpoint_safely(model: Any, ckpt_path: str, device: str) -> Any:
    """
    Mirror the external nuPlan helper while keeping the logic local to BridgeSim.
    """
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)

    if isinstance(state_dict, dict) and state_dict.get("ema_state_dict") is not None:
        state_dict = state_dict["ema_state_dict"]
    elif isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]

    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Unsupported checkpoint format at {ckpt_path}")

    ckpt_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

    model_keys = list(model.state_dict().keys())
    new_state_dict = {}
    missing_keys = []

    for key in model_keys:
        if key in ckpt_state_dict:
            new_state_dict[key] = ckpt_state_dict[key]
        elif key.replace(".module.", ".") in ckpt_state_dict:
            new_state_dict[key] = ckpt_state_dict[key.replace(".module.", ".")]
        else:
            missing_keys.append(key)

    lane_missing = [key for key in missing_keys if "lane_encoder" in key]
    if lane_missing:
        raise RuntimeError(
            "Lane encoder weights are missing from the checkpoint. "
            f"First missing keys: {lane_missing[:5]}"
        )

    model.load_state_dict(new_state_dict, strict=False)
    return model
