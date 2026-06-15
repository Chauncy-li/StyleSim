# BridgeSim x nuPlan Diffusion

This directory contains an isolated migration path for the external
`nuplan_baseline` diffusion planner.

It does **not** modify the current BridgeSim model registry, unified evaluator,
or multi-model batch scripts.

## Documentation

- `DATA_ALIGNMENT.md`
  - Detailed field-by-field correspondence between raw BridgeSim scenario data
    and the external nuPlan diffusion planner schema.

## Added entrypoints

- `bridgesim/evaluation/nuplan_diffusion_evaluator.py`
  - Run one external nuPlan diffusion checkpoint inside BridgeSim.
- `bridgesim/evaluation/nuplan_diffusion_export_dataset.py`
  - Export BridgeSim scenarios into the `.npz` schema expected by the external
    nuPlan diffusion training code.

## What is aligned

- BridgeSim `tracks` -> nuPlan-style `neighbor_agents_past` / `neighbor_agents_future`
- BridgeSim ego track -> `ego_agent_past` / `ego_agent_future` / `ego_current_state`
- BridgeSim static obstacle tracks -> `static_objects`
- BridgeSim `map_features` -> `lanes`
- MetaDrive live navigation plus BridgeData lane-graph reconstruction -> `route_lanes`
- BridgeSim traffic lights -> lane traffic-light one-hot features

## Ego reference default

The isolated migration path now defaults to:

- `bridge_ego_reference = rear_axle`

This matches the native nuPlan diffusion preprocessing chain. The evaluator,
exporter, and quick-batch wrapper also expose:

- `--bridge-ego-reference {rear_axle,center,cog}`

so the assumption can still be overridden for debugging.

## Important current limitations

- `static_objects` is now reconstructed from explicit static obstacle tracks
  such as cones and barriers, but category coverage still depends on what the
  converted BridgeData scenario preserves.
- `route_lanes` now uses grouped lane connectivity (`left/right neighbors` and
  `exit_lanes`) to better approximate nuPlan roadblock routing, but it is still
  not native nuPlan `route_roadblock_ids`.
- Ego acceleration is estimated from timestamp-aware velocity differences
  because BridgeData stores serialized tracks rather than native nuPlan runtime
  acceleration states.
- Online evaluation still prefers MetaDrive live navigation as the seed for
  `route_lanes`; if it is unavailable, the reconstruction falls back to the
  local BridgeData lane graph.

## Minimal export smoke test

```bash
conda run -n bridgesim python bridgesim/evaluation/nuplan_diffusion_export_dataset.py \
  --scenario-root /media/lsw/Data/Ubuntu_copy/BridgeData \
  --output-dir /tmp/nuplan_diffusion_export_smoke \
  --scenario-limit 1 \
  --bridge-ego-reference rear_axle
```

Outputs:

- `npz/*.npz`
- `train.json`
- `val.json`
- `all.json`
- `normalization_bridgesim.json`

## Minimal evaluator template

```bash
conda run -n bridgesim python bridgesim/evaluation/nuplan_diffusion_evaluator.py \
  --checkpoint /path/to/checkpoint.pth \
  --nuplan-args /path/to/args.json \
  --nuplan-root /home/lsw/meta_programs/Nuplan-Baseline-3090 \
  --scenario-path /media/lsw/Data/Ubuntu_copy/BridgeData/sd_xxx \
  --output-dir /tmp/nuplan_diffusion_eval \
  --traffic-mode log_replay \
  --controller pure_pursuit \
  --replan-rate 1 \
  --sim-dt 0.1 \
  --eval-mode closed_loop \
  --bridge-ego-reference rear_axle
```
