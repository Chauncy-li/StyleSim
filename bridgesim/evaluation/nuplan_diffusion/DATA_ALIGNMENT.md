# BridgeSim x nuPlan Diffusion Data Alignment

This note documents how the isolated BridgeSim-to-nuPlan diffusion migration
maps raw BridgeSim scenario data into the schema expected by the external
`nuplan_baseline` diffusion planner.

It is meant to answer three questions:

1. Which raw BridgeSim fields are used?
2. Which nuPlan diffusion fields are reconstructed?
3. Which parts are exact format matches versus semantic approximations?

## Scope

This document describes the isolated migration path implemented in:

- `bridgesim/evaluation/nuplan_diffusion/feature_builder.py`
- `bridgesim/evaluation/nuplan_diffusion/adapter.py`
- `bridgesim/evaluation/nuplan_diffusion_export_dataset.py`
- `bridgesim/evaluation/nuplan_diffusion_evaluator.py`

It does not modify the default BridgeSim unified evaluator flow.

## Ground Truth Reference

The external planner format is defined by:

- `nuplan_baseline/common/dataset.py`
- `nuplan_baseline/data_process/data_processor.py`
- `nuplan_baseline/data_process/map_process.py`
- `nuplan_baseline/model/diff_planner/layer/{encoder,decoder}.py`

The checkpoint currently validated in BridgeSim expects:

- `agent_num = 32`
- `predicted_neighbor_num = 10`
- `static_objects_num = 5`
- `static_objects_state_dim = 10`
- `lane_num = 70`
- `lane_len = 20`
- `route_num = 25`
- `route_len = 20`
- `future_len = 80`
- `time_len = 21`

## Raw BridgeData Source

A BridgeSim scenario pickle provides the raw ingredients used by the migration:

- `metadata`
  - `sdc_id`
  - `timestep`
  - `map`
  - `scenario_id`
- `tracks`
  - per-track `position`, `heading`, `velocity`, `width`, `length`, `height`, `valid`
- `map_features`
  - lane geometry via `polyline`
  - lane width via `width`
  - feature type via `type`
- `dynamic_map_states`
  - per-lane traffic light state sequence in `state.object_state`

## End-to-End Correspondence

The migration is built around one shared reconstruction layer:

- online closed-loop evaluation:
  - `BridgeData scenario + live MetaDrive env -> feature_builder -> adapter -> external diffusion planner`
- offline training export:
  - `BridgeData scenario -> feature_builder -> .npz export -> external ClosedLoopPlannerData`

This means the online evaluator and offline exporter share the same field logic,
shapes, and local-frame conventions.

## Field Mapping

| nuPlan diffusion field | Shape | BridgeSim source | Current status | Notes |
| --- | --- | --- | --- | --- |
| `ego_current_state` | `[10]` | ego history reconstructed from `tracks[sdc_id].state` | Format-aligned | Built from local `x, y, cos, sin, vx, vy, ax, ay, steering, yaw_rate` |
| `ego_agent_past` | `[21, 7]` | ego `position`, `heading`, `velocity` | Format-aligned | Local frame history: `x, y, yaw, vx, vy, ax, ay` |
| `ego_agent_future` | `[80, 3]` | ego future `position`, `heading` | Format-aligned | Local frame future: `x, y, yaw` |
| `neighbor_agents_past` | `[32, 21, 11]` | non-ego `tracks` | Format-aligned | `x, y, cos, sin, vx, vy, width, length, type(3)` |
| `neighbor_agents_past_mask` | `[32, 21]` | non-ego `valid` plus backfill logic | Format-aligned | Boolean per-step validity mask |
| `neighbor_agents_future` | `[10, 80, 3]` | selected future non-ego `tracks` | Format-aligned | Local frame future: `x, y, yaw` |
| `neighbor_agents_future_mask` | `[10, 80]` | selected future non-ego `valid` | Format-aligned | `True` means invalid, matching external loss usage |
| `lanes` | `[70, 20, 12]` | `map_features` + `dynamic_map_states` | Format-aligned | `x, y, dx, dy, left_dx, left_dy, right_dx, right_dy, tl_onehot(4)` |
| `lanes_mask` | `[70, 20]` | lane availability after resampling | Format-aligned | Point-wise mask; this is what the external code actually uses |
| `lanes_speed_limit` | `[70, 1]` | lane feature speed-limit keys if present | Format-aligned | Falls back to zero when absent |
| `lanes_has_speed_limit` | `[70, 1]` | derived from lane feature keys | Format-aligned | Boolean |
| `route_lanes` | `[25, 20, 12]` | MetaDrive live navigation plus BridgeData lane-graph grouping | Approximate semantics | Shape and encoding match; route source is still not native nuPlan roadblock routing |
| `route_lanes_mask` | `[25, 20]` | route vector validity | Format-aligned | Point-wise mask |
| `route_lanes_speed_limit` | `[25, 1]` | copied from selected route vectors | Approximate semantics | Exact when route comes from lane records; fallback route may not match nuPlan route pruning |
| `route_lanes_has_speed_limit` | `[25, 1]` | copied from selected route vectors | Approximate semantics | Same caveat as above |
| `static_objects` | `[5, 10]` | current-frame static obstacle tracks in `tracks` | Format-aligned, partially semantic-aligned | Reconstructed from `TRAFFIC_CONE` / `TRAFFIC_BARRIER`-style tracks as `x, y, cos, sin, w, l, type(4)` |
| `code_lat` | `[]` | currently not reconstructed | Placeholder | Filled with `-1` |
| `code_lon` | `[]` | currently not reconstructed | Placeholder | Filled with `-1` |

## Coordinate Convention

### Local axes

The most important convention is the local-frame axis order.

Inside the migrated nuPlan-format features:

- local `x` = forward
- local `y` = left

This matches the external diffusion planner expectation.

Inside BridgeSim controllers:

- trajectory column `0` = left
- trajectory column `1` = forward

So the adapter explicitly converts:

- nuPlan planner output `[forward, left]`
- to BridgeSim controller input `[left, forward]`

### Ego reference point

The external nuPlan diffusion pipeline is anchored on the ego rear axle:

- `data_processor.py` uses `ego_state.rear_axle`
- `ego_process.py` exports rear-axle `x, y, heading, vx, vy, ax, ay`
- `simulation/planner.py` converts predicted local poses back to global states
  through nuPlan's rear-axle trajectory utilities

The isolated BridgeSim migration now mirrors that behavior explicitly:

- offline export:
  - defaults to `--bridge-ego-reference rear_axle`
  - converts BridgeData ego poses to rear axle only if the caller overrides the
    reference as `center` or `cog`
- online inference:
  - uses the same `bridge_ego_reference` rule for the current ego anchor
  - reconstructs ego history from the actually executed BridgeSim ego states
    rather than replaying GT ego history from the scenario pickle
- output execution:
  - planner predictions are interpreted as rear-axle-relative nuPlan poses
  - if the BridgeSim execution reference is overridden to `center` or `cog`,
    predictions are converted back before handing them to the controller

Current default:

- `bridge_ego_reference = rear_axle`

Why this is the default:

- it is the native nuPlan planner convention
- a local 10-frame closed-loop comparison on the same scenario showed
  `rear_axle` outperforming the `center` and `cog` alternatives in this BridgeSim
  conversion chain

## Important Exact Matches

The following parts are already aligned tightly enough for both training export
and BridgeSim closed-loop inference:

- field names
- tensor shapes
- boolean mask polarity
- time horizon
- time step
- lane feature dimensionality
- agent feature dimensionality
- local-frame axis convention
- rear-axle ego anchoring by default
- live closed-loop ego-history reconstruction during inference
- checkpoint configuration sizes from `args.json`

## Current Semantic Approximations

Three areas still deserve separate attention when comparing against native
nuPlan preprocessing.

### 1. Static objects

Current behavior:

- current-frame static obstacle tracks are reconstructed into
  `[x, y, cos, sin, width, length, type(4)]`
- nearest `5` static objects are selected by distance to ego
- zero or missing extents fall back to per-type size priors computed from the
  scenario and then to conservative type defaults

Why:

- BridgeData preserves explicit static obstacle track types such as
  `TRAFFIC_CONE` and `TRAFFIC_BARRIER`
- it does not always preserve every native nuPlan static category or exact box
  extents at every frame, so some dimensions still require fallback priors

Practical impact:

- the encoder now receives a populated static-object branch
- remaining mismatch is mainly category coverage and occasional fallback box
  sizes, not tensor format

### 2. Route lanes

Current behavior:

- lane features are first grouped into roadblock-like components using
  `left_neighbor` / `right_neighbor`
- group transitions are derived from `exit_lanes`
- online evaluation:
  - uses MetaDrive live navigation as the route seed
  - maps those lanes back onto BridgeData lane groups
- offline export:
  - uses future ego-path probes to infer route groups
  - then applies connected-prefix pruning and graph extension

Why:

- BridgeSim evaluation scenarios do not expose native nuPlan
  `route_roadblock_ids` directly in the same form as the nuPlan devkit

Practical impact:

- geometry format is correct
- route semantics are approximate, especially offline

### 3. Dynamics reconstruction

Current behavior:

- ego and agent kinematics are reconstructed from BridgeData track states
- offline export reconstructs ego velocity and acceleration from the
  rear-axle-shifted ego positions
- online inference reconstructs ego velocity and acceleration from the actually
  executed ego-history queue
- `ego_current_state` steering and yaw-rate follow the external nuPlan logic:
  last-two-frame wrapped heading delta, Pacifica wheelbase, low-speed zeroing,
  and the same clipping ranges

Why:

- BridgeData stores serialized track states rather than native nuPlan runtime
  `EgoState` objects with direct rear-axle acceleration
- BridgeSim inference also exposes only current ego state directly, so we
  reconstruct the executed rear-axle history inside the isolated adapter

Practical impact:

- feature format now matches closely enough for both export and closed-loop
  evaluation
- the remaining difference is source-level: acceleration is reconstructed from
  stored velocity instead of read directly from nuPlan runtime objects

## What Can Be Improved In Code

Yes. The remaining gaps can still be improved without touching the default
BridgeSim evaluator path, as long as changes stay isolated inside the
`nuplan_diffusion` directory.

### High-value next improvement: route lanes

Recommended direction:

- continue improving the current BridgeData lane-graph route reconstruction:
  - stronger group-level route scoring
  - better bridging across ambiguous connectors
  - optional route metadata preservation during conversion if available
- keep live MetaDrive navigation as the online preference
- further reduce the remaining gap between online and offline route selection

Expected benefit:

- best gain for planner behavior fidelity
- likely the most important semantic upgrade for training/export consistency

### Medium-value improvement: static objects

Recommended direction:

- expand category coverage beyond the explicit obstacle tracks already used
- preserve more exact box extents during conversion if the raw dataset exposes
  them
- optionally retain additional static categories if future BridgeData
  conversions keep them

Expected benefit:

- improves encoder completeness further
- lower risk than route reconstruction

### Medium-value improvement: dynamics details

Recommended direction:

- preserve native acceleration directly during conversion if future BridgeData
  exports expose it
- optionally derive steering using smoother temporal windows instead of only
  two frames
- optionally preserve raw heading when motion is nearly static instead of
  relying on position-derived future heading smoothing

Expected benefit:

- smaller than route-lane improvement
- useful for reducing subtle train/eval mismatch

## Recommended Priority

If improving the migration incrementally, the best order is:

1. route-lane semantics
2. static-object category/extent refinement
3. kinematic-detail refinement
4. codebook label reconstruction if full training parity is later needed

## Bottom Line

Current status:

- format parity: good
- model-input parity: good
- semantic parity: partial

The migrated pipeline is already valid for BridgeSim closed-loop evaluation and
nuPlan-style `.npz` export, but it is still an approximation of the native
nuPlan preprocessing stack in the three areas listed above.
