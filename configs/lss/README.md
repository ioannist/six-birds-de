# LSS config schema v0

These configs define datasets, masks, splits, and stub theory parameters for LSS runs.

Use the smoke config:
`python experiments/_smoke_lss_config.py --config configs/lss/smoke_small.yaml`

Future LSS experiments will read these configs and resolve dataset paths via the registry.
This enables “swap config, rerun” without editing code.
