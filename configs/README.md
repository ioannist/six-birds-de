# 3×2pt config notes

`configs/3x2pt_y6_3x2pt_smoke.yaml` is a placeholder.

When Y6 3×2pt URLs become public:
1. Add registry entry with:
   `python scripts/add_registry_entry_from_urls.py --dataset <y6_key> --version <v> --url <url1> --url <url2> ...`
2. Update `configs/3x2pt_y6_3x2pt_smoke.yaml` to point at that dataset key and filename.
3. Run:
   `make exp-des-y6-3x2pt-smoke`
