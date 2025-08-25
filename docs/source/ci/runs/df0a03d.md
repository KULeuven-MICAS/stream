# CI run df0a03d

- Run: [link](https://github.com/KULeuven-MICAS/stream_aie/actions/runs/17211455753)
- Time: Denver `2025-08-25 08:16:42 MDT` • Brussels `2025-08-25 16:16:42 CEST`

---

## GEMM Deployment Summary

- Workflow: `AIE Deployment Gemm`
- Commit: `df0a03dc682434a37402ddc45e46e31df7014a1b`
- Runner: `venus`
- Run time: Denver `2025-08-25 08:16:15 MDT` • Brussels `2025-08-25 16:16:15 CEST`
- Run: [#31](https://github.com/KULeuven-MICAS/stream_aie/actions/runs/17211455753); Attempt 1

| HW | M | K | N | Status | Note |
|---|---|---|---|--------|------|
| single_core | 64 | 128 | 64 | ✅ success |  |
| single_core | 64 | 64 | 64 | ✅ success |  |

**Totals:** ✅ `2`  •  ❌ `0`  •  All: `2`

### Details for Successful Runs

<details><summary><strong>[single_core] M=64 K=128 N=64</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 16 | 26,240 | 1,038.00 | 31.57 | 49.33 | 19.98 | 31.22 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_core] M=64 K=64 N=64</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 8 | 11,247 | 1,070.00 | 30.62 | 47.85 | 23.31 | 36.42 |

</details>
