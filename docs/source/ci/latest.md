## GEMM Deployment Summary

- Workflow: `AIE Deployment Gemm`
- Commit: `9d53d7e633eae1ffce9a936a4259c0b4d736d6ec`
- Runner: `venus`
- Run time: Denver `2025-08-22 08:15:54 MDT` • Brussels `2025-08-22 16:15:54 CEST`
- Run: [#14](https://github.com/KULeuven-MICAS/stream_aie/actions/runs/17157677847); Attempt 1

| HW | M | K | N | Status | Note |
|---|---|---|---|--------|------|
| single_core | 64 | 128 | 64 | ❌ failed | missing status.ok |
| single_core | 64 | 64 | 64 | ✅ success |  |

**Totals:** ✅ `1`  •  ❌ `1`  •  All: `2`

### Details for Successful Runs

<details><summary><strong>[single_core] M=64 K=64 N=64</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 8 | 10,248 | 1,071.00 | 30.60 | 47.81 | 25.58 | 39.97 |

</details>
