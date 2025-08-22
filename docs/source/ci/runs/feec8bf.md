# CI run feec8bf

- Run: [link](https://github.com/KULeuven-MICAS/stream_aie/actions/runs/17158188258)
- Time: Denver `2025-08-22 08:38:00 MDT` • Brussels `2025-08-22 16:38:00 CEST`

---

## GEMM Deployment Summary

- Workflow: `AIE Deployment Gemm`
- Commit: `feec8bf749c2f20cb24cec474758b901ef3dc823`
- Runner: `venus`
- Run time: Denver `2025-08-22 08:37:41 MDT` • Brussels `2025-08-22 16:37:41 CEST`
- Run: [#15](https://github.com/KULeuven-MICAS/stream_aie/actions/runs/17158188258); Attempt 1

| HW | M | K | N | Status | Note |
|---|---|---|---|--------|------|
| single_core | 64 | 128 | 64 | ❌ failed | missing status.ok |
| single_core | 64 | 64 | 64 | ✅ success |  |

**Totals:** ✅ `1`  •  ❌ `1`  •  All: `2`

### Details for Successful Runs

<details><summary><strong>[single_core] M=64 K=64 N=64</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 8 | 10,249 | 1,071.00 | 30.60 | 47.81 | 25.58 | 39.96 |

</details>
