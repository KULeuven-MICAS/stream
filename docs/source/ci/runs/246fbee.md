# CI run 246fbee

- Run: [link](https://github.com/KULeuven-MICAS/stream_aie/actions/runs/17266392568)
- Time: Denver `2025-08-27 06:17:45 MDT` • Brussels `2025-08-27 14:17:45 CEST`

---

## GEMM Deployment Summary

- Workflow: `AIE Deployment Gemm`
- Commit: `246fbeeb8d68cd5c1cd80b37f9016cdb193d5a5e`
- Runner: `venus`
- Run time: Denver `2025-08-27 06:17:24 MDT` • Brussels `2025-08-27 14:17:24 CEST`
- Run: [#60](https://github.com/KULeuven-MICAS/stream_aie/actions/runs/17266392568); Attempt 1

| HW | M | K | N | Rows | Cols | Status | Note |
|---|---|---|---|------|------|--------|------|
| single_col | 64 | 64 | 64 | 2 | 1 | 🐬 success |  |
| single_core | 64 | 64 | 64 | 1 | 1 | 🐬 success |  |
| whole_array | 64 | 64 | 128 | 2 | 4 | 🐬 success |  |

**Totals:** 🐬 `3`  •  ❌ `0`  •  All: `3`

<details><summary><strong>[single_col] M=64 K=64 N=64 R=2 C=1</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile3,1 | 4 | 5,781 | 1,134.00 | 28.90 | 45.15 | 22.67 | 35.43 |
| tile2,1 | 4 | 5,781 | 1,134.00 | 28.90 | 45.15 | 22.67 | 35.43 |

</details>

<details><summary><strong>[single_core] M=64 K=64 N=64 R=1 C=1</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 8 | 12,808 | 1,134.00 | 28.90 | 45.15 | 20.47 | 31.98 |

</details>

<details><summary><strong>[whole_array] M=64 K=64 N=128 R=2 C=4</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile3,1 | 2 | 2,317 | 1,150.00 | 28.49 | 44.52 | 28.28 | 44.20 |
| tile3,4 | 2 | 2,317 | 1,150.00 | 28.49 | 44.52 | 28.28 | 44.20 |
| tile3,3 | 2 | 2,317 | 1,150.00 | 28.49 | 44.52 | 28.28 | 44.20 |
| tile3,2 | 2 | 2,317 | 1,150.00 | 28.49 | 44.52 | 28.28 | 44.20 |
| tile2,2 | 2 | 2,317 | 1,150.00 | 28.49 | 44.52 | 28.28 | 44.20 |
| tile2,1 | 2 | 2,317 | 1,150.00 | 28.49 | 44.52 | 28.28 | 44.20 |
| tile2,4 | 2 | 2,317 | 1,150.00 | 28.49 | 44.52 | 28.28 | 44.20 |
| tile2,3 | 2 | 2,317 | 1,150.00 | 28.49 | 44.52 | 28.28 | 44.20 |

</details>
