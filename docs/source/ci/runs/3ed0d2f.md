# CI run 3ed0d2f

- Run: [link](https://github.com/KULeuven-MICAS/stream_aie/actions/runs/17232284253)
- Time: Denver `2025-08-26 02:40:27 MDT` • Brussels `2025-08-26 10:40:27 CEST`

---

## GEMM Deployment Summary

- Workflow: `AIE Deployment Gemm`
- Commit: `3ed0d2f8cef99eb36cd88893a0bd602208b9b827`
- Runner: `venus`
- Run time: Denver `2025-08-26 02:40:08 MDT` • Brussels `2025-08-26 10:40:08 CEST`
- Run: [#38](https://github.com/KULeuven-MICAS/stream_aie/actions/runs/17232284253); Attempt 1

| HW | M | K | N | Status | Note |
|---|---|---|---|--------|------|
| single_col | 128 | 128 | 128 | ✅ success |  |
| single_col | 128 | 128 | 256 | ❌ failed | missing status.ok |
| single_col | 128 | 128 | 64 | ✅ success |  |
| single_col | 128 | 256 | 128 | ❌ failed | missing status.ok |
| single_col | 128 | 256 | 256 | ❌ failed | missing status.ok |
| single_col | 128 | 256 | 64 | ❌ failed | missing status.ok |
| single_col | 128 | 64 | 128 | ✅ success |  |
| single_col | 128 | 64 | 256 | ❌ failed | missing status.ok |
| single_col | 128 | 64 | 64 | ✅ success |  |
| single_col | 256 | 128 | 128 | ✅ success |  |
| single_col | 256 | 128 | 256 | ❌ failed | missing status.ok |
| single_col | 256 | 128 | 64 | ✅ success |  |
| single_col | 256 | 256 | 128 | ❌ failed | missing status.ok |
| single_col | 256 | 256 | 256 | ❌ failed | missing status.ok |
| single_col | 256 | 256 | 64 | ❌ failed | missing status.ok |
| single_col | 256 | 64 | 128 | ✅ success |  |
| single_col | 256 | 64 | 256 | ❌ failed | missing status.ok |
| single_col | 256 | 64 | 64 | ✅ success |  |
| single_col | 64 | 128 | 128 | ✅ success |  |
| single_col | 64 | 128 | 256 | ❌ failed | missing status.ok |
| single_col | 64 | 128 | 64 | ✅ success |  |
| single_col | 64 | 256 | 128 | ❌ failed | missing status.ok |
| single_col | 64 | 256 | 256 | ❌ failed | missing status.ok |
| single_col | 64 | 256 | 64 | ❌ failed | missing status.ok |
| single_col | 64 | 64 | 128 | ✅ success |  |
| single_col | 64 | 64 | 256 | ❌ failed | missing status.ok |
| single_col | 64 | 64 | 64 | ✅ success |  |
| single_core | 128 | 128 | 128 | ❌ failed | missing status.ok |
| single_core | 128 | 128 | 256 | ✅ success |  |
| single_core | 128 | 128 | 64 | ✅ success |  |
| single_core | 128 | 256 | 128 | ✅ success |  |
| single_core | 128 | 256 | 256 | ✅ success |  |
| single_core | 128 | 256 | 64 | ✅ success |  |
| single_core | 128 | 64 | 128 | ✅ success |  |
| single_core | 128 | 64 | 256 | ✅ success |  |
| single_core | 128 | 64 | 64 | ✅ success |  |
| single_core | 256 | 128 | 128 | ❌ failed | missing status.ok |
| single_core | 256 | 128 | 256 | ✅ success |  |
| single_core | 256 | 128 | 64 | ✅ success |  |
| single_core | 256 | 256 | 128 | ✅ success |  |
| single_core | 256 | 256 | 256 | ✅ success |  |
| single_core | 256 | 256 | 64 | ✅ success |  |
| single_core | 256 | 64 | 128 | ✅ success |  |
| single_core | 256 | 64 | 256 | ✅ success |  |
| single_core | 256 | 64 | 64 | ✅ success |  |
| single_core | 64 | 128 | 128 | ✅ success |  |
| single_core | 64 | 128 | 256 | ✅ success |  |
| single_core | 64 | 128 | 64 | ✅ success |  |
| single_core | 64 | 256 | 128 | ✅ success |  |
| single_core | 64 | 256 | 256 | ✅ success |  |
| single_core | 64 | 256 | 64 | ✅ success |  |
| single_core | 64 | 64 | 128 | ✅ success |  |
| single_core | 64 | 64 | 256 | ✅ success |  |
| single_core | 64 | 64 | 64 | ✅ success |  |
| whole_array | 128 | 128 | 128 | ✅ success |  |
| whole_array | 128 | 128 | 256 | ❌ failed | missing status.ok |
| whole_array | 128 | 128 | 64 | ✅ success |  |
| whole_array | 128 | 256 | 128 | ✅ success |  |
| whole_array | 128 | 256 | 256 | ❌ failed | missing status.ok |
| whole_array | 128 | 256 | 64 | ✅ success |  |
| whole_array | 128 | 64 | 128 | ✅ success |  |
| whole_array | 128 | 64 | 256 | ❌ failed | missing status.ok |
| whole_array | 128 | 64 | 64 | ✅ success |  |
| whole_array | 256 | 128 | 128 | ✅ success |  |
| whole_array | 256 | 128 | 256 | ❌ failed | missing status.ok |
| whole_array | 256 | 128 | 64 | ✅ success |  |
| whole_array | 256 | 256 | 128 | ✅ success |  |
| whole_array | 256 | 256 | 256 | ❌ failed | missing status.ok |
| whole_array | 256 | 256 | 64 | ✅ success |  |
| whole_array | 256 | 64 | 128 | ✅ success |  |
| whole_array | 256 | 64 | 256 | ❌ failed | missing status.ok |
| whole_array | 256 | 64 | 64 | ✅ success |  |
| whole_array | 64 | 128 | 128 | ✅ success |  |
| whole_array | 64 | 128 | 256 | ✅ success |  |
| whole_array | 64 | 128 | 64 | ✅ success |  |
| whole_array | 64 | 256 | 128 | ✅ success |  |
| whole_array | 64 | 256 | 256 | ✅ success |  |
| whole_array | 64 | 256 | 64 | ✅ success |  |
| whole_array | 64 | 64 | 128 | ✅ success |  |
| whole_array | 64 | 64 | 256 | ✅ success |  |
| whole_array | 64 | 64 | 64 | ✅ success |  |

**Totals:** ✅ `58`  •  ❌ `23`  •  All: `81`

### Details for Successful Runs

<details><summary><strong>[single_col] M=128 K=128 N=128</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 32 | 61,029 | 1,150.00 | 28.49 | 44.52 | 17.18 | 26.85 |
| tile3,1 | 32 | 61,029 | 1,150.00 | 28.49 | 44.52 | 17.18 | 26.85 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_col] M=128 K=128 N=64</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 16 | 28,363 | 1,150.00 | 28.49 | 44.52 | 18.48 | 28.88 |
| tile3,1 | 16 | 28,364 | 1,150.00 | 28.49 | 44.52 | 18.48 | 28.88 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_col] M=128 K=64 N=128</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile3,1 | 16 | 31,187 | 1,150.00 | 28.49 | 44.52 | 16.81 | 26.27 |
| tile2,1 | 16 | 31,187 | 1,150.00 | 28.49 | 44.52 | 16.81 | 26.27 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_col] M=128 K=64 N=64</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 8 | 12,940 | 1,150.00 | 28.49 | 44.52 | 20.26 | 31.65 |
| tile3,1 | 8 | 12,940 | 1,150.00 | 28.49 | 44.52 | 20.26 | 31.65 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_col] M=256 K=128 N=128</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile3,1 | 64 | 120,208 | 1,150.00 | 28.49 | 44.52 | 17.45 | 27.26 |
| tile2,1 | 64 | 120,208 | 1,150.00 | 28.49 | 44.52 | 17.45 | 27.26 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_col] M=256 K=128 N=64</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 32 | 57,950 | 1,150.00 | 28.49 | 44.52 | 18.09 | 28.27 |
| tile3,1 | 32 | 57,951 | 1,150.00 | 28.49 | 44.52 | 18.09 | 28.27 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_col] M=256 K=64 N=128</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 32 | 59,503 | 1,150.00 | 28.49 | 44.52 | 17.62 | 27.53 |
| tile3,1 | 32 | 59,503 | 1,150.00 | 28.49 | 44.52 | 17.62 | 27.53 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_col] M=256 K=64 N=64</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 16 | 27,104 | 1,150.00 | 28.49 | 44.52 | 19.34 | 30.22 |
| tile3,1 | 16 | 27,105 | 1,150.00 | 28.49 | 44.52 | 19.34 | 30.22 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_col] M=64 K=128 N=128</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile3,1 | 16 | 31,450 | 1,150.00 | 28.49 | 44.52 | 16.67 | 26.05 |
| tile2,1 | 16 | 31,965 | 1,150.00 | 28.49 | 44.52 | 16.40 | 25.63 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_col] M=64 K=128 N=64</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile3,1 | 8 | 13,574 | 1,150.00 | 28.49 | 44.52 | 19.31 | 30.18 |
| tile2,1 | 8 | 14,089 | 1,150.00 | 28.49 | 44.52 | 18.61 | 29.07 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_col] M=64 K=64 N=128</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 8 | 13,293 | 1,134.00 | 28.90 | 45.15 | 19.72 | 30.81 |
| tile3,1 | 8 | 13,807 | 1,134.00 | 28.90 | 45.15 | 18.99 | 29.67 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_col] M=64 K=64 N=64</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 4 | 5,781 | 1,134.00 | 28.90 | 45.15 | 22.67 | 35.43 |
| tile3,1 | 4 | 5,845 | 1,150.00 | 28.49 | 44.52 | 22.42 | 35.04 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_core] M=128 K=128 N=256</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 128 | 242,699 | 1,150.00 | 28.49 | 44.52 | 17.28 | 27.00 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_core] M=128 K=128 N=64</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 32 | 58,982 | 1,150.00 | 28.49 | 44.52 | 17.78 | 27.78 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_core] M=128 K=256 N=128</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 128 | 232,308 | 1,150.00 | 28.49 | 44.52 | 18.05 | 28.21 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_core] M=128 K=256 N=256</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 256 | 468,019 | 1,150.00 | 28.49 | 44.52 | 17.92 | 28.01 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_core] M=128 K=256 N=64</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 64 | 114,420 | 1,150.00 | 28.49 | 44.52 | 18.33 | 28.64 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_core] M=128 K=64 N=128</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 32 | 55,463 | 1,150.00 | 28.49 | 44.52 | 18.91 | 29.54 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_core] M=128 K=64 N=256</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 64 | 112,177 | 1,150.00 | 28.49 | 44.52 | 18.70 | 29.21 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_core] M=128 K=64 N=64</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 16 | 27,115 | 1,150.00 | 28.49 | 44.52 | 19.34 | 30.21 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_core] M=256 K=128 N=256</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 256 | 479,207 | 1,148.99 | 28.52 | 44.56 | 17.51 | 27.35 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_core] M=256 K=128 N=64</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 64 | 118,158 | 1,150.00 | 28.49 | 44.52 | 17.75 | 27.73 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_core] M=256 K=256 N=128</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 256 | 459,151 | 1,150.00 | 28.49 | 44.52 | 18.27 | 28.55 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_core] M=256 K=256 N=256</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 511 | 924,987 | 1,151.18 | 28.46 | 44.48 | 18.14 | 28.34 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_core] M=256 K=256 N=64</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 128 | 227,853 | 1,150.00 | 28.49 | 44.52 | 18.41 | 28.76 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_core] M=256 K=64 N=128</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 64 | 112,121 | 1,150.00 | 28.49 | 44.52 | 18.70 | 29.23 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_core] M=256 K=64 N=256</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 128 | 225,899 | 1,346.84 | 24.33 | 38.01 | 18.57 | 29.01 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_core] M=256 K=64 N=64</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 32 | 55,443 | 1,150.00 | 28.49 | 44.52 | 18.91 | 29.55 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_core] M=64 K=128 N=128</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 32 | 61,060 | 1,150.00 | 28.49 | 44.52 | 17.17 | 26.83 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_core] M=64 K=128 N=256</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 64 | 124,379 | 1,150.00 | 28.49 | 44.52 | 16.86 | 26.35 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_core] M=64 K=128 N=64</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 16 | 29,402 | 1,150.00 | 28.49 | 44.52 | 17.83 | 27.86 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_core] M=64 K=256 N=128</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 64 | 118,856 | 1,150.00 | 28.49 | 44.52 | 17.64 | 27.57 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_core] M=64 K=256 N=256</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 128 | 241,159 | 1,150.00 | 28.49 | 44.52 | 17.39 | 27.18 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_core] M=64 K=256 N=64</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 32 | 57,706 | 1,150.00 | 28.49 | 44.52 | 18.17 | 28.39 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_core] M=64 K=64 N=128</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 16 | 26,796 | 1,134.00 | 28.90 | 45.15 | 19.57 | 30.57 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_core] M=64 K=64 N=256</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 32 | 54,842 | 1,134.00 | 28.90 | 45.15 | 19.12 | 29.87 |

</details>

### Details for Successful Runs

<details><summary><strong>[single_core] M=64 K=64 N=64</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 8 | 12,786 | 1,134.00 | 28.90 | 45.15 | 20.50 | 32.04 |

</details>

### Details for Successful Runs

<details><summary><strong>[whole_array] M=128 K=128 N=128</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile3,1 | 16 | 29,403 | 1,150.00 | 28.49 | 44.52 | 17.83 | 27.86 |
| tile2,1 | 16 | 29,410 | 1,150.00 | 28.49 | 44.52 | 17.83 | 27.85 |
| tile2,2 | 16 | 30,344 | 1,150.00 | 28.49 | 44.52 | 17.28 | 27.00 |
| tile3,2 | 16 | 30,345 | 1,150.00 | 28.49 | 44.52 | 17.28 | 27.00 |

</details>

### Details for Successful Runs

<details><summary><strong>[whole_array] M=128 K=128 N=64</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 8 | 13,407 | 1,128.88 | 29.03 | 45.35 | 19.55 | 30.55 |
| tile3,1 | 8 | 13,408 | 1,129.25 | 29.02 | 45.34 | 19.55 | 30.55 |
| tile3,2 | 8 | 13,960 | 1,150.00 | 28.49 | 44.52 | 18.78 | 29.34 |
| tile2,2 | 8 | 15,383 | 1,957.00 | 16.74 | 26.16 | 17.04 | 26.63 |

</details>

### Details for Successful Runs

<details><summary><strong>[whole_array] M=128 K=256 N=128</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,2 | 32 | 57,707 | 1,150.00 | 28.49 | 44.52 | 18.17 | 28.39 |
| tile2,1 | 32 | 57,708 | 1,150.00 | 28.49 | 44.52 | 18.17 | 28.39 |
| tile3,1 | 32 | 58,037 | 1,150.00 | 28.49 | 44.52 | 18.07 | 28.23 |
| tile3,2 | 32 | 58,037 | 1,150.00 | 28.49 | 44.52 | 18.07 | 28.23 |

</details>

### Details for Successful Runs

<details><summary><strong>[whole_array] M=128 K=256 N=64</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,2 | 16 | 27,130 | 1,150.00 | 28.49 | 44.52 | 19.33 | 30.20 |
| tile2,1 | 16 | 27,131 | 1,150.00 | 28.49 | 44.52 | 19.32 | 30.19 |
| tile3,1 | 16 | 27,879 | 1,150.00 | 28.49 | 44.52 | 18.81 | 29.38 |
| tile3,2 | 16 | 27,879 | 1,150.00 | 28.49 | 44.52 | 18.81 | 29.38 |

</details>

### Details for Successful Runs

<details><summary><strong>[whole_array] M=128 K=64 N=128</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,2 | 8 | 12,658 | 1,118.00 | 29.31 | 45.80 | 20.71 | 32.36 |
| tile2,1 | 8 | 12,721 | 1,118.00 | 29.31 | 45.80 | 20.61 | 32.20 |
| tile3,1 | 8 | 12,786 | 1,134.00 | 28.90 | 45.15 | 20.50 | 32.04 |
| tile3,2 | 8 | 13,491 | 1,134.00 | 28.90 | 45.15 | 19.43 | 30.36 |

</details>

### Details for Successful Runs

<details><summary><strong>[whole_array] M=128 K=64 N=64</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,2 | 4 | 5,717 | 1,118.00 | 29.31 | 45.80 | 22.93 | 35.82 |
| tile2,1 | 4 | 5,717 | 1,118.00 | 29.31 | 45.80 | 22.93 | 35.82 |
| tile3,1 | 4 | 5,781 | 1,134.00 | 28.90 | 45.15 | 22.67 | 35.43 |
| tile3,2 | 4 | 5,781 | 1,134.00 | 28.90 | 45.15 | 22.67 | 35.43 |

</details>

### Details for Successful Runs

<details><summary><strong>[whole_array] M=256 K=128 N=128</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,2 | 32 | 58,982 | 1,150.00 | 28.49 | 44.52 | 17.78 | 27.78 |
| tile2,1 | 32 | 58,983 | 1,150.00 | 28.49 | 44.52 | 17.78 | 27.78 |
| tile3,2 | 32 | 59,499 | 1,150.00 | 28.49 | 44.52 | 17.62 | 27.54 |
| tile3,1 | 32 | 59,499 | 1,150.00 | 28.49 | 44.52 | 17.62 | 27.54 |

</details>

### Details for Successful Runs

<details><summary><strong>[whole_array] M=256 K=128 N=64</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 16 | 28,363 | 1,150.00 | 28.49 | 44.52 | 18.48 | 28.88 |
| tile3,2 | 16 | 28,363 | 1,150.00 | 28.49 | 44.52 | 18.48 | 28.88 |
| tile3,1 | 16 | 28,363 | 1,150.00 | 28.49 | 44.52 | 18.48 | 28.88 |
| tile2,2 | 16 | 28,364 | 1,150.00 | 28.49 | 44.52 | 18.48 | 28.88 |

</details>

### Details for Successful Runs

<details><summary><strong>[whole_array] M=256 K=256 N=128</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,2 | 64 | 114,422 | 1,150.00 | 28.49 | 44.52 | 18.33 | 28.64 |
| tile2,1 | 64 | 114,422 | 1,150.00 | 28.49 | 44.52 | 18.33 | 28.64 |
| tile3,1 | 64 | 115,517 | 1,150.00 | 28.49 | 44.52 | 18.15 | 28.37 |
| tile3,2 | 64 | 115,518 | 1,150.00 | 28.49 | 44.52 | 18.15 | 28.37 |

</details>

### Details for Successful Runs

<details><summary><strong>[whole_array] M=256 K=256 N=64</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 32 | 55,488 | 1,150.00 | 28.49 | 44.52 | 18.90 | 29.53 |
| tile2,2 | 32 | 55,488 | 1,150.00 | 28.49 | 44.52 | 18.90 | 29.53 |
| tile3,1 | 32 | 55,725 | 1,150.00 | 28.49 | 44.52 | 18.82 | 29.40 |
| tile3,2 | 32 | 55,725 | 1,150.00 | 28.49 | 44.52 | 18.82 | 29.40 |

</details>

### Details for Successful Runs

<details><summary><strong>[whole_array] M=256 K=64 N=128</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,2 | 16 | 34,695 | 1,150.00 | 28.49 | 44.52 | 15.11 | 23.61 |
| tile2,1 | 16 | 34,696 | 1,150.00 | 28.49 | 44.52 | 15.11 | 23.61 |
| tile3,1 | 16 | 35,801 | 1,150.00 | 28.49 | 44.52 | 14.64 | 22.88 |
| tile3,2 | 16 | 35,802 | 1,150.00 | 28.49 | 44.52 | 14.64 | 22.88 |

</details>

### Details for Successful Runs

<details><summary><strong>[whole_array] M=256 K=64 N=64</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,2 | 8 | 12,940 | 1,150.00 | 28.49 | 44.52 | 20.26 | 31.65 |
| tile2,1 | 8 | 12,941 | 1,150.00 | 28.49 | 44.52 | 20.26 | 31.65 |
| tile3,2 | 8 | 13,352 | 1,150.00 | 28.49 | 44.52 | 19.63 | 30.68 |
| tile3,1 | 8 | 13,352 | 1,150.00 | 28.49 | 44.52 | 19.63 | 30.68 |

</details>

### Details for Successful Runs

<details><summary><strong>[whole_array] M=64 K=128 N=128</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,2 | 8 | 13,573 | 1,150.00 | 28.49 | 44.52 | 19.31 | 30.18 |
| tile2,1 | 8 | 13,573 | 1,150.00 | 28.49 | 44.52 | 19.31 | 30.18 |
| tile3,1 | 8 | 13,573 | 1,150.00 | 28.49 | 44.52 | 19.31 | 30.18 |
| tile3,2 | 8 | 14,627 | 1,150.00 | 28.49 | 44.52 | 17.92 | 28.00 |

</details>

### Details for Successful Runs

<details><summary><strong>[whole_array] M=64 K=128 N=256</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,2 | 16 | 28,364 | 1,150.00 | 28.49 | 44.52 | 18.48 | 28.88 |
| tile2,1 | 16 | 28,364 | 1,150.00 | 28.49 | 44.52 | 18.48 | 28.88 |
| tile3,1 | 16 | 29,503 | 1,150.00 | 28.49 | 44.52 | 17.77 | 27.77 |
| tile3,2 | 16 | 29,510 | 1,150.00 | 28.49 | 44.52 | 17.77 | 27.76 |

</details>

### Details for Successful Runs

<details><summary><strong>[whole_array] M=64 K=128 N=64</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,2 | 4 | 6,178 | 1,150.00 | 28.49 | 44.52 | 21.22 | 33.15 |
| tile2,1 | 4 | 6,179 | 1,150.00 | 28.49 | 44.52 | 21.21 | 33.14 |
| tile3,2 | 4 | 6,686 | 1,150.00 | 28.49 | 44.52 | 19.60 | 30.63 |
| tile3,1 | 4 | 6,694 | 1,150.00 | 28.49 | 44.52 | 19.58 | 30.59 |

</details>

### Details for Successful Runs

<details><summary><strong>[whole_array] M=64 K=256 N=128</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile3,1 | 16 | 29,351 | 1,150.00 | 28.49 | 44.52 | 17.86 | 27.91 |
| tile2,1 | 16 | 29,359 | 1,150.00 | 28.49 | 44.52 | 17.86 | 27.90 |
| tile2,2 | 16 | 30,307 | 1,150.00 | 28.49 | 44.52 | 17.30 | 27.03 |
| tile3,2 | 16 | 30,308 | 1,150.00 | 28.49 | 44.52 | 17.30 | 27.03 |

</details>

### Details for Successful Runs

<details><summary><strong>[whole_array] M=64 K=256 N=256</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile3,1 | 32 | 62,158 | 1,150.00 | 28.49 | 44.52 | 16.87 | 26.36 |
| tile2,1 | 32 | 62,165 | 1,150.00 | 28.49 | 44.52 | 16.87 | 26.36 |
| tile2,2 | 32 | 62,866 | 1,150.00 | 28.49 | 44.52 | 16.68 | 26.06 |
| tile3,2 | 32 | 62,867 | 1,150.00 | 28.49 | 44.52 | 16.68 | 26.06 |

</details>

### Details for Successful Runs

<details><summary><strong>[whole_array] M=64 K=256 N=64</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 8 | 12,953 | 1,150.00 | 28.49 | 44.52 | 20.24 | 31.62 |
| tile3,1 | 8 | 12,953 | 1,150.00 | 28.49 | 44.52 | 20.24 | 31.62 |
| tile3,2 | 8 | 13,694 | 1,150.00 | 28.49 | 44.52 | 19.14 | 29.91 |
| tile2,2 | 8 | 13,702 | 1,150.00 | 28.49 | 44.52 | 19.13 | 29.89 |

</details>

### Details for Successful Runs

<details><summary><strong>[whole_array] M=64 K=64 N=128</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 4 | 5,858 | 1,150.00 | 28.49 | 44.52 | 22.37 | 34.96 |
| tile2,2 | 4 | 5,858 | 1,150.00 | 28.49 | 44.52 | 22.37 | 34.96 |
| tile3,2 | 4 | 5,858 | 1,150.00 | 28.49 | 44.52 | 22.37 | 34.96 |
| tile3,1 | 4 | 5,858 | 1,150.00 | 28.49 | 44.52 | 22.37 | 34.96 |

</details>

### Details for Successful Runs

<details><summary><strong>[whole_array] M=64 K=64 N=256</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile2,1 | 8 | 12,940 | 1,150.00 | 28.49 | 44.52 | 20.26 | 31.65 |
| tile3,2 | 8 | 12,940 | 1,150.00 | 28.49 | 44.52 | 20.26 | 31.65 |
| tile2,2 | 8 | 12,941 | 1,150.00 | 28.49 | 44.52 | 20.26 | 31.65 |
| tile3,1 | 8 | 12,941 | 1,150.00 | 28.49 | 44.52 | 20.26 | 31.65 |

</details>

### Details for Successful Runs

<details><summary><strong>[whole_array] M=64 K=64 N=64</strong></summary>

| Tile | Kernels | Total cycles | Avg cycles per kernel | MACs/cycle (kernel) | Peak eff. kernel % | MACs/cycle (system) | Peak eff. system % |
|------|---------|--------------|-----------------------|---------------------|--------------------|---------------------|--------------------|
| tile3,2 | 2 | 2,317 | 1,150.00 | 28.49 | 44.52 | 28.28 | 44.20 |
| tile3,1 | 2 | 2,317 | 1,150.00 | 28.49 | 44.52 | 28.28 | 44.20 |
| tile2,2 | 2 | 2,317 | 1,150.00 | 28.49 | 44.52 | 28.28 | 44.20 |
| tile2,1 | 2 | 2,317 | 1,150.00 | 28.49 | 44.52 | 28.28 | 44.20 |

</details>
