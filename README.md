# Computational Engineering — ICM, University of Warsaw

Coursework and projects from the Computational Engineering programme at the
Interdisciplinary Centre for Mathematical and Computational Modelling (ICM),
University of Warsaw. Parallel programming in C/C++, applied data analysis in the social
sciences, natural language processing, and air traffic data.

Author: Filip Rusiecki

---

## Featured

### 🔬 [Parallel Computing — 2D heat equation](./PARALLEL_COMPUTING)

Jacobi solver for the steady-state heat equation on a 1024×1024 grid, implemented four
ways — sequential, OpenMP, MPI, and hybrid — then benchmarked on a cluster up to 256
processes.

**25 minutes down to 13 seconds; speedup of 114× at 256 MPI processes.**

Includes 1D domain decomposition with non-blocking halo exchange, full speedup and
efficiency tables, and a roofline analysis showing why the OpenMP version plateaus at 16
threads (memory bandwidth, not synchronization).

`C` · `OpenMP` · `MPI` · `SLURM`

---

### 📊 [Alcohol market in Poland and Europe](./ALCOHOL_MARKET)

Analysis of alcohol affordability, pricing and consumption patterns in Poland from 1990 to
2023, benchmarked against the rest of Europe. Data from GUS, KCPU/PARPA, Eurostat and WHO,
ingested automatically and visualised with 23 figures including choropleth maps.

**Only 1.6% of Poles drink daily — one of Europe's lowest rates — yet Poland's annual
per-capita consumption exceeds Portugal's (20.7% daily) and Italy's (12.1%).** The report
works out why, and what it means for how alcohol policy should be evaluated.

`Python` · `pandas` · `matplotlib` · `GeoPandas` · `Eurostat API`

---

## Contents

| Directory / file | Course | Description |
|---|---|---|
| [`PARALLEL_COMPUTING/`](./PARALLEL_COMPUTING) | PR-2023L | Heat equation in OpenMP, MPI and hybrid; vector addition and L2 norm exercises. Benchmarks up to 256 processes. |
| [`ALCOHOL_MARKET/`](./ALCOHOL_MARKET) | ONS-2024Z | Alcohol consumption, pricing and affordability in Poland and Europe. |
| [`NLP/`](./NLP) | NLP-2024Z | Classifier for detecting abusive clauses in Polish contracts, on the `laugustyniak/abusive-clauses-pl` dataset. Handles class imbalance via weighting and balanced validation; optimised for F1 to balance false accusations against missed abuses. |
| [`DATABASES/`](./DATABASES) | — | Source data for the alcohol project: GUS price and spending tables, PARPA yearly consumption (2014–2023), WHO exports. |
| `ATC_BADA.ipynb` | — | Flight trajectory calculation and analysis using EUROCONTROL BADA aircraft performance data. |

---

## Courses

| Code | Course | Instructor |
|---|---|---|
| PR-2023L | Programowanie Równoległe | dr Dorota Dąbrowska |
| ONS-2024Z | Obliczenia naukowe w naukach społecznych | dr hab. Dominik Batorski |
| NLP-2024Z | Natural Language Processing | — |

---

## Related repositories

- [**Praca_Magisterska**](https://github.com/RusieckiFilip/Praca_Magisterska) — master's
  thesis: 3D CNN for lung nodule detection and malignancy classification in CT
- [**rltraffic**](https://github.com/RusieckiFilip/rltraffic) — offline multi-agent Decision
  Transformer for traffic signal control
- [**Praca-Inzynierska**](https://github.com/RusieckiFilip/Praca-Inzynierska) — engineering
  thesis, Aircraft Engines Division, MEiL
- [**ATC_ICM**](https://github.com/RusieckiFilip/ATC_ICM) — flight trajectory analysis with
  PAŻP, EUROCONTROL and INDRA

---

Each project directory has its own README with methodology, results and known limitations.
