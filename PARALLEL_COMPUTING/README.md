# 2D Heat Equation (Jacobi) — OpenMP, MPI and Hybrid MPI+OpenMP

Semester project for *Programowanie Równoległe* (PR-2023L), Computational Engineering,
ICM University of Warsaw. Course instructor: dr Dorota Dąbrowska.

Solves the steady-state heat equation on a square plate with Dirichlet boundary conditions
using Jacobi iteration, implemented four ways — sequential, shared memory (OpenMP),
distributed memory (MPI) and hybrid — then benchmarked on a cluster up to 256 processes.

📄 [Full report (PDF, Polish, 65 pages)](./PR%20-%20Filip%20Rusiecki%20-%20451709%20(7).pdf)

---

## Problem

Laplace's equation ∇²T = 0 on a 1024 × 1024 grid, discretized with a 5-point stencil and
solved by Jacobi iteration:

```
T_new[i][j] = ( T[i-1][j] + T[i+1][j] + T[i][j-1] + T[i][j+1] ) / 4
```

| Parameter | Value |
|---|---|
| Grid | 1024 × 1024 |
| Boundary temperatures | top 70.0, bottom 1.0, left 20.0, right 50.0 |
| Convergence criterion | RMS change between iterations < 6 × 10⁻⁵ |
| Iterations to converge | 181,254 |

Two buffers are allocated and swapped on alternating iterations, so no array copy is ever
needed — the new values are written to one array while the old ones are read from the other.

![Converged temperature field](./SEQUENTIAL/temperature.png)

---

## Results

Baseline: **1473 s** sequential (single core). All speedups are relative to this.

### OpenMP — single node, shared memory

| Threads | Time [s] | Speedup | Efficiency |
|---:|---:|---:|---:|
| 1 | 1472.76 | 1.00 | 100% |
| 2 | 749.68 | 1.96 | 98% |
| 4 | 377.87 | 3.90 | 97% |
| 8 | 206.60 | 7.13 | 89% |
| 16 | 107.58 | 13.69 | 86% |
| 32 | 102.24 | 14.41 | 45% |

The 1-thread OpenMP time matches the sequential baseline to within 0.02%, confirming that
the parallel runtime itself adds no measurable overhead.

### MPI — distributed memory

| Processes | Time [s] | Speedup | Efficiency | Node layout |
|---:|---:|---:|---:|---|
| 2 | 743.75 | 1.98 | 99% | 1 proc/node |
| 4 | 375.26 | 3.93 | 98% | 1 proc/node |
| 8 | 195.90 | 7.52 | 94% | 1 proc/node |
| 16 | 98.98 | 14.88 | 93% | 1 proc/node |
| 32 | 53.62 | 27.47 | 86% | 16 nodes × 2 |
| 64 | 31.17 | 47.26 | 74% | 16 nodes × 4 |
| 128 | 21.75 | 67.74 | 53% | 16 nodes × 8 |
| 256 | 12.94 | 113.81 | 44% | 16 nodes × 16 |

25 minutes down to 13 seconds.

### Hybrid MPI + OpenMP

Did not beat pure MPI at equivalent core counts. Measured roughly 3–4× slower than the
naive `threads × processes` expectation. See [Known issues](#known-issues) — several
defects in the hybrid code account for much of this.

![Hybrid heat map](./HYBRID_MPI_OPENMP/heatmap_openMP_MPI.png)

---

## Analysis

### The OpenMP plateau is memory bandwidth, not synchronization

Going from 16 to 32 threads bought almost nothing (107.6 → 102.2 s). The reason is that
Jacobi on this grid is **memory-bandwidth bound**, not compute bound.

Two 1024² double arrays occupy 16.8 MB, well beyond L3 on any single socket. Every
iteration therefore streams the whole working set from main memory:

```
181,254 iterations × 16.8 MB ≈ 3.0 TB of memory traffic
```

At roughly 100 GB/s of sustained bandwidth that puts a floor near **30 s** for a single
node — and the measured 32-thread time of 102 s sits within about 3.5× of it. Adding cores
past that point does not help, because cores were never the bottleneck.

The sequential run confirms it independently: ~500 × 10⁹ floating-point operations in
1473 s is ≈ **340 MFLOP/s**, two orders of magnitude below what a modern core can sustain
when it is not waiting on memory.

### The MPI-vs-OpenMP comparison is not apples-to-apples

For 2–16 processes the MPI runs used **one process per node**. So the MPI column at
16 processes describes 16 separate machines, while the OpenMP column at 16 threads
describes one. MPI is not scaling better than OpenMP there — it has 16× the aggregate
memory bandwidth and 16× the last-level cache.

A fair shared-vs-distributed comparison would pin both to the same physical hardware.
The honest reading of these tables: **OpenMP saturates one node at around 16 threads, and
MPI lets you add more nodes.** That is a statement about hardware, not about the merits of
either programming model.

### Where strong scaling ends

Efficiency falls off a cliff past 128 processes, and the cause is arithmetic. At 256
processes each rank owns **4 grid rows**. The halo exchange moves 1024 doubles in each
direction per neighbour — comparable in volume to the rank's entire local computation.
Doubling from 128 to 256 processes returns only 1.68×, which is exactly the strong-scaling
limit you would predict once communication volume approaches computation volume.

### Convergence checking dominates at scale

`MPI_Allreduce` is called once per iteration — **181,254 global reductions**. At 256 ranks
spread over 16 nodes, at an order of 20 µs per reduction, that is roughly 3.6 s of the
12.9 s total: about a quarter of the runtime spent deciding whether to stop.

Checking convergence every 100 iterations instead would be mathematically harmless (the
residual changes very slowly) and is the single largest easy win available in this code.

---

## Repository layout

```
SEQUENTIAL/
  HEAT_TRANSFER/A4 HEAT SEQ.c        single-threaded reference
  temperature.png                     converged field

OPENMP/
  HEAT_TRANSFER/A4_HEAT_OPENMP.c     shared-memory version
  zad1_OpenMP.c                       task S1 — vector addition A = B + C
  zad2_openMP.c                       task S2 — L2 norm

MPI/
  HEAT_TRANSFER/A4_HEAT_MPI.c        1D row decomposition, non-blocking halo exchange
  zad1_MPI.c                          task S1
  zad2_MPI.c                          task S2

HYBRID_MPI_OPENMP/
  HEAT_TRANSFER/A4_HEAT_MPI_OPENMP_WERSJA1.c    attempt 1 — parallel region inside the solver
  HEAT_TRANSFER/A4_HEAT_MPI_OPENMP_WERSJA2.cpp  attempt 2 — MPI_THREAD_SERIALIZED, single region
  heatmap_openMP_MPI.png

PR - Filip Rusiecki - 451709 (7).pdf  full report with plots and SLURM scripts
```

### MPI decomposition

Rows are split across ranks. Rank 0 and the last rank carry one ghost row each (they own a
physical boundary); interior ranks carry two. The row counts work out to exactly N−2
computed interior rows across all ranks.

Halo exchange posts `MPI_Irecv` before `MPI_Isend` and closes with `MPI_Waitall`, so
neighbouring ranks cannot deadlock on buffering.

### Side tasks

- **S1** — parallel vector addition, A[i] = B[i] + C[i], n ≈ 6 × 10⁶. OpenMP uses
  per-thread `rand_r` seeds to keep random fill thread-safe; MPI splits into local arrays
  with no communication at all, which is why it scales near-linearly.
- **S2** — L2 norm of a vector, n = 2 × 10⁸. OpenMP uses `reduction(+:sum)`; MPI uses
  `MPI_Reduce` to rank 0. The single reduction is visible in the timings at small scale.

---

## Building

```bash
# sequential (uses omp_get_wtime for timing, hence -fopenmp)
gcc -fopenmp -O2 -o heat_seq "SEQUENTIAL/HEAT_TRANSFER/A4 HEAT SEQ.c" -lm

# OpenMP
gcc -fopenmp -O2 -o heat_omp OPENMP/HEAT_TRANSFER/A4_HEAT_OPENMP.c -lm
OMP_NUM_THREADS=16 ./heat_omp

# MPI
mpicc -O2 -o heat_mpi MPI/HEAT_TRANSFER/A4_HEAT_MPI.c -lm
mpirun -np 16 ./heat_mpi

# hybrid (version 2)
mpicxx -fopenmp -O2 -o heat_hybrid HYBRID_MPI_OPENMP/HEAT_TRANSFER/A4_HEAT_MPI_OPENMP_WERSJA2.cpp -lm
```

Verified compiling with GCC 13.3. SLURM batch scripts used for the cluster runs are
included at the end of the report PDF.

---

## Known issues

Found on a later review of this code. Left documented rather than silently patched, since
the benchmark numbers above were produced by it.

- **`A4_HEAT_MPI_OPENMP_WERSJA1.c` does not compile.** `return` appears inside an OpenMP
  structured block (illegal branch out of a parallel region), and the `default(none)`
  clause omits `local_error` and `local_n` while listing a variable that does not exist in
  that scope.
- **`WERSJA2` can deadlock.** `global_error` is declared inside the parallel region and is
  therefore private to each thread. Only the thread executing the `single` block receives
  the reduced value; that thread eventually breaks out of the loop while the others, still
  seeing the initial value, re-enter and block forever on the next `omp barrier`. Fix:
  hoist the declaration above `#pragma omp parallel` and add it to the `shared` clause.
- **`WERSJA2` hardcodes `omp_set_num_threads(2)`**, overriding whatever thread count the
  job requested. This likely explains a large part of the disappointing hybrid timings.
- **The parallel region in `WERSJA1` is created inside the solver function**, so it is
  entered and torn down 181,254 times. `WERSJA2` correctly hoists it outside the loop.
- **No communication/computation overlap.** Both hybrid versions call `MPI_Waitall` and
  then compute. Posting the halo exchange, computing the interior rows (which do not
  depend on the halo), then waiting and finishing the two boundary rows would hide the
  communication entirely.
- **`N % size` is unchecked.** With a process count that does not divide 1024 evenly, rows
  are silently dropped. The `local_n > N + 2` guard can never trigger and protects nothing.
- **Local arrays are stack-allocated VLAs.** At 2 processes this is ~8.4 MB against a
  typical 8 MB stack limit.
- **The sequential file in this directory is set to `N 512`**, while the report benchmarks
  1024. Change the `#define` to reproduce the 1473 s baseline.
- **The stopping criterion measures the change between iterates, not the residual.** For
  Jacobi the per-iteration change underestimates the true error substantially, so the
  converged solution is less accurate than ε = 6 × 10⁻⁵ suggests.

---

## What I would do differently

- Overlap halo exchange with interior computation — the fix the hybrid version actually needed
- Reduce convergence checks to once every ~100 iterations
- Benchmark shared vs distributed memory on identical hardware before drawing conclusions
  about either
- Report a roofline estimate alongside speedup, so the memory-bandwidth ceiling is visible
  instead of being mistaken for synchronization overhead
- Red-black Gauss-Seidel or SOR instead of Jacobi. The assignment specified Jacobi, but SOR
  with a well-chosen relaxation factor converges in O(N) rather than O(N²) iterations —
  an algorithmic speedup that would dwarf everything achieved here by parallelization

---

## References

1. Pacheco, P. S. & Malensek, M. (2022). *An Introduction to Parallel Programming*, 2nd ed.
2. [Parallel programming with MPI and OpenMP](https://www.dcc.fc.up.pt/~ricroc/aulas/1516/cp/apontamentos/slides_mpi_openmp.pdf) — R. Rocha, University of Porto
3. Thread-safe random number generation with `rand_r` —
   [Stack Overflow](https://stackoverflow.com/questions/3973665/how-do-i-use-rand-r-and-how-do-i-use-it-in-a-thread-safe-way)
