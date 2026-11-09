# duet

**Run one LLM on your laptop's CPU and integrated GPU together, and pick the faster chip for each job.**

duet is a small scheduler on top of [llama.cpp](https://github.com/ggml-org/llama.cpp), plus a faster
CPU attention kernel for generating tokens. It was built and measured on an Intel i5-12500H with Iris Xe
graphics (16 GB RAM), running MiniCPM5-1B Q4_K_M.

- **Reading prompts → GPU.** About 2.5–3.5× faster than the CPU on this laptop.
- **Generating tokens → whichever chip is faster at the current context length.** With the new CPU
  attention kernel that's the CPU at every length measured. `--calibrate` measures the crossover on your machine.
- **Moving between chips:** the chat memory (KV cache) is copied over (~50 ms, ~15 MB at 500 tokens).
  If a chip is only a few tokens behind, it re-reads them instead.

Result: up to **1.9× faster than stock llama.cpp on the CPU** at 4000 tokens of context, and 7–26% faster
than llama.cpp on the GPU (tables below).

## Download

A ready-built Windows x64 zip (no compiler needed) is on the [Releases page](https://github.com/krish-patel-01/duet/releases/latest). Unzip it and run `hybrid.exe -m model.gguf --chat -i`. You need the [VC++ 2015–2022 x64 runtime](https://aka.ms/vs/17/release/vc_redist.x64.exe) and a Vulkan GPU driver.

## What's in here

| Path | What it is |
|---|---|
| `src/hybrid.cpp` | The CPU + GPU scheduler (`hybrid.exe`) |
| `patches/fast-decode-attention.patch` | The CPU attention kernel, for llama.cpp `6a1a922d2` (build b10819) |
| `patches/fast-decode-attention-b11969.patch` | The same kernel for llama.cpp b11969 (v0.8.0) |
| `build.bat` | Builds `hybrid.exe` |
| `build_cpu_backend.bat`, `build_cpu_backend_v3.bat` | Build the patched CPU backend DLLs (b10819 / b11969) |
| `bench*.ps1` | Benchmarks (hybrid vs CPU vs GPU, CPU speed by context length, llama-server) |
| `check_kernel_server.py`, `compare_traces.py` | Correctness checks for the kernel |
| `lib/*.def` | Export lists used to make import libraries for the llama.cpp DLLs |
| `docs/BASELINE.md` | The starting measurements that shaped the design |

## Setup (Windows x64)

Needs Visual Studio 2022 Build Tools with the C++ workload and **clang** (C++ Clang tools for Windows), and git.

```bat
git clone https://github.com/krish-patel-01/duet
cd duet

rem 1. llama.cpp source at the matching commit, with the kernel patch
git clone https://github.com/ggml-org/llama.cpp third_party\llama.cpp
git -C third_party\llama.cpp checkout 6a1a922d2
git -C third_party\llama.cpp apply ..\..\patches\fast-decode-attention.patch

rem 2. llama.cpp's prebuilt Vulkan DLLs from release b10819 go in bin\
rem    https://github.com/ggml-org/llama.cpp/releases/tag/b10819  ->  llama-b10819-bin-win-vulkan-x64.zip

rem 3. build
build.bat
build_cpu_backend.bat

rem 4. a model in models\ (e.g. MiniCPM5-1B-Q4_K_M.gguf)
```

For llama.cpp b11969, clone into `third_party\llama.cpp-b11969`, check out `b11969`, apply
`fast-decode-attention-b11969.patch` and run `build_cpu_backend_v3.bat`.

## Use

```
bin\hybrid.exe -m models\MiniCPM5-1B-Q4_K_M.gguf --chat -p "Why is the sky blue?" -n 200
bin\hybrid.exe -m models\MiniCPM5-1B-Q4_K_M.gguf -i --no-think          # multi-turn chat
bin\hybrid.exe -m models\MiniCPM5-1B-Q4_K_M.gguf --chat -f doc.txt --calibrate
```

Options: `--mode hybrid|cpu|gpu`, `--switch-at N`, `--calibrate`, `-t` / `-tb` (CPU threads),
`-c` (context), `--temp/--top-k/--top-p/--seed`, `--no-think`, `--bench N`, `--selftest`, `-v`.

## Check and benchmark

```
bin\hybrid.exe -m models\MiniCPM5-1B-Q4_K_M.gguf --selftest   # hand-off correctness
powershell -File bench.ps1                                     # hybrid vs cpu vs gpu (plug in the charger first)
```

`--selftest` compares next-token scores computed after a hand-off with those computed natively.
Differences match normal CPU-vs-GPU rounding (cosine ≈ 0.999, same top token). In `--bench` runs,
greedy output was token-for-token identical across hybrid, CPU-only and GPU-only (same hash).

## Fast CPU decode attention (custom kernel)

`bin\ggml-cpu-alderlake.dll` is built from llama.cpp 6a1a922d2 plus
[patches/fast-decode-attention.patch](patches/fast-decode-attention.patch) (`build_cpu_backend.bat`).
The prebuilt DLL is kept in `dll-variants\prebuilt.dll`. Keep extra DLLs out of `bin\`, because the
backend loader tries every `ggml-cpu-*.dll` it finds there.

For one-token decode with an F16 KV cache, llama.cpp's CPU flash attention handles one query head at
a time. MiniCPM5 has 16 query heads sharing 2 KV heads, so every K/V row was re-read 8 times, keys were
processed one at a time with scalar `expf`, and V was accumulated in F16. The new kernel:
- processes all 8 query heads of a KV group together, so each K and V row is loaded once;
- works in blocks of 64 keys with a vectorised online softmax and FP32 accumulators in registers;
- hands out key blocks dynamically across threads, so fast P-cores take more work than E-cores, and
  merges the per-thread partial results with llama.cpp's existing reduce step;
- prefetches each key's V row and the K rows ahead while computing scores.

Set `GGML_FA_DECODE=0` to fall back to the stock path. Prompt reading (more than one token) is unchanged.

Check: `--logit-trace` feeds the same tokens one at a time with the kernel on and off, and
`compare_traces.py` compares every step's logits. At 100 and 3000 tokens of context, the top token
agreed at 100% of steps, and the differences matched normal CPU-vs-GPU rounding (cosine ≥ 0.9975, KL ≤ 2e-3).

CPU-only generation speed (6 threads, same conditions back to back):

| Context | Stock llama.cpp | New kernel |
|---:|---:|---:|
| 64   | ~46 tok/s | ~46 tok/s |
| 1024 | 40 | 49 |
| 2048 | 28 | 42 |
| 4096 | 18–21 | 36–40 (**~1.9×**) |
| 6000 | 15 | 31–36 (**~2.2×**) |

With the kernel, the CPU generates faster than the Iris Xe at every context up to 4096 (calibration:
39.5 vs 30.0 tok/s at 4k), and the GPU slows down faster as context grows. So hybrid now reads
prompts on the GPU and generates on the CPU.

## Final results (plugged in, median of 3, 300 generated tokens)

`cpu-stock` = prebuilt llama.cpp on the CPU. `gpu` = llama.cpp Vulkan on the Iris Xe. `hybrid` = duet.

| Prompt | Mode | Prompt tok/s | First token | Gen tok/s | Total time |
|---:|---|---:|---:|---:|---:|
| 50   | cpu-stock | 250 | 0.20 s | 44.9 | 6.86 s |
| 50   | cpu (new kernel) | 267 | 0.19 s | 46.2 | 6.67 s |
| 50   | gpu       | 339 | 0.15 s | 36.7 | 8.30 s |
| 50   | **hybrid** | 362 | 0.14 s | 46.3 | **6.60 s** |
| 500  | cpu-stock | 248 | 2.02 s | 37.1 | 10.05 s |
| 500  | cpu (new kernel) | 260 | 1.93 s | 45.9 | 8.44 s |
| 500  | gpu       | 586 | 0.85 s | 35.7 | 9.22 s |
| 500  | **hybrid** | 593 | 0.84 s | 44.7 | **7.58 s** |
| 2000 | cpu-stock | 202 | 9.89 s | 25.4 | 21.68 s |
| 2000 | cpu (new kernel) | 208 | 9.62 s | 40.9 | 16.93 s |
| 2000 | gpu       | 450 | 4.44 s | 32.5 | 13.64 s |
| 2000 | **hybrid** | 440 | 4.55 s | 40.0 | **12.03 s** |
| 4000 | cpu-stock | 164 | 24.45 s | 17.8 | 41.28 s |
| 4000 | cpu (new kernel) | 167 | 24.01 s | 35.7 | 32.42 s |
| 4000 | gpu       | 311 | 12.88 s | 29.2 | 23.11 s |
| 4000 | **hybrid** | 307 | 13.03 s | 35.3 | **21.50 s** |

Hybrid vs stock llama.cpp CPU: 4% faster at 50 tokens, 1.33× at 500, 1.8× at 2000, **1.9× at 4000**.
Hybrid vs llama.cpp on the GPU: 26%, 22%, 13% and 7% faster.

## Earlier results (before the attention kernel; charger plugged in, median of 3 runs, 300 generated tokens)

MiniCPM5-1B Q4_K_M, i5-12500H + Iris Xe, 2026-10-25. `cpu` mode ≈ stock llama.cpp CPU, `gpu` mode ≈ stock llama.cpp Vulkan.

| Prompt | Mode | Prompt tok/s | First token | Gen tok/s | Total time |
|---:|---|---:|---:|---:|---:|
| 50   | CPU    | 251 | 0.20 s | 47.5 | 6.50 s |
| 50   | GPU    | 375 | 0.13 s | 40.2 | 7.58 s |
| 50   | **Hybrid** | 373 | 0.13 s | **48.1** | **6.35 s** |
| 500  | CPU    | 241 | 2.07 s | 40.4 | 9.47 s |
| 500  | GPU    | 654 | 0.77 s | 39.3 | **8.38 s** |
| 500  | Hybrid | 636 | 0.79 s | 38.7 | 8.51 s |
| 2000 | CPU    | 201 | 9.93 s | 27.6 | 20.75 s |
| 2000 | GPU    | 514 | 3.90 s | 36.2 | **12.15 s** |
| 2000 | Hybrid | 501 | 3.99 s | 36.0 | 12.29 s |

What this shows:
- Hybrid always lands on the best chip on its own: within ~1.5% of the best single mode at every length,
  and the fastest overall on short prompts (GPU reads the prompt, CPU generates).
- vs CPU-only (stock llama.cpp CPU): 2% faster at 50 tokens, 10% at 500, **41% faster (1.7×) at 2000**.
  Time to first token is 2.5–2.7× faster for longer prompts.
- vs GPU-only: 16% faster at 50 tokens, ~1–2% slower at 500 and 2000 (calibration time is not included,
  but warm-up and the two-model setup cost a little).
- At 500 tokens the CPU/GPU crossover sits right at the prompt length, so hybrid mostly picks the GPU;
  at this context the two chips are within ~3% anyway.
- Generation itself can't beat the faster chip, since each chip is at its own limit (RAM speed for the CPU,
  attention cost for both at long context). The remaining gains have to come from a faster CPU attention
  kernel (so the CPU stays ahead longer) or from speculative decoding.

## Problems found and fixed

1. **Wrong chat template.** llama.cpp's built-in matcher treats MiniCPM5's ChatML template as the
   old MiniCPM format, and this tokenizer doesn't add the BOS the template asks for. Fixed by formatting
   ChatML directly and adding BOS.
2. **KV restore landed after the write head.** `llama_state_seq_set_data` places cells after the
   cache's current head, so later steps attended over a bigger window. Fixed by fully clearing the
   receiving cache first.
3. **OpenMP spin-wait starved the iGPU.** CPU-backend worker threads spin ~200 ms after each job. GPU
   steps still run a tiny CPU graph, so the threads never slept, used up the shared power budget,
   and GPU steps doubled from ~25 ms to ~50 ms. Fixed by switching `kmp_set_blocktime` per step:
   spin for CPU steps, no spin for GPU steps.
4. **First-use shader compiles.** The GPU's first small-batch prompt took ~3.8 s while pipelines
   compiled. Fixed by warming up common batch sizes at startup.


## Building notes

`build_cpu_backend.bat` builds the CPU backend with the attention kernel and installs it into `bin\`
(CMake + Ninja + clang-cl from VS Build Tools, linked against LLVM's `libomp.dll` like the prebuilt one).
`build_cpu_backend.bat all` builds all 14 x64 CPU variants. `bench_cpu_ctx.ps1 [-Dll path]` measures CPU
generation speed at several context lengths for any backend DLL.

Keep extra `ggml-cpu-*.dll` files out of `bin\`: the backend loader tries every one it finds there.

## License

MIT. Built on [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT, © the ggml authors).
