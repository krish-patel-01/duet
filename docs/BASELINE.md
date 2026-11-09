# CPU inference baseline — MiniCPM5-1B Q4_K_M

Machine: Intel i5-12500H (4 P-cores + 8 E-cores, AVX2 + AVX-VNNI, no AVX-512), 16 GB DDR4-3200 dual channel.
Engine: llama.cpp build 6a1a922d2 (10819), `ggml-cpu-alderlake` backend, from `D:\runtime\win-x64\cpu`.
Model: `D:\models\MiniCPM5-1B-Q4_K_M.gguf` (651 MiB, 1.08B params).

## llama.cpp results (llama-bench, pp512 / tg128, mlock)

| Threads / pinning            | Prompt (tok/s) | Generation (tok/s) |
|------------------------------|---------------:|-------------------:|
| 4 (unpinned)                 | 249 | 56.7 |
| 6                            | 249 | **56.9** |
| 8                            | 274 | 54.2 |
| 12                           | 267 | 55.4 |
| 16                           | **294** | 53.7 |
| 4 P-cores (0x55)             | 236 | 56.0 |
| 8 P-core threads (0xFF)      | 226 | 57.3 |
| 8 E-cores only (0xFF00)      | 158 | 49.2 |
| 4 P + 8 E (0xFF55)           | 273 | 55.6 |

## Measured RAM read bandwidth (AVX2 streaming read, 1 GiB)

| Threads | GB/s |
|--------:|-----:|
| 4  | 34.0 |
| 8  | 34.6 |
| 12 | 36.6 |
| 16 | 37.0 |

## What this means

- **Generation is at the hardware limit.** 57 tok/s × ~0.65 GB of weights read per token ≈ 37 GB/s, which matches
  the measured RAM ceiling. A faster matmul kernel cannot raise this. Thread pinning doesn't help either.
- **The only way to beat it at Q4 is to produce more than one token per weight read.** Options:
  speculative decoding with a smaller draft model, n-gram / prompt-lookup drafting (no extra model), or
  multi-token prediction heads. Checking 4–8 drafted tokens in one pass costs about the same RAM traffic as 1 token.
  Expected: 1.3–2.5× on repetitive/structured text (code, RAG, edits), less on free-form chat.
- **Prompt reading is compute-bound and has headroom.** 294 tok/s ≈ 0.64 TOPS. Rough int8 peak with AVX-VNNI on
  4 P + 8 E cores is ~2–3 TOPS, so a well-tuned Q4_K × int8 VNNI GEMM with P/E-aware work splitting
  could plausibly reach 1.3–2× llama.cpp here.

## Iris Xe (Vulkan) results

Device: Intel Iris Xe (80 EU), driver 32.0.101.7088, shares system RAM (UMA), fp16 + int8 dot, no matrix cores.

| Test | CPU best | Iris Xe (all layers, -ngl 99) |
|------|---------:|------------------------------:|
| Prompt pp512 | 294 | **927** (3.2×) |
| Prompt pp2048 | 247 | **605** (2.4×) |
| Generation, empty context | **57** | 43 |
| Generation @ 512 ctx | **43** | 42 |
| Generation @ 1024 ctx | 37 | **40** |
| Generation @ 2048 ctx | 29 | **38** |
| Generation @ 4096 ctx | 20 (26 with 12 threads / fa off) | **31** |

- Partial offload (ngl 8/16) is worse than either pure mode. Weights on the CPU with GPU op-offload (ngl 0) gave no prompt speedup.
- ubatch 256/512/1024 makes almost no difference on the GPU.
- **CPU generation falls off fast as context grows** (57 → 20 tok/s at 4k). The GPU falls much less, and the CPU gets
  faster with flash-attn *off* or more threads, so CPU attention is compute-bound and poorly optimized, not a RAM limit.
  This is a real target for a custom kernel.

### Best setup today, with no new code
- Short chats: CPU build, 6 threads (~57 tok/s gen, ~250–290 tok/s prompt).
- Long prompts / RAG / long chats: Vulkan build, `-ngl 99 -fa on` (~600–900 tok/s prompt, 38–42 tok/s gen at 1–4k context).

## Hybrid scheduler results (built — see hybrid/README.md)

Plugged in, median of 3, 300 generated tokens, total time:

| Prompt | CPU-only | GPU-only | Hybrid |
|---:|---:|---:|---:|
| 50   | 6.50 s | 7.58 s | **6.35 s** |
| 500  | 9.47 s | **8.38 s** | 8.51 s |
| 2000 | 20.75 s | **12.15 s** | 12.29 s |

Hybrid tracks the best chip automatically: up to 1.7× faster than CPU-only llama.cpp on long prompts,
16% faster than GPU-only on short ones.

## With the fast CPU attention kernel (see hybrid/README.md)

Plugged in, median of 3, 300 generated tokens, total time:

| Prompt | Stock llama.cpp CPU | llama.cpp GPU | Hybrid + new kernel |
|---:|---:|---:|---:|
| 50   | 6.86 s | 8.30 s | **6.60 s** |
| 500  | 10.05 s | 9.22 s | **7.58 s** (1.33× vs CPU) |
| 2000 | 21.68 s | 13.64 s | **12.03 s** (1.8×) |
| 4000 | 41.28 s | 23.11 s | **21.50 s** (1.9×) |

CPU generation at 4k context: 17.8 → 35.7 tok/s (2×).

## Proposed engine plan (updated)

0. Hybrid scheduler: run the prompt on the Iris Xe (3× faster), then hand the KV cache to whichever device is faster
   for generation at the current context length (CPU under ~800 tokens, GPU above). Since memory is shared (UMA),
   the hand-off can be zero-copy.
0b. Fast CPU decode attention: split-K across P+E cores, int8/f16 KV, AVX2 kernels, aiming to hold CPU generation
   near ~50 tok/s at 4k instead of 20.

1. Minimal C++ engine for MiniCPM5 only: GGUF loader, Q4_K/Q6_K weights, tokenizer, KV cache. Verify the output
   matches llama.cpp token-for-token.
2. Prompt path: repack Q4_K into an interleaved layout, AVX-VNNI int8 GEMM, hybrid-core scheduler
   (P-cores take bigger tiles, dynamic work stealing).
3. Generation path: fused bandwidth-optimal GEMV (already ~ceiling) + speculative verification of k tokens per pass,
   using n-gram drafting first, then a tiny draft model if one exists.
4. Benchmark against this table after every step.
