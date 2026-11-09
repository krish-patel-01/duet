"""A/B check of the fast decode attention kernel through llama-server.

Runs the same long prompt with greedy decoding three ways on one build:
  kernel  - our decode kernel (F16 KV cache)
  stock   - GGML_FA_DECODE=0, llama.cpp's own path (F16 KV cache)
  ref     - GGML_FA_DECODE=0 with an F32 KV cache: the most precise reference
and compares kernel and stock against ref: how many greedy tokens match, and the largest
difference in top-5 token probabilities over the shared prefix. Also reports speed.

usage: python check_kernel_server.py SERVER_EXE MODEL.gguf [prompt_repeats] [n_predict] [extra server args...]
"""
import json, math, os, subprocess, sys, time, urllib.request

exe, model = sys.argv[1], sys.argv[2]
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 40
n_predict = int(sys.argv[4]) if len(sys.argv) > 4 else 64
extra = sys.argv[5:]
PORT = 39290
para = ("The history of the printing press shows how a single technical change can reshape a society. "
        "Before movable type, books were copied by hand, which made them rare and expensive. "
        "After Gutenberg, pamphlets, newspapers and textbooks spread ideas quickly across Europe. ")
prompt = "Read the notes below and then continue the essay.\n\n" + para * reps + "\nContinue the essay in your own words:"


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def prob(t):
    return t["prob"] if "prob" in t else math.exp(t["logprob"])


def run(kernel_on, kv_args=(), server=None):
    env = dict(os.environ)
    env["GGML_FA_DECODE"] = "1" if kernel_on else "0"
    p = subprocess.Popen([server or exe, "-m", model, "--host", "127.0.0.1", "--port", str(PORT), "-c", "8192",
                          "--load-mode", "none", "--reasoning", "off", *kv_args] + extra,
                         env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(240):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                time.sleep(0.5)
        post("/completion", {"prompt": "Hi", "n_predict": 4, "temperature": 0})  # warm-up
        r = post("/completion", {"prompt": prompt, "n_predict": n_predict, "temperature": 0, "top_k": 1,
                                 "n_probs": 5, "cache_prompt": False, "ignore_eos": True})
        cps = r["completion_probabilities"]
        steps = [{t["token"]: prob(t) for t in (s.get("top_probs") or s.get("top_logprobs"))} for s in cps]
        return [s["token"] for s in cps], steps, r["timings"]
    finally:
        p.terminate()
        p.wait(30)


def compare(a, b):
    (at, ap, _), (bt, bp, _) = a, b
    n = 0
    for x, y in zip(at, bt):
        if x != y:
            break
        n += 1
    d = 0.0
    for x, y in zip(ap[:n], bp[:n]):
        for tok in set(x) & set(y):
            d = max(d, abs(x[tok] - y[tok]))
    return n, d


kern = run(True)
stock = run(False)
kern2 = run(True)
# Reference: an independent implementation - the GPU build (REF_EXE) with the whole model on the
# GPU - or, without REF_EXE, the CPU stock path with an F32 KV cache.
ref_exe = os.environ.get("REF_EXE")
ref = run(False, ("-ngl", "99"), ref_exe) if ref_exe else run(False, ("-ctk", "f32", "-ctv", "f32"))
print(f"model: {os.path.basename(model)}   prompt tokens: {kern[2]['prompt_n']}, generated: {len(kern[0])}")
print(f"  reference: {'GPU ' + os.path.basename(os.path.dirname(ref_exe)) if ref_exe else 'CPU F32 KV'}")
for name, r in (("kernel ", kern), ("kernel2", kern2), ("stock  ", stock)):
    n, d = compare(r, ref)
    print(f"  {name} vs reference: identical greedy tokens {n}/{len(r[0])}, max top-5 prob diff {d:.4f}"
          f"   gen {r[2]['predicted_per_second']:.1f} tok/s")
for a, b, name in ((kern, stock, "kernel vs stock  "), (kern, kern2, "kernel vs kernel2")):
    n, d = compare(a, b)
    print(f"  {name}: identical {n}/{len(a[0])}, max diff {d:.4f}")
