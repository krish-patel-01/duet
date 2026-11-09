// hybrid.cpp - CPU + Iris Xe (Vulkan) hybrid scheduler on top of llama.cpp.
//
// The model is loaded twice from the same GGUF: once on the GPU, once on the CPU.
// - Big chunks of new text (prompts) are read on the GPU, which is ~3x faster at it.
// - Each generated token runs on whichever device is faster at the current context
//   length: CPU for short contexts, GPU past the switch point (measured with --calibrate).
// - The two KV caches are kept in step lazily. A device that is far behind gets the other
//   device's KV cache copied over; one that is only a few tokens behind just re-reads them.

#define NOMINMAX
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <cmath>

#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static double secs_since(clk::time_point t0) {
    return std::chrono::duration<double>(clk::now() - t0).count();
}

[[noreturn]] static void die(const char* msg) {
    fprintf(stderr, "error: %s\n", msg);
    exit(1);
}

// ---------------------------------------------------------------- arguments

static bool g_no_think = false;  // --no-think: start the reply with an empty <think> block

struct Args {
    std::string model;
    std::string prompt;
    std::string mode = "hybrid";  // hybrid | cpu | gpu
    int n_predict = 256;
    int n_ctx = 8192;
    // context length where generation moves CPU -> GPU. With the fast CPU attention kernel the
    // CPU stayed faster at every length calibrated (up to 4096), so the default never switches.
    int switch_at = INT_MAX;
    bool switch_at_set = false;
    int resync_max = 32;          // a device this many tokens behind re-reads them itself
    int cpu_threads = 6;
    int cpu_threads_batch = 16;
    bool chat = false;
    bool interactive = false;
    bool calibrate = false;
    bool selftest = false;
    std::string logit_trace;      // --logit-trace FILE: teacher-forced per-step logits dump
    int bench_prompt = 0;         // >0: benchmark with a synthetic prompt of this many tokens
    bool ignore_eos = false;
    bool verbose = false;
    float temp = 0.0f;            // 0 = greedy
    int top_k = 40;
    float top_p = 0.95f;
    uint32_t seed = 42;
};

static void usage() {
    fprintf(stderr,
        "usage: hybrid -m model.gguf [options]\n"
        "  -p TEXT            prompt\n"
        "  -f FILE            read the prompt from a file\n"
        "  -n N               tokens to generate (default 256)\n"
        "  -c N               context size (default 8192)\n"
        "  --mode M           hybrid | cpu | gpu (default hybrid)\n"
        "  --switch-at N      context length where generation moves to the GPU (default: never)\n"
        "  --calibrate        measure the switch point on this machine at startup\n"
        "  --resync-max N     re-read up to N missing tokens instead of copying the KV cache (default 32)\n"
        "  -t N               CPU threads for generation (default 6)\n"
        "  -tb N              CPU threads for prompt reading (default 16)\n"
        "  --chat             wrap the prompt in the model's chat template\n"
        "  -i, --interactive  multi-turn chat on the console\n"
        "  --bench N          benchmark: synthetic N-token prompt, then -n tokens (ignores EOS)\n"
        "  --temp F --top-k N --top-p F --seed N   sampling (default greedy)\n"
        "  --ignore-eos       keep generating past end-of-sequence\n"
        "  --no-think         skip the model's thinking block in chat mode\n"
        "  -v                 show llama.cpp logs\n");
    exit(1);
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) usage();
            return argv[++i];
        };
        if (s == "-m") a.model = next();
        else if (s == "-p") a.prompt = next();
        else if (s == "-f") {
            FILE* f = fopen(next(), "rb");
            if (!f) die("cannot open prompt file");
            char b[65536];
            size_t n;
            while ((n = fread(b, 1, sizeof(b), f)) > 0) a.prompt.append(b, n);
            fclose(f);
        }
        else if (s == "-n") a.n_predict = atoi(next());
        else if (s == "-c") a.n_ctx = atoi(next());
        else if (s == "--mode") a.mode = next();
        else if (s == "--switch-at") { a.switch_at = atoi(next()); a.switch_at_set = true; }
        else if (s == "--resync-max") a.resync_max = atoi(next());
        else if (s == "-t") a.cpu_threads = atoi(next());
        else if (s == "-tb") a.cpu_threads_batch = atoi(next());
        else if (s == "--chat") a.chat = true;
        else if (s == "-i" || s == "--interactive") { a.interactive = true; a.chat = true; }
        else if (s == "--calibrate") a.calibrate = true;
        else if (s == "--selftest") a.selftest = true;
        else if (s == "--logit-trace") a.logit_trace = next();
        else if (s == "--bench") { a.bench_prompt = atoi(next()); a.ignore_eos = true; }
        else if (s == "--ignore-eos") a.ignore_eos = true;
        else if (s == "--temp") a.temp = (float)atof(next());
        else if (s == "--top-k") a.top_k = atoi(next());
        else if (s == "--top-p") a.top_p = (float)atof(next());
        else if (s == "--seed") a.seed = (uint32_t)atoi(next());
        else if (s == "-v") a.verbose = true;
        else if (s == "--no-think") g_no_think = true;
        else usage();
    }
    if (a.model.empty()) usage();
    if (a.mode != "hybrid" && a.mode != "cpu" && a.mode != "gpu") usage();
    return a;
}

// ---------------------------------------------------------------- OpenMP spin control
//
// The CPU backend runs on OpenMP. After each job its worker threads spin for ~200 ms before
// sleeping. GPU steps still run a tiny CPU graph (token embedding lookup), so during GPU
// generation those workers never get to sleep: they burn the power budget the iGPU shares
// with the CPU and GPU steps slow from ~25 to ~50 ms. So: spin while the CPU generates
// (fast wake-up), no spin while the GPU generates.

typedef void (*kmp_set_blocktime_fn)(int);
static kmp_set_blocktime_fn g_set_blocktime = nullptr;
static int g_blocktime = -1;

static void omp_spin_init() {
    HMODULE h = GetModuleHandleA("libomp.dll");
    if (!h) h = LoadLibraryA("libomp.dll");
    if (h) g_set_blocktime = (kmp_set_blocktime_fn)GetProcAddress(h, "kmp_set_blocktime");
}

static void omp_spin(bool on) {
    int ms = on ? 200 : 0;
    if (g_set_blocktime && ms != g_blocktime) { g_set_blocktime(ms); g_blocktime = ms; }
}

// ---------------------------------------------------------------- engines

struct Engine {
    const char* name;
    bool is_gpu;
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    int n_synced = 0;     // how many conversation tokens this KV cache holds
    int logits_pos = -1;  // the logits buffer predicts the token after this many tokens
    // stats
    double t_prefill = 0, t_gen = 0;
    long n_prefill = 0, n_gen = 0;
    long n_sampled = 0;   // generated tokens whose logits came from this device
    std::vector<float> step_ms;
};

struct Stats {
    int handoffs = 0;
    double t_handoff = 0;
    double mb_handoff = 0;
};

class Scheduler {
public:
    Engine cpu{"CPU", false};
    Engine gpu{"GPU", true};
    bool has_cpu = false, has_gpu = false;
    std::vector<llama_token> toks;  // the whole conversation
    int switch_at = 700;
    int resync_max = 32;
    Stats st;

    void init_batch(int n) {
        n_batch = n;
        batch = llama_batch_init(n, 0, 1);
    }
    ~Scheduler() { if (n_batch) llama_batch_free(batch); }

    Engine* other(Engine* e) {
        if (e == &cpu) return has_gpu ? &gpu : nullptr;
        return has_cpu ? &cpu : nullptr;
    }

    Engine& decode_device(int L) {
        if (!has_gpu) return cpu;
        if (!has_cpu) return gpu;
        return L >= switch_at ? gpu : cpu;
    }

    // Drop everything at position >= keep from an engine's KV cache.
    void trim(Engine& e, int keep) {
        llama_memory_seq_rm(llama_get_memory(e.ctx), 0, keep, -1);
        e.n_synced = std::min(e.n_synced, keep);
        if (e.logits_pos > keep) e.logits_pos = -1;
    }

    void clear_all() {
        for (Engine* e : {&cpu, &gpu}) {
            if (!e->ctx) continue;
            llama_memory_clear(llama_get_memory(e->ctx), true);
            e->n_synced = 0;
            e->logits_pos = -1;
        }
        toks.clear();
    }

    // Feed toks[e.n_synced .. to) through engine e. With want_logits, the logits for
    // position `to` (the next token) end up in e's buffer.
    void run(Engine& e, int to, bool want_logits) {
        if (e.n_synced > to) trim(e, to);
        if (want_logits && e.n_synced == to && e.logits_pos != to) trim(e, to - 1);
        int from = e.n_synced;
        if (from >= to) return;
        omp_spin(!e.is_gpu);
        while (from < to) {
            int n = std::min(n_batch, to - from);
            batch.n_tokens = n;
            for (int i = 0; i < n; i++) {
                batch.token[i] = toks[from + i];
                batch.pos[i] = from + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = (want_logits && from + i == to - 1) ? 1 : 0;
            }
            auto t0 = clk::now();
            if (llama_decode(e.ctx, batch) != 0) die("llama_decode failed (context full?)");
            llama_synchronize(e.ctx);
            double dt = secs_since(t0);
            if (n == 1) { e.t_gen += dt; e.n_gen += 1; e.step_ms.push_back((float)(dt * 1000)); }
            else        { e.t_prefill += dt; e.n_prefill += n; }
            from += n;
        }
        e.n_synced = to;
        e.logits_pos = want_logits ? to : -1;
    }

    // Copy src's KV cache into dst, replacing what dst had.
    void handoff(Engine& src, Engine& dst) {
        auto t0 = clk::now();
        size_t sz = llama_state_seq_get_size(src.ctx, 0);
        if (buf.size() < sz) buf.resize(sz);
        size_t got = llama_state_seq_get_data(src.ctx, buf.data(), sz, 0);
        // A full clear (not just seq_rm) resets the cache's write head to cell 0. Otherwise the
        // restored cells land after the old head, every later step attends over that larger
        // window, and GPU generation drops from ~40 to ~18 tok/s.
        llama_memory_clear(llama_get_memory(dst.ctx), false);
        if (llama_state_seq_set_data(dst.ctx, buf.data(), got, 0) == 0) die("KV cache hand-off failed");
        dst.n_synced = src.n_synced;
        dst.logits_pos = -1;
        st.handoffs++;
        st.t_handoff += secs_since(t0);
        st.mb_handoff += got / 1e6;
    }

    // Make some engine hold the next-token logits for toks[0..L) and return it.
    Engine& step(int L) {
        if (has_gpu && gpu.n_synced == L && gpu.logits_pos == L) return gpu;
        if (has_cpu && cpu.n_synced == L && cpu.logits_pos == L) return cpu;

        Engine& d = decode_device(L);
        Engine* o = other(&d);
        if (o && L - d.n_synced > resync_max) {
            if (L - o->n_synced <= resync_max) {
                handoff(*o, d);                // the other device is current: take its cache
            } else if (o->is_gpu) {
                run(*o, L, true);              // a big unread chunk: the GPU reads it
                return *o;
            }
            // else d is the GPU and the CPU is far behind too: the GPU reads it below
        }
        run(d, L, true);
        return d;
    }

    // Measure single-token speed of both devices at several context lengths and
    // return the context length where the GPU becomes faster.
    int calibrate(const std::vector<llama_token>& filler, bool print) {
        const int depths[] = {4096, 3072, 2048, 1536, 1024, 768, 512, 256, 128};
        const int reps = 12;
        clear_all();
        toks.assign(filler.begin(), filler.begin() + depths[0]);
        run(gpu, depths[0], false);
        handoff(gpu, cpu);
        st = Stats{};
        std::vector<double> tc, tg;
        for (int d : depths) {
            trim(cpu, d);
            trim(gpu, d);
            double r[2];
            Engine* es[2] = {&cpu, &gpu};
            for (int k = 0; k < 2; k++) {
                Engine& e = *es[k];
                double best = 1e9;
                for (int rep = 0; rep < 2; rep++) {
                    auto t0 = clk::now();
                    for (int j = 0; j < reps; j++) decode_one(e, filler[d + j], d + j);
                    best = std::min(best, secs_since(t0));
                    trim(e, d);
                }
                r[k] = reps / best;
            }
            tc.push_back(r[0]);
            tg.push_back(r[1]);
            if (print) fprintf(stderr, "  context %5d: CPU %5.1f tok/s   GPU %5.1f tok/s\n", d, r[0], r[1]);
        }
        clear_all();
        for (Engine* e : {&cpu, &gpu}) { e->t_gen = e->t_prefill = 0; e->n_gen = e->n_prefill = 0; }
        // depths are descending; find the crossover going up from the smallest
        int n = (int)tc.size();
        if (tg[n - 1] >= tc[n - 1]) return 0;  // GPU faster everywhere
        for (int i = n - 1; i > 0; i--) {
            if (tg[i - 1] >= tc[i - 1]) {
                double a = tc[i] - tg[i], b = tc[i - 1] - tg[i - 1];  // a > 0, b <= 0
                return depths[i] + (int)((depths[i - 1] - depths[i]) * a / (a - b));
            }
        }
        // CPU faster everywhere we measured. Time per token grows about linearly with context, so
        // extend a line through the two longest measurements of each device to find where they cross.
        double tc0 = 1 / tc[1], tc1 = 1 / tc[0], tg0 = 1 / tg[1], tg1 = 1 / tg[0];
        double dd = depths[0] - depths[1];
        double bc = (tc1 - tc0) / dd, bg = (tg1 - tg0) / dd;  // seconds per token, per context token
        if (bc <= bg) return INT_MAX;                        // CPU never falls behind
        return depths[0] + (int)((tc1 - tg1) < 0 ? (tg1 - tc1) / (bc - bg) : 0);
    }

private:
    llama_batch batch{};
    int n_batch = 0;
    std::vector<uint8_t> buf;

    void decode_one(Engine& e, llama_token t, int pos) {
        omp_spin(!e.is_gpu);
        batch.n_tokens = 1;
        batch.token[0] = t;
        batch.pos[0] = pos;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        if (llama_decode(e.ctx, batch) != 0) die("llama_decode failed during calibration");
        llama_synchronize(e.ctx);
    }
};

// ---------------------------------------------------------------- helpers

static bool g_verbose = false;
static void log_cb(ggml_log_level level, const char* text, void*) {
    if (g_verbose || level == GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
}

static std::string exe_dir() {
    char p[MAX_PATH];
    GetModuleFileNameA(nullptr, p, MAX_PATH);
    std::string s = p;
    return s.substr(0, s.find_last_of("\\/"));
}

static std::vector<llama_token> tokenize(const llama_vocab* v, const std::string& s, bool add_special) {
    int n = -llama_tokenize(v, s.c_str(), (int)s.size(), nullptr, 0, add_special, true);
    std::vector<llama_token> out(n);
    if (llama_tokenize(v, s.c_str(), (int)s.size(), out.data(), n, add_special, true) < 0) die("tokenize failed");
    return out;
}

static std::string piece(const llama_vocab* v, llama_token t) {
    char b[256];
    int n = llama_token_to_piece(v, t, b, sizeof(b), 0, false);
    return n > 0 ? std::string(b, n) : std::string();
}

struct Msg { std::string role, content; };

static std::string format_chat(const llama_model* m, const std::vector<Msg>& msgs, bool add_ass) {
    std::vector<llama_chat_message> cm;
    for (auto& x : msgs) cm.push_back({x.role.c_str(), x.content.c_str()});
    const char* tmpl = llama_model_chat_template(m, nullptr);
    // llama.cpp's built-in matcher mistakes MiniCPM5's ChatML template for the old MiniCPM one,
    // so any template that uses <|im_start|> is formatted as ChatML here.
    if (tmpl && !strstr(tmpl, "<|im_start|>")) {
        std::vector<char> b(4096);
        int n = llama_chat_apply_template(tmpl, cm.data(), cm.size(), add_ass, b.data(), (int)b.size());
        if (n > (int)b.size()) {
            b.resize(n);
            n = llama_chat_apply_template(tmpl, cm.data(), cm.size(), add_ass, b.data(), (int)b.size());
        }
        if (n >= 0) return std::string(b.data(), n);
    }
    // ChatML (BOS is added by the tokenizer)
    // With --no-think, past assistant turns keep the empty think block they were generated after,
    // so the formatted history matches the tokens that were actually fed.
    const std::string stub = g_no_think ? "<think>\n\n</think>\n\n" : "";
    std::string s;
    for (auto& x : msgs) {
        s += "<|im_start|>" + x.role + "\n";
        if (x.role == "assistant") s += stub;
        s += x.content + "<|im_end|>\n";
    }
    if (add_ass) s += "<|im_start|>assistant\n" + stub;
    return s;
}

// Tokenize the start of a chat. Templates that begin with {{ bos_token }} need BOS even when
// the tokenizer would not add it on its own (MiniCPM5 is one of these).
static std::vector<llama_token> tokenize_chat_start(const llama_model* m, const llama_vocab* v,
                                                    const std::string& s) {
    auto t = tokenize(v, s, true);
    const char* tmpl = llama_model_chat_template(m, nullptr);
    llama_token bos = llama_vocab_bos(v);
    if (tmpl && strstr(tmpl, "bos_token") && bos >= 0 && (t.empty() || t[0] != bos)) t.insert(t.begin(), bos);
    return t;
}

static llama_model* load_model(const Args& a, ggml_backend_dev_t gpu_dev) {
    llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t devs[2] = {gpu_dev, nullptr};  // {nullptr, ...} = no devices = CPU only
    mp.devices = devs;
    mp.n_gpu_layers = gpu_dev ? 999 : 0;
    llama_model* m = llama_model_load_from_file(a.model.c_str(), mp);
    if (!m) die("failed to load model");
    return m;
}

static llama_context* make_ctx(const Args& a, llama_model* m, bool gpu) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = a.n_ctx;
    cp.n_batch = 2048;
    cp.n_ubatch = 512;
    cp.n_seq_max = 1;
    // the GPU context only runs tiny CPU ops (embedding lookup), so one thread is enough
    cp.n_threads = gpu ? 1 : a.cpu_threads;
    cp.n_threads_batch = gpu ? 1 : a.cpu_threads_batch;
    // Both caches must use the same layout so they can be handed back and forth.
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.offload_kqv = gpu;
    cp.op_offload = gpu;
    cp.no_perf = true;
    llama_context* c = llama_init_from_model(m, cp);
    if (!c) die("failed to create context");
    return c;
}

// The Vulkan backend builds GPU pipelines the first time it sees each batch shape, which
// can stall the first real prompt for seconds. Run the common shapes once up front.
static void warmup(Scheduler& s, const std::vector<llama_token>& filler) {
    const int sizes[] = {512, 64, 16, 8, 5, 3, 2, 1};
    for (Engine* e : {&s.cpu, &s.gpu}) {
        if (!e->ctx) continue;
        for (int n : sizes) {
            if (!e->is_gpu && n != 64 && n != 1) continue;
            s.clear_all();
            s.toks.assign(filler.begin(), filler.begin() + n);
            s.run(*e, n, true);
        }
    }
    s.clear_all();
    for (Engine* e : {&s.cpu, &s.gpu}) { e->t_gen = e->t_prefill = 0; e->n_gen = e->n_prefill = 0; }
    s.st = Stats{};
}

static std::vector<llama_token> make_filler(const llama_vocab* v, int n) {
    const char* para =
        "The history of computing is a story of steady improvement in speed, memory and cost. "
        "Early machines filled rooms and used vacuum tubes; later ones used transistors, then "
        "integrated circuits, and finally microprocessors containing billions of switches. ";
    std::string text;
    std::vector<llama_token> t;
    for (int k = 1; (int)t.size() < n; k++) {
        text += "Section " + std::to_string(k) + ". " + para;
        if (k % 8 == 0) t = tokenize(v, text, true);
    }
    t = tokenize(v, text, true);
    t.resize(n);
    return t;
}

// ---------------------------------------------------------------- main

int main(int argc, char** argv) {
    SetConsoleOutputCP(CP_UTF8);
    Args a = parse_args(argc, argv);
    g_verbose = a.verbose;
    llama_log_set(log_cb, nullptr);
    ggml_backend_load_all_from_path(exe_dir().c_str());
    llama_backend_init();
    omp_spin_init();
    if (a.verbose) fprintf(stderr, "[OpenMP spin control: %s]\n", g_set_blocktime ? "on" : "not available");

    ggml_backend_dev_t gpu_dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        auto t = ggml_backend_dev_type(d);
        if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) { gpu_dev = d; break; }
    }
    if (a.mode != "cpu" && !gpu_dev) die("no Vulkan GPU found (use --mode cpu)");

    auto t_load = clk::now();
    Scheduler s;
    s.has_gpu = a.mode != "cpu";
    s.has_cpu = a.mode != "gpu";
    s.resync_max = a.resync_max;
    if (s.has_gpu) { s.gpu.model = load_model(a, gpu_dev); s.gpu.ctx = make_ctx(a, s.gpu.model, true); }
    if (s.has_cpu) { s.cpu.model = load_model(a, nullptr); s.cpu.ctx = make_ctx(a, s.cpu.model, false); }
    const llama_model* model = s.has_gpu ? s.gpu.model : s.cpu.model;
    const llama_vocab* vocab = llama_model_get_vocab(model);
    s.init_batch(2048);

    std::vector<llama_token> filler = make_filler(vocab, std::max(4200, a.bench_prompt + a.n_predict + 16));
    warmup(s, filler);
    fprintf(stderr, "[mode %s | GPU: %s | loaded in %.1fs]\n", a.mode.c_str(),
            s.has_gpu ? ggml_backend_dev_description(gpu_dev) : "off", secs_since(t_load));

    s.switch_at = a.switch_at;
    if (a.mode == "hybrid" && a.calibrate) {
        fprintf(stderr, "[calibrating generation speed vs context length]\n");
        int x = s.calibrate(filler, true);
        if (!a.switch_at_set) s.switch_at = x;
        fprintf(stderr, "[measured switch point: %s]\n",
                x == INT_MAX ? "never (CPU always faster)" : std::to_string(x).c_str());
    }
    if (a.mode == "hybrid") {
        if (s.switch_at == INT_MAX) fprintf(stderr, "[generation stays on the CPU]\n");
        else fprintf(stderr, "[generation moves CPU -> GPU at context %d]\n", s.switch_at);
    }

    // sampler
    llama_sampler* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (a.temp <= 0) {
        llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
    } else {
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(a.top_k));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(a.top_p, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(a.temp));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(a.seed));
    }

    // Generate up to n tokens after s.toks. Returns the generated text.
    auto generate = [&](int n, bool stream, double* ttft, std::vector<llama_token>* out_ids) {
        std::string text;
        auto t0 = clk::now();
        for (int i = 0; i < n; i++) {
            int L = (int)s.toks.size();
            if (L >= a.n_ctx - 1) { fprintf(stderr, "\n[context full]\n"); break; }
            Engine& e = s.step(L);
            llama_token t = llama_sampler_sample(smpl, e.ctx, -1);
            if (i == 0 && ttft) *ttft = secs_since(t0);
            if (llama_vocab_is_eog(vocab, t) && !a.ignore_eos) {
                if (a.verbose) fprintf(stderr, "[end token %d '%s']\n", t, piece(vocab, t).c_str());
                break;
            }
            e.n_sampled++;
            s.toks.push_back(t);
            if (out_ids) out_ids->push_back(t);
            std::string p = piece(vocab, t);
            text += p;
            if (stream) { fputs(p.c_str(), stdout); fflush(stdout); }
        }
        return text;
    };

    auto print_stats = [&](int n_prompt, double ttft, double total) {
        long n_out = s.cpu.n_sampled + s.gpu.n_sampled;
        double gen_time = total - ttft;
        fprintf(stderr, "\n[prompt %d tok, first token after %.2fs (%.0f tok/s)", n_prompt, ttft,
                ttft > 0 ? n_prompt / ttft : 0.0);
        fprintf(stderr, " | generated %ld tok in %.2fs = %.1f tok/s", n_out, total,
                n_out > 1 ? (n_out - 1) / gen_time : 0.0);
        if (s.has_cpu && s.has_gpu)
            fprintf(stderr, " (CPU %ld, GPU %ld) | hand-offs %d, %.0f ms, %.0f MB", s.cpu.n_sampled,
                    s.gpu.n_sampled, s.st.handoffs, s.st.t_handoff * 1000, s.st.mb_handoff);
        fprintf(stderr, "]\n");
        if (a.verbose) {
            for (Engine* e : {&s.cpu, &s.gpu}) {
                if (!e->ctx) continue;
                fprintf(stderr, "[%s: read %ld tok in %.2fs, generated %ld single steps in %.2fs = %.1f tok/s]\n",
                        e->name, e->n_prefill, e->t_prefill, e->n_gen, e->t_gen,
                        e->n_gen ? e->n_gen / e->t_gen : 0.0);
            }
        }
    };

    if (a.selftest) {
        // Checks that a KV cache handed GPU -> CPU gives the same next-token scores as the CPU
        // computing the whole prompt itself (up to normal CPU-vs-GPU rounding differences).
        if (!(s.has_cpu && s.has_gpu)) die("--selftest needs --mode hybrid");
        std::string p = a.prompt.empty() ? "The quick brown fox jumps over the lazy dog." : a.prompt;
        if (a.chat) p = format_chat(model, {{"user", p}}, true);
        auto pt = a.chat ? tokenize_chat_start(model, vocab, p) : tokenize(vocab, p, true);
        int L = (int)pt.size();
        int nv = llama_vocab_n_tokens(vocab);
        auto grab = [&](Engine& e) {
            float* lg = llama_get_logits_ith(e.ctx, -1);
            return std::vector<float>(lg, lg + nv);
        };
        auto cmp = [&](const char* label, const std::vector<float>& x, const std::vector<float>& y) {
            double md = 0, dot = 0, nx = 0, ny = 0;
            int ax = 0, ay = 0;
            for (int i = 0; i < nv; i++) {
                md = std::max(md, (double)fabsf(x[i] - y[i]));
                dot += x[i] * y[i]; nx += x[i] * x[i]; ny += y[i] * y[i];
                if (x[i] > x[ax]) ax = i;
                if (y[i] > y[ay]) ay = i;
            }
            fprintf(stderr, "  %-34s max diff %.4f  cosine %.6f  top token %s\n", label, md,
                    dot / sqrt(nx * ny), ax == ay ? "same" : "DIFFERENT");
            return ax == ay;
        };
        s.clear_all(); s.toks = pt;
        s.run(s.gpu, L, true);                 auto g = grab(s.gpu);
        s.run(s.cpu, L, true);                 auto c = grab(s.cpu);       // CPU reads the prompt itself
        s.trim(s.cpu, 0);
        s.handoff(s.gpu, s.cpu); s.run(s.cpu, L, true); auto h = grab(s.cpu); // CPU after hand-off
        // and the other direction: CPU cache -> GPU
        s.trim(s.gpu, 0);
        s.handoff(s.cpu, s.gpu); s.run(s.gpu, L, true); auto hg = grab(s.gpu);
        fprintf(stderr, "[selftest: %d prompt tokens]\n", L);
        cmp("GPU vs CPU (both read the prompt)", g, c);
        bool ok1 = cmp("CPU after GPU->CPU hand-off vs CPU", h, c);
        bool ok2 = cmp("GPU after CPU->GPU hand-off vs GPU", hg, g);
        fprintf(stderr, "[selftest %s]\n", ok1 && ok2 ? "PASSED" : "FAILED");
        return ok1 && ok2 ? 0 : 1;
    }

    if (!a.logit_trace.empty()) {
        // Teacher-forced trace: read --bench P filler tokens, then feed the next -n filler tokens one
        // at a time and save every step's logits. Two runs (e.g. GGML_FA_DECODE=0 vs 1) can then be
        // compared step by step with no sampling drift.
        int P = std::max(1, a.bench_prompt);
        if (P + a.n_predict + 1 > (int)filler.size() || P + a.n_predict + 1 > a.n_ctx) die("trace too long");
        int nv = llama_vocab_n_tokens(vocab);
        FILE* f = fopen(a.logit_trace.c_str(), "wb");
        if (!f) die("cannot open trace file");
        fwrite(&nv, sizeof(int), 1, f);
        s.toks.assign(filler.begin(), filler.begin() + P);
        for (int i = 0; i <= a.n_predict; i++) {
            Engine& e = s.step((int)s.toks.size());
            fwrite(llama_get_logits_ith(e.ctx, -1), sizeof(float), nv, f);
            s.toks.push_back(filler[P + i]);
        }
        fclose(f);
        fprintf(stderr, "[wrote %d steps of logits to %s]\n", a.n_predict + 1, a.logit_trace.c_str());
        return 0;
    }

    if (a.bench_prompt > 0) {
        s.toks.assign(filler.begin(), filler.begin() + a.bench_prompt);
        double ttft = 0;
        std::vector<llama_token> ids;
        auto t0 = clk::now();
        generate(a.n_predict, false, &ttft, &ids);
        double total = secs_since(t0);
        unsigned long long h = 1469598103934665603ULL;
        for (auto t : ids) h = (h ^ (unsigned)t) * 1099511628211ULL;
        long n_out = (long)ids.size();
        printf("BENCH mode=%s prompt=%d gen=%ld ttft=%.3f prompt_tps=%.1f gen_tps=%.2f total=%.3f "
               "cpu_tok=%ld gpu_tok=%ld handoffs=%d handoff_ms=%.1f hash=%016llx\n",
               a.mode.c_str(), a.bench_prompt, n_out, ttft, a.bench_prompt / ttft,
               (n_out - 1) / (total - ttft), total, s.cpu.n_sampled, s.gpu.n_sampled, s.st.handoffs,
               s.st.t_handoff * 1000, h);
        for (Engine* e : {&s.cpu, &s.gpu}) {
            if (!e->ctx) continue;
            printf("  %s: read %ld tok in %.2fs, %ld single steps in %.2fs = %.1f tok/s\n", e->name,
                   e->n_prefill, e->t_prefill, e->n_gen, e->t_gen, e->n_gen ? e->n_gen / e->t_gen : 0.0);
            if (a.verbose && !e->step_ms.empty()) {
                printf("    step ms:");
                for (size_t i = 0; i < e->step_ms.size(); i += std::max<size_t>(1, e->step_ms.size() / 25))
                    printf(" %.0f", e->step_ms[i]);
                printf("\n");
            }
        }
        return 0;
    }

    if (!a.interactive) {
        std::string p = a.prompt.empty() ? "Hello! Tell me about yourself." : a.prompt;
        if (a.chat) p = format_chat(model, {{"user", p}}, true);
        s.toks = a.chat ? tokenize_chat_start(model, vocab, p) : tokenize(vocab, p, true);
        int n_prompt = (int)s.toks.size();
        if (a.verbose) {
            fprintf(stderr, "[prompt tokens:");
            for (auto t : s.toks) fprintf(stderr, " %d'%s'", t, piece(vocab, t).c_str());
            fprintf(stderr, "]\n[eos %d eot %d bos %d]\n", llama_vocab_eos(vocab), llama_vocab_eot(vocab), llama_vocab_bos(vocab));
        }
        double ttft = 0;
        auto t0 = clk::now();
        generate(a.n_predict, true, &ttft, nullptr);
        print_stats(n_prompt, ttft, secs_since(t0));
        return 0;
    }

    // interactive multi-turn chat
    std::vector<Msg> msgs;
    std::string prev_formatted;
    fprintf(stderr, "[chat: type a message, empty line or /exit to quit, /reset to clear]\n");
    for (;;) {
        fprintf(stderr, "\n> ");
        std::string line;
        if (!std::getline(std::cin, line) || line.empty() || line == "/exit") break;
        if (line == "/reset") { s.clear_all(); msgs.clear(); prev_formatted.clear(); continue; }
        msgs.push_back({"user", line});
        std::string f = format_chat(model, msgs, true);
        std::string add = f.substr(std::min(prev_formatted.size(), f.size()));
        auto nt = s.toks.empty() ? tokenize_chat_start(model, vocab, add) : tokenize(vocab, add, false);
        s.toks.insert(s.toks.end(), nt.begin(), nt.end());
        for (Engine* e : {&s.cpu, &s.gpu}) e->n_sampled = 0;
        s.st = Stats{};
        double ttft = 0;
        auto t0 = clk::now();
        std::string reply = generate(a.n_predict, true, &ttft, nullptr);
        print_stats((int)nt.size(), ttft, secs_since(t0));
        msgs.push_back({"assistant", reply});
        prev_formatted = format_chat(model, msgs, false);
        // keep the token history in line with the template's closing tokens
        std::string tail = prev_formatted.substr(std::min(f.size() + reply.size(), prev_formatted.size()));
        auto tt = tokenize(vocab, tail, false);
        s.toks.insert(s.toks.end(), tt.begin(), tt.end());
    }
    llama_sampler_free(smpl);
    return 0;
}
