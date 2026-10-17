#include "llama.h"
#include "ggml-backend.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s model.gguf cpu|none|layer|row [tokens] [logits.bin] [device]\n", argv[0]);
        return 1;
    }
    const char * mode = argv[2];
    bool cpu = strcmp(mode, "cpu") == 0;
    if (!cpu && strcmp(mode, "none") && strcmp(mode, "layer") && strcmp(mode, "row")) return 1;
    int predict = argc > 3 ? atoi(argv[3]) : 8;
    if (predict < 1 || predict > 64) return 1;
    int device = argc > 5 ? atoi(argv[5]) : 0;
    if (device < 0 || device > 1 || (strcmp(mode, "none") && device)) return 1;
    llama_backend_init();
    ggml_backend_dev_t devices[3] = {nullptr, nullptr, nullptr};
    if (!cpu) {
        devices[0] = ggml_backend_dev_by_name(device ? "KUNLUN1" : "KUNLUN0");
        if (!devices[0]) return 1;
        if (strcmp(mode, "none")) {
            devices[1] = ggml_backend_dev_by_name("KUNLUN1");
            if (!devices[1]) return 1;
        }
    }
    float split[2] = {1, 1};
    auto mp = llama_model_default_params();
    mp.devices = devices;
    mp.n_gpu_layers = cpu ? 0 : 99;
    mp.main_gpu = 0;
    mp.tensor_split = split;
    mp.split_mode = !strcmp(mode, "row") ? LLAMA_SPLIT_MODE_ROW : !strcmp(mode, "layer") ? LLAMA_SPLIT_MODE_LAYER : LLAMA_SPLIT_MODE_NONE;
    auto * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    auto cp = llama_context_default_params();
    cp.n_ctx = 256; cp.n_batch = 8; cp.n_ubatch = 8;
    cp.n_threads = 6; cp.n_threads_batch = 6;
    cp.offload_kqv = std::getenv("GGML_KUNLUN_CPU_KV") == nullptr;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.no_perf = false;
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;
    auto * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> teacher;
    if (const char * ids = std::getenv("GGML_KUNLUN_TEACHER_TOKENS")) {
        while (*ids) {
            if (*ids == ' ' || *ids == ',') { ++ids; continue; }
            char * end = nullptr;
            long value = std::strtol(ids,&end,10);
            if (end == ids || value < 0 || value >= llama_vocab_n_tokens(vocab)) return 1;
            teacher.push_back(value); ids=end;
        }
        if ((int)teacher.size() < predict) return 1;
        fprintf(stderr,"TEACHER_FORCED positions=%d\n",predict);
    }
    const char * prompt = "The capital of France is";
    int nt = -llama_tokenize(vocab, prompt, strlen(prompt), nullptr, 0, true, true);
    if (nt < 1 || nt > 8) return 1;
    std::vector<llama_token> tokens(nt), generated;
    if (llama_tokenize(vocab, prompt, strlen(prompt), tokens.data(), nt, true, true) != nt) return 1;
    auto * batch = llama_batch_ext_init(ctx);
    auto set_batch = [&](const llama_token * ids, int count, int position) {
        llama_batch_ext_clear(batch);
        for (int i = 0; i < count; ++i) {
            int index = llama_batch_ext_add_token(batch, 0, ids[i]);
            llama_pos pos = position + i;
            llama_batch_ext_set_pos(batch, index, &pos);
        }
        llama_batch_ext_set_output_logits(batch, count - 1, true);
    };
    FILE * dump = argc > 4 ? fopen(argv[4], "wb") : nullptr;
    if (argc > 4 && !dump) return 1;
    set_batch(tokens.data(), nt, 0);
    int position = nt;
    printf("%s", prompt); fflush(stdout);
    for (int i = 0; i < predict; ++i) {
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)) return 1;
        float * logits = llama_get_logits_ith(ctx, -1);
        int nv = llama_vocab_n_tokens(vocab);
        llama_token best = 0;
        for (int j = 0; j < nv; ++j) {
            if (!std::isfinite(logits[j])) { fprintf(stderr, "nonfinite logits\n"); return 1; }
            if (logits[j] > logits[best]) best = j;
        }
        if (dump && fwrite(logits, sizeof(float), nv, dump) != (size_t)nv) return 1;
        generated.push_back(best);
        if (teacher.empty() && llama_vocab_is_eog(vocab, best)) break;
        char piece[256];
        int size = llama_token_to_piece(vocab, best, piece, sizeof(piece), 0, true);
        if (size < 0) return 1;
        fwrite(piece, 1, size, stdout); fflush(stdout);
        llama_token next = teacher.empty() ? best : teacher[i];
        set_batch(&next, 1, position++);
    }
    if (dump) fclose(dump);
    printf("\n");
    fprintf(stderr, "TOKENS mode=%s", mode);
    for (auto token : generated) fprintf(stderr, " %d", token);
    fprintf(stderr, "\n");
    llama_perf_context_print(ctx);
    llama_batch_ext_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
